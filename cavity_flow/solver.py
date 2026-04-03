"""MAC method solver for lid-driven cavity flow."""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import torch

from cavity_flow.boundary import apply_boundary_conditions
from cavity_flow.poisson import solve_poisson_cg

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Return MPS device if available, otherwise CPU.

    Returns:
        A torch.device pointing to 'mps' on Apple Silicon, or 'cpu'.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _max_explicit_diffusion_dt(nx: int, ny: int, re: float) -> float:
    """Return the stable time-step upper bound for explicit diffusion.

    The viscous term in this solver is advanced explicitly, so the time step
    must satisfy the 2D FTCS diffusion stability condition:

        nu * dt * (1 / dx^2 + 1 / dy^2) <= 1 / 2

    where nu = 1 / Re, dx = 1 / nx, and dy = 1 / ny.

    Args:
        nx: Number of cells in the x direction.
        ny: Number of cells in the y direction.
        re: Reynolds number.

    Returns:
        Maximum stable explicit-diffusion time step.
    """
    dx = 1.0 / nx
    dy = 1.0 / ny
    nu = 1.0 / re
    return 1.0 / (2.0 * nu * ((1.0 / dx**2) + (1.0 / dy**2)))


@dataclasses.dataclass
class SolverConfig:
    """Configuration for the MAC cavity flow solver.

    Attributes:
        nx: Number of cells in the x direction.
        ny: Number of cells in the y direction.
        re: Reynolds number (Re = U * L / nu, where U=1, L=1).
        dt: Time step size.
        max_steps: Maximum number of time steps.
        convergence_tol: L-infinity norm threshold for steady-state detection.
        lid_velocity: Velocity of the moving top lid.
        poisson_tol: Convergence tolerance for the CG Poisson solver.
        poisson_max_iter: Maximum CG iterations per time step.
    """

    nx: int = 300
    ny: int = 300
    re: float = 100.0
    dt: float = 2.5e-4
    max_steps: int = 10000
    convergence_tol: float = 1e-6
    lid_velocity: float = 1.0
    poisson_tol: float = 1e-6
    poisson_max_iter: int = 2000

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(
                f"nx and ny must be positive integers; got nx={self.nx}, ny={self.ny}"
            )
        if self.re <= 0.0:
            raise ValueError(f"Reynolds number must be positive; got re={self.re}")
        if self.dt <= 0.0:
            raise ValueError(f"Time step must be positive; got dt={self.dt}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got max_steps={self.max_steps}")

        max_stable_dt = _max_explicit_diffusion_dt(self.nx, self.ny, self.re)
        if self.dt > max_stable_dt:
            raise ValueError(
                "Time step exceeds the explicit diffusion stability limit; "
                f"got dt={self.dt:.3e}, max_dt={max_stable_dt:.3e} "
                f"for nx={self.nx}, ny={self.ny}, re={self.re}"
            )


class CavityFlowSolver:
    """Solves lid-driven cavity flow on a staggered MAC grid.

    The domain is [0, 1] x [0, 1] discretised into (nx x ny) cells.

    Staggered-grid layout:
      p[i, j]  — pressure at cell centre,     shape (nx, ny)
      u[i, j]  — x-velocity at vertical face,  shape (nx+1, ny)
      v[i, j]  — y-velocity at horizontal face, shape (nx, ny+1)

    The implicit pressure correction step solves the Poisson equation
    using Conjugate Gradient running on a PyTorch MPS device (Apple
    Silicon Metal Performance Shaders) with automatic CPU fallback.

    Args:
        config: Solver configuration dataclass.
        device: Optional explicit torch.device override.
    """

    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config if config is not None else SolverConfig()
        self.device = device if device is not None else _get_device()

        nx, ny = self.config.nx, self.config.ny
        self.dx = 1.0 / nx
        self.dy = 1.0 / ny

        # Velocity and pressure fields (all on the target device)
        self.u = torch.zeros(nx + 1, ny, dtype=torch.float32, device=self.device)
        self.v = torch.zeros(nx, ny + 1, dtype=torch.float32, device=self.device)
        self.p = torch.zeros(nx, ny, dtype=torch.float32, device=self.device)

        # Apply initial boundary conditions (sets lid velocity on top)
        self.u, self.v = apply_boundary_conditions(
            self.u, self.v, self.config.lid_velocity
        )

        logger.info(
            "CavityFlowSolver initialised: nx=%d, ny=%d, Re=%.1f, device=%s",
            nx,
            ny,
            self.config.re,
            self.device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interpolate_u_to_v_faces(self) -> torch.Tensor:
        """Interpolate u from vertical faces to the v (horizontal) face centres.

        Returns:
            u interpolated to shape (nx, ny+1).
        """
        # u is (nx+1, ny); average adjacent columns to get (nx, ny)
        u_cell = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        # Pad in y to reach (nx, ny+1)
        return torch.cat(
            [u_cell[:, :1], 0.5 * (u_cell[:, :-1] + u_cell[:, 1:]), u_cell[:, -1:]], dim=1
        )

    def _interpolate_v_to_u_faces(self) -> torch.Tensor:
        """Interpolate v from horizontal faces to the u (vertical) face centres.

        Returns:
            v interpolated to shape (nx+1, ny).
        """
        # v is (nx, ny+1); average adjacent rows to get (nx, ny)
        v_cell = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        # Pad in x to reach (nx+1, ny)
        return torch.cat(
            [v_cell[:1, :], 0.5 * (v_cell[:-1, :] + v_cell[1:, :]), v_cell[-1:, :]], dim=0
        )

    def _advect_u(self) -> torch.Tensor:
        """Compute advective + diffusive RHS for u on interior faces.

        Uses upwind differencing for advection and second-order central
        differences for diffusion (viscous term scaled by 1/Re).

        Returns:
            du/dt contribution, shape (nx+1, ny).
        """
        dx, dy = self.dx, self.dy
        nu = 1.0 / self.config.re
        u = self.u
        v_at_u = self._interpolate_v_to_u_faces()

        # Advection: u * du/dx (upwind)
        dudx = torch.where(
            u > 0,
            (u - torch.roll(u, 1, dims=0)) / dx,
            (torch.roll(u, -1, dims=0) - u) / dx,
        )
        # Advection: v * du/dy (upwind)
        dudy = torch.where(
            v_at_u > 0,
            (u - torch.roll(u, 1, dims=1)) / dy,
            (torch.roll(u, -1, dims=1) - u) / dy,
        )

        # Diffusion: nu * (d²u/dx² + d²u/dy²)
        d2udx2 = (torch.roll(u, -1, dims=0) - 2.0 * u + torch.roll(u, 1, dims=0)) / dx ** 2
        d2udy2 = (torch.roll(u, -1, dims=1) - 2.0 * u + torch.roll(u, 1, dims=1)) / dy ** 2

        return -(u * dudx + v_at_u * dudy) + nu * (d2udx2 + d2udy2)

    def _advect_v(self) -> torch.Tensor:
        """Compute advective + diffusive RHS for v on interior faces.

        Returns:
            dv/dt contribution, shape (nx, ny+1).
        """
        dx, dy = self.dx, self.dy
        nu = 1.0 / self.config.re
        v = self.v
        u_at_v = self._interpolate_u_to_v_faces()

        # Advection: u * dv/dx (upwind)
        dvdx = torch.where(
            u_at_v > 0,
            (v - torch.roll(v, 1, dims=0)) / dx,
            (torch.roll(v, -1, dims=0) - v) / dx,
        )
        # Advection: v * dv/dy (upwind)
        dvdy = torch.where(
            v > 0,
            (v - torch.roll(v, 1, dims=1)) / dy,
            (torch.roll(v, -1, dims=1) - v) / dy,
        )

        # Diffusion: nu * (d²v/dx² + d²v/dy²)
        d2vdx2 = (torch.roll(v, -1, dims=0) - 2.0 * v + torch.roll(v, 1, dims=0)) / dx ** 2
        d2vdy2 = (torch.roll(v, -1, dims=1) - 2.0 * v + torch.roll(v, 1, dims=1)) / dy ** 2

        return -(u_at_v * dvdx + v * dvdy) + nu * (d2vdx2 + d2vdy2)

    def _compute_divergence(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute the discrete divergence of (u, v) at cell centres.

        Args:
            u: x-velocity, shape (nx+1, ny).
            v: y-velocity, shape (nx, ny+1).

        Returns:
            Divergence field, shape (nx, ny).
        """
        du_dx = (u[1:, :] - u[:-1, :]) / self.dx
        dv_dy = (v[:, 1:] - v[:, :-1]) / self.dy
        return du_dx + dv_dy

    # ------------------------------------------------------------------
    # Time step
    # ------------------------------------------------------------------

    def _step(self) -> float:
        """Advance the solution by one time step.

        Steps:
          1. Compute tentative velocities u*, v* via forward Euler.
          2. Apply boundary conditions to u*, v*.
          3. Solve pressure Poisson equation:
               ∇²p = (1/dt) * ∇·u*
          4. Correct velocities to be divergence-free.
          5. Apply boundary conditions to corrected velocities.

        Returns:
            L-infinity norm of velocity change (convergence indicator).
        """
        dt = self.config.dt

        u_old = self.u.clone()
        v_old = self.v.clone()

        # --- 1. Tentative velocities (explicit Euler) ---
        u_star = self.u + dt * self._advect_u()
        v_star = self.v + dt * self._advect_v()

        # --- 2. Boundary conditions on tentative field ---
        u_star, v_star = apply_boundary_conditions(
            u_star, v_star, self.config.lid_velocity
        )

        # --- 3. Pressure Poisson solve (implicit, on MPS) ---
        # Solve (-∇²)φ = -∇·u*/dt  (negative Laplacian is positive semi-definite)
        divergence = self._compute_divergence(u_star, v_star)
        rhs = -divergence / dt

        phi = solve_poisson_cg(
            rhs,
            dx=self.dx,
            dy=self.dy,
            tol=self.config.poisson_tol,
            max_iter=self.config.poisson_max_iter,
        )

        # --- 4. Velocity correction (interior faces only) ---
        # Only correct faces that are NOT fixed by boundary conditions.
        # u interior: i=1..nx-1, j=1..ny-2 (excludes left/right walls and lid/bottom)
        # v interior: i=1..nx-2, j=1..ny-1 (excludes left/right walls and top/bottom)
        self.u = u_star.clone()
        self.v = v_star.clone()

        self.u[1:-1, 1:-1] = (
            u_star[1:-1, 1:-1]
            - dt * (phi[1:, 1:-1] - phi[:-1, 1:-1]) / self.dx
        )
        self.v[1:-1, 1:-1] = (
            v_star[1:-1, 1:-1]
            - dt * (phi[1:-1, 1:] - phi[1:-1, :-1]) / self.dy
        )

        # Boundary faces (walls and lid) are already correct from u_star/v_star;
        # re-apply to guarantee exactness after floating-point operations.
        self.u, self.v = apply_boundary_conditions(
            self.u, self.v, self.config.lid_velocity
        )

        self.p = phi

        # Convergence: max velocity change over the interior
        du = (self.u - u_old).abs().max().item()
        dv = (self.v - v_old).abs().max().item()
        return max(du, dv)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> dict[str, torch.Tensor]:
        """Run the solver until steady state or max_steps is reached.

        Returns:
            Dictionary with keys 'u', 'v', 'p' holding the final fields,
            plus 'steps' (number of steps taken) and 'converged' (bool tensor).
        """
        cfg = self.config
        converged = False

        for step in range(1, cfg.max_steps + 1):
            delta = self._step()

            if step % 500 == 0:
                logger.info("step=%6d  max_velocity_change=%.3e", step, delta)

            if delta < cfg.convergence_tol:
                logger.info(
                    "Converged at step=%d  delta=%.3e < tol=%.3e",
                    step,
                    delta,
                    cfg.convergence_tol,
                )
                converged = True
                break

        if not converged:
            logger.warning(
                "Reached max_steps=%d without convergence (last delta=%.3e).",
                cfg.max_steps,
                delta,
            )

        return {
            "u": self.u,
            "v": self.v,
            "p": self.p,
            "steps": torch.tensor(step),
            "converged": torch.tensor(converged),
        }
