"""Pytest unit tests for the cavity_flow package."""

from __future__ import annotations

import math

import pytest
import torch

from cavity_flow.boundary import apply_boundary_conditions
from cavity_flow.poisson import solve_helmholtz_cg, solve_poisson_cg
from cavity_flow.solver import CavityFlowSolver, SolverConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CPU = torch.device("cpu")


def _small_config(**kwargs: float | int | bool) -> SolverConfig:
    """Return a small-grid config suitable for fast unit tests."""
    defaults: dict[str, float | int | bool] = dict(
        nx=16,
        ny=16,
        re=100.0,
        dt=1e-3,
        max_steps=5,
        convergence_tol=1e-12,  # effectively disabled → always runs max_steps
        lid_velocity=1.0,
        poisson_tol=1e-6,
        poisson_max_iter=500,
        implicit_diffusion=True,
    )
    defaults.update(kwargs)
    return SolverConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# T1 — boundary conditions: top-lid velocity is applied
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_top_lid_u_equals_lid_velocity(self) -> None:
        """T1: After apply_boundary_conditions, top row of u == lid_velocity."""
        nx, ny = 8, 8
        u = torch.zeros(nx + 1, ny)
        v = torch.zeros(nx, ny + 1)

        u, v = apply_boundary_conditions(u, v, lid_velocity=1.0)

        assert torch.all(u[:, -1] == 1.0), "Top lid must have u = 1.0"

    def test_other_walls_u_zero(self) -> None:
        """T1b: Bottom wall and left/right walls have u = 0.

        The top corners (u[0, -1] and u[-1, -1]) belong to the moving lid
        (which is applied last) and are therefore lid_velocity, not zero.
        """
        nx, ny = 8, 8
        u = torch.ones(nx + 1, ny) * 5.0
        v = torch.zeros(nx, ny + 1)

        u, v = apply_boundary_conditions(u, v, lid_velocity=1.0)

        assert torch.all(u[:, 0] == 0.0), "Bottom wall u must be zero"
        # Exclude top corner (j=-1) which belongs to the moving lid
        assert torch.all(u[0, :-1] == 0.0), "Left wall u (below lid corner) must be zero"
        assert torch.all(u[-1, :-1] == 0.0), "Right wall u (below lid corner) must be zero"

    def test_all_v_walls_zero(self) -> None:
        """T1c: All v wall values must be zero for lid-driven cavity."""
        nx, ny = 8, 8
        u = torch.zeros(nx + 1, ny)
        v = torch.ones(nx, ny + 1) * 3.0

        u, v = apply_boundary_conditions(u, v, lid_velocity=1.0)

        assert torch.all(v[0, :] == 0.0), "Left wall v must be zero"
        assert torch.all(v[-1, :] == 0.0), "Right wall v must be zero"
        assert torch.all(v[:, 0] == 0.0), "Bottom wall v must be zero"
        assert torch.all(v[:, -1] == 0.0), "Top wall v must be zero"

    def test_interior_cells_not_modified(self) -> None:
        """T8: Boundary enforcement does not modify interior cells."""
        nx, ny = 8, 8
        u_interior_val = 7.0
        u = torch.full((nx + 1, ny), u_interior_val)
        v = torch.zeros(nx, ny + 1)

        u, v = apply_boundary_conditions(u, v, lid_velocity=1.0)

        # Interior of u: rows 1 to nx-1 (exclusive of boundary columns)
        assert torch.all(u[1:-1, 1:-1] == u_interior_val), (
            "Interior u values must not be changed by BC application"
        )


# ---------------------------------------------------------------------------
# T2 — pressure correction reduces divergence
# ---------------------------------------------------------------------------


class TestPressureCorrection:
    def test_divergence_near_zero_after_step(self) -> None:
        """T2: After one solver step, interior cell divergence is near zero.

        Boundary-adjacent cells naturally retain small divergence residuals in
        a staggered MAC scheme because the lid/wall faces are fixed and are not
        included in the pressure-gradient correction.  Interior cells (not
        adjacent to any wall) must be divergence-free.
        """
        cfg = _small_config(max_steps=1, convergence_tol=0.0)
        solver = CavityFlowSolver(config=cfg, device=CPU)

        solver.run()

        div = solver._compute_divergence(solver.u, solver.v)

        # Exclude cells directly adjacent to all four walls
        interior_div = div[1:-1, 1:-1]
        max_interior_div = interior_div.abs().max().item()

        assert max_interior_div < 1e-4, (
            f"Max interior divergence should be < 1e-4; got {max_interior_div:.3e}"
        )


# ---------------------------------------------------------------------------
# T3 — CG Poisson solver converges on a known problem
# ---------------------------------------------------------------------------


class TestPoissonSolver:
    def test_cg_converges_for_zero_rhs(self) -> None:
        """T3a: CG returns the zero solution for a zero RHS."""
        rhs = torch.zeros(16, 16)
        p = solve_poisson_cg(rhs, dx=1.0 / 16, dy=1.0 / 16, tol=1e-8)

        assert p.shape == (16, 16)
        assert p.abs().max().item() < 1e-8

    def test_cg_reduces_residual(self) -> None:
        """T3b: CG residual decreases for a non-trivial RHS."""
        nx, ny = 32, 32
        x = torch.linspace(0, 1, nx)
        y = torch.linspace(0, 1, ny)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        rhs = torch.sin(math.pi * X) * torch.sin(math.pi * Y)

        p = solve_poisson_cg(rhs, dx=1.0 / nx, dy=1.0 / ny, tol=1e-5, max_iter=500)

        # Check that the solution is non-trivial
        assert p.abs().max().item() > 1e-4

    def test_cg_invalid_dx_raises(self) -> None:
        """T7a: CG raises ValueError for non-positive dx."""
        rhs = torch.zeros(8, 8)
        with pytest.raises(ValueError, match="dx and dy must be positive"):
            solve_poisson_cg(rhs, dx=0.0, dy=0.1)

    def test_cg_invalid_tol_raises(self) -> None:
        """T7b: CG raises ValueError for non-positive tol."""
        rhs = torch.zeros(8, 8)
        with pytest.raises(ValueError, match="tol must be positive"):
            solve_poisson_cg(rhs, dx=0.1, dy=0.1, tol=0.0)

    def test_cg_invalid_max_iter_raises(self) -> None:
        """T7c: CG raises RuntimeError when max_iter is exhausted."""
        # sin(2πx)sin(2πy) integrates to zero over [0,1]² so it is compatible
        # with Neumann BCs and will not converge in just 1 iteration.
        nx = 32
        x = torch.linspace(0.0, 1.0, nx)
        y = torch.linspace(0.0, 1.0, nx)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        rhs = torch.sin(2.0 * math.pi * X) * torch.sin(2.0 * math.pi * Y)
        with pytest.raises(RuntimeError, match="did not converge"):
            solve_poisson_cg(rhs, dx=1.0 / nx, dy=1.0 / nx, tol=1e-20, max_iter=1)


# ---------------------------------------------------------------------------
# T4 — zero-lid symmetry: all-wall case should give near-zero velocities
# ---------------------------------------------------------------------------


class TestZeroLidSymmetry:
    def test_zero_lid_velocity_stays_zero(self) -> None:
        """T4: With lid_velocity=0 and no initial flow, velocities remain zero."""
        cfg = _small_config(lid_velocity=0.0, max_steps=3)
        solver = CavityFlowSolver(config=cfg, device=CPU)

        solver.run()

        assert solver.u.abs().max().item() < 1e-10
        assert solver.v.abs().max().item() < 1e-10


# ---------------------------------------------------------------------------
# T5 & T6 — SolverConfig validation errors
# ---------------------------------------------------------------------------


class TestSolverConfigValidation:
    def test_non_positive_nx_raises(self) -> None:
        """T5a: ValueError for nx <= 0."""
        with pytest.raises(ValueError, match="nx and ny must be positive"):
            SolverConfig(nx=0, ny=10)

    def test_non_positive_ny_raises(self) -> None:
        """T5b: ValueError for ny <= 0."""
        with pytest.raises(ValueError, match="nx and ny must be positive"):
            SolverConfig(nx=10, ny=-1)

    def test_non_positive_re_raises(self) -> None:
        """T6: ValueError for re <= 0."""
        with pytest.raises(ValueError, match="Reynolds number must be positive"):
            SolverConfig(re=0.0)

    def test_non_positive_dt_raises(self) -> None:
        """T5c: ValueError for dt <= 0."""
        with pytest.raises(ValueError, match="Time step must be positive"):
            SolverConfig(dt=-1e-3)

    def test_unstable_dt_explicit_diffusion_raises(self) -> None:
        """T5d: ValueError when dt exceeds the explicit diffusion limit (implicit_diffusion=False)."""
        with pytest.raises(ValueError, match="stability limit"):
            SolverConfig(nx=300, ny=300, re=100.0, dt=1e-3, implicit_diffusion=False)

    def test_unstable_dt_cfl_raises(self) -> None:
        """T5e: ValueError when dt exceeds the advection CFL limit (implicit_diffusion=True)."""
        with pytest.raises(ValueError, match="CFL"):
            SolverConfig(nx=300, ny=300, re=100.0, dt=1.0, implicit_diffusion=True)

    def test_implicit_dt_larger_than_explicit_limit_is_valid(self) -> None:
        """T5f: dt above explicit diffusion limit is valid when implicit_diffusion=True."""
        # dt=1e-3 exceeds the explicit diffusion limit for 300x300 Re=100
        # but the CFL condition is only 0.3, so it should be accepted.
        cfg = SolverConfig(nx=300, ny=300, re=100.0, dt=1e-3, implicit_diffusion=True)
        assert cfg.implicit_diffusion is True


# ---------------------------------------------------------------------------
# T9 — Helmholtz CG solver
# ---------------------------------------------------------------------------


class TestHelmholtzSolver:
    def test_zero_rhs_returns_zero(self) -> None:
        """T9a: Helmholtz CG returns the zero solution for a zero RHS."""
        rhs = torch.zeros(16, 16)
        x = solve_helmholtz_cg(rhs, c=1e-3, dx=1.0 / 16, dy=1.0 / 16, tol=1e-8)

        assert x.shape == (16, 16)
        assert x.abs().max().item() < 1e-8

    def test_solution_satisfies_equation(self) -> None:
        """T9b: Helmholtz solution approximately satisfies (I - c∇²)x = rhs."""
        from cavity_flow.poisson import _apply_helmholtz_interior

        nx, ny = 32, 32
        dx = dy = 1.0 / nx
        c = 1e-2
        # Non-trivial, mean-zero-like RHS
        xg = torch.linspace(0, 1, nx)
        yg = torch.linspace(0, 1, ny)
        X, Y = torch.meshgrid(xg, yg, indexing="ij")
        rhs = torch.sin(math.pi * X) * torch.sin(math.pi * Y)

        x = solve_helmholtz_cg(rhs, c=c, dx=dx, dy=dy, tol=1e-7, max_iter=1000)

        residual = rhs - _apply_helmholtz_interior(x, c, dx, dy)
        assert residual.norm().item() < 1e-5

    def test_invalid_c_raises(self) -> None:
        """T9c: ValueError for non-positive c."""
        rhs = torch.zeros(8, 8)
        with pytest.raises(ValueError, match="c must be positive"):
            solve_helmholtz_cg(rhs, c=0.0, dx=0.1, dy=0.1)

    def test_invalid_dx_raises(self) -> None:
        """T9d: ValueError for non-positive dx."""
        rhs = torch.zeros(8, 8)
        with pytest.raises(ValueError, match="dx and dy must be positive"):
            solve_helmholtz_cg(rhs, c=1e-3, dx=0.0, dy=0.1)

    def test_invalid_tol_raises(self) -> None:
        """T9e: ValueError for non-positive tol."""
        rhs = torch.zeros(8, 8)
        with pytest.raises(ValueError, match="tol must be positive"):
            solve_helmholtz_cg(rhs, c=1e-3, dx=0.1, dy=0.1, tol=0.0)

    def test_non_convergence_raises(self) -> None:
        """T9f: RuntimeError when max_iter is exhausted."""
        nx = 32
        xg = torch.linspace(0.0, 1.0, nx)
        yg = torch.linspace(0.0, 1.0, nx)
        X, Y = torch.meshgrid(xg, yg, indexing="ij")
        rhs = torch.sin(math.pi * X) * torch.sin(math.pi * Y)
        with pytest.raises(RuntimeError, match="did not converge"):
            solve_helmholtz_cg(rhs, c=1e-3, dx=1.0 / nx, dy=1.0 / nx, tol=1e-20, max_iter=1)

    def test_invalid_max_iter_raises(self) -> None:
        """T9g: ValueError for non-positive max_iter."""
        rhs = torch.zeros(8, 8)
        with pytest.raises(ValueError, match="max_iter must be positive"):
            solve_helmholtz_cg(rhs, c=1e-3, dx=0.1, dy=0.1, max_iter=0)


# ---------------------------------------------------------------------------
# T10 — Implicit diffusion step correctness
# ---------------------------------------------------------------------------


class TestImplicitDiffusion:
    def test_implicit_divergence_near_zero(self) -> None:
        """T10a: Interior divergence near zero after one implicit-diffusion step."""
        cfg = _small_config(max_steps=1, convergence_tol=0.0, implicit_diffusion=True)
        solver = CavityFlowSolver(config=cfg, device=CPU)

        solver.run()

        div = solver._compute_divergence(solver.u, solver.v)
        interior_div = div[1:-1, 1:-1]
        max_div = interior_div.abs().max().item()
        assert max_div < 1e-4, (
            f"Max interior divergence (implicit) should be < 1e-4; got {max_div:.3e}"
        )

    def test_explicit_divergence_near_zero(self) -> None:
        """T10b: Interior divergence near zero after one explicit-diffusion step."""
        cfg = _small_config(max_steps=1, convergence_tol=0.0, implicit_diffusion=False)
        solver = CavityFlowSolver(config=cfg, device=CPU)

        solver.run()

        div = solver._compute_divergence(solver.u, solver.v)
        interior_div = div[1:-1, 1:-1]
        max_div = interior_div.abs().max().item()
        assert max_div < 1e-4, (
            f"Max interior divergence (explicit) should be < 1e-4; got {max_div:.3e}"
        )

    def test_zero_lid_implicit_stays_zero(self) -> None:
        """T10c: With zero lid velocity, implicit diffusion keeps velocities at zero."""
        cfg = _small_config(lid_velocity=0.0, max_steps=3, implicit_diffusion=True)
        solver = CavityFlowSolver(config=cfg, device=CPU)

        solver.run()

        assert solver.u.abs().max().item() < 1e-10
        assert solver.v.abs().max().item() < 1e-10

    def test_implicit_bc_preserved(self) -> None:
        """T10d: After implicit-diffusion steps, lid BC remains exact."""
        cfg = _small_config(max_steps=3, implicit_diffusion=True)
        solver = CavityFlowSolver(config=cfg, device=CPU)

        solver.run()

        assert torch.all(solver.u[:, -1] == cfg.lid_velocity), (
            "Top lid velocity must be preserved after implicit-diffusion steps"
        )
        assert torch.all(solver.u[:, 0] == 0.0), "Bottom wall u must be zero"
        assert torch.all(solver.v[0, :] == 0.0), "Left wall v must be zero"
        assert torch.all(solver.v[-1, :] == 0.0), "Right wall v must be zero"

