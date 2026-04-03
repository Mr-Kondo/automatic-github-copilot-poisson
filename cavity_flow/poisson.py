"""Conjugate Gradient Poisson solver backed by PyTorch (MPS/CPU)."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)

# 5-point discrete *negative* Laplacian stencil.
# Applying this kernel gives (-∇²)p, which is positive semi-definite with
# Neumann boundary conditions.  CG requires a positive (semi-)definite
# operator, so we solve (-∇²)p = -rhs instead of ∇²p = rhs.
_NEG_LAPLACIAN_KERNEL = torch.tensor(
    [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
    dtype=torch.float32,
)


def _apply_neg_laplacian(p: torch.Tensor, inv_h2: float) -> torch.Tensor:
    """Apply the discrete 5-point *negative* Laplacian operator.

    Replicate padding enforces Neumann (zero-gradient) boundary conditions.
    The resulting operator (-∇²) is positive semi-definite; its null space
    is spanned by the constant function.

    Args:
        p: Pressure field, shape (nx, ny).
        inv_h2: 1 / (h²) where h = dx = dy (square cells assumed).

    Returns:
        (-∇²)p, shape (nx, ny).
    """
    device = p.device
    kernel = _NEG_LAPLACIAN_KERNEL.to(device)

    p4 = p.unsqueeze(0).unsqueeze(0)
    p_padded = F.pad(p4, (1, 1, 1, 1), mode="replicate")
    result = F.conv2d(p_padded, kernel.unsqueeze(0).unsqueeze(0))
    return result.squeeze(0).squeeze(0) * inv_h2


def solve_poisson_cg(
    rhs: torch.Tensor,
    dx: float,
    dy: float,
    tol: float = 1e-6,
    max_iter: int = 2000,
) -> torch.Tensor:
    """Solve the Poisson equation ∇²p = -rhs using Conjugate Gradient.

    Internally the solver operates on the positive semi-definite system
    (-∇²)p = rhs (note the sign convention: callers should pass
    ``rhs = -divergence / dt`` for the MAC pressure step).

    The null space of (-∇²) under Neumann boundary conditions is the
    set of constant functions.  The solver projects the RHS and the
    iterate to be mean-zero at every step (deflated / projected CG),
    which restricts the problem to the orthogonal complement where the
    operator is positive definite.

    The solver runs entirely on the device of ``rhs`` (MPS or CPU).

    Args:
        rhs: Right-hand side, shape (nx, ny).
        dx: Cell width in x direction.
        dy: Cell height in y direction.
        tol: Absolute L2 residual tolerance for convergence.
        max_iter: Maximum number of CG iterations.

    Returns:
        Pressure field p (mean-zero), shape (nx, ny).

    Raises:
        ValueError: For invalid solver parameters.
        RuntimeError: If the solver does not converge within max_iter.
    """
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(f"dx and dy must be positive; got dx={dx}, dy={dy}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive; got tol={tol}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive; got max_iter={max_iter}")

    inv_h2 = 1.0 / (dx * dy)

    # Project rhs to satisfy compatibility condition (mean must be zero for
    # Neumann BCs: the net pressure source over the domain must vanish).
    rhs_proj = rhs - rhs.mean()

    p = torch.zeros_like(rhs_proj)
    r = rhs_proj - _apply_neg_laplacian(p, inv_h2)
    r = r - r.mean()  # project residual into orthogonal complement
    d = r.clone()
    r_dot = torch.dot(r.flatten(), r.flatten())

    for iteration in range(max_iter):
        if r_dot.item() < tol ** 2:
            logger.debug(
                "CG converged at iteration %d, residual²=%.3e", iteration, r_dot.item()
            )
            return p - p.mean()

        Ad = _apply_neg_laplacian(d, inv_h2)
        Ad = Ad - Ad.mean()  # project operator output
        dAd = torch.dot(d.flatten(), Ad.flatten())

        if dAd.abs().item() < 1e-30:
            logger.warning("CG stagnated at iteration %d (dAd ~ 0)", iteration)
            return p - p.mean()

        alpha = r_dot / dAd
        p = p + alpha * d
        p = p - p.mean()  # project iterate — removes null space growth
        r = r - alpha * Ad
        r = r - r.mean()  # project residual

        r_dot_new = torch.dot(r.flatten(), r.flatten())
        beta = r_dot_new / r_dot
        d = r + beta * d
        r_dot = r_dot_new

    raise RuntimeError(
        f"Poisson CG did not converge in {max_iter} iterations; "
        f"final residual²={r_dot.item():.3e}"
    )


# ---------------------------------------------------------------------------
# Helmholtz solver  (I − c ∇²) x = b  with zero-Dirichlet BCs
# ---------------------------------------------------------------------------


def _apply_helmholtz_interior(
    x: torch.Tensor,
    c: float,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Apply the operator (I − c ∇²) to *x* using zero Dirichlet boundary conditions.

    Pads *x* with zeros before computing the Laplacian, which is equivalent to
    Dirichlet-zero boundary conditions on the outer ring of the domain.  When
    non-zero Dirichlet values are needed, callers must adjust the RHS *before*
    calling this function (see ``solve_helmholtz_cg``).

    Args:
        x: Field to operate on, shape (ni, nj).
        c: Diffusion coefficient (must be positive; typically ``nu * dt``).
        dx: Cell width in x.
        dy: Cell height in y.

    Returns:
        (I − c ∇²) x, shape (ni, nj).
    """
    inv_dx2 = 1.0 / dx ** 2
    inv_dy2 = 1.0 / dy ** 2

    # Zero-pad: Dirichlet-zero at all four sides.
    x4 = x.unsqueeze(0).unsqueeze(0)
    xp = F.pad(x4, (1, 1, 1, 1), mode="constant", value=0.0).squeeze(0).squeeze(0)

    lap = (
        (xp[2:, 1:-1] - 2.0 * x + xp[:-2, 1:-1]) * inv_dx2
        + (xp[1:-1, 2:] - 2.0 * x + xp[1:-1, :-2]) * inv_dy2
    )
    return x - c * lap


def solve_helmholtz_cg(
    rhs: torch.Tensor,
    c: float,
    dx: float,
    dy: float,
    tol: float = 1e-5,
    max_iter: int = 2000,
) -> torch.Tensor:
    """Solve (I − c ∇²) x = rhs using Conjugate Gradient.

    Uses **zero Dirichlet** boundary conditions on the perimeter of the domain.
    For non-zero boundary values the caller must subtract their contribution
    from *rhs* before calling (see the implicit-diffusion step in the solver).

    Because (I − c ∇²) with c > 0 and Dirichlet BCs is symmetric positive
    definite, CG converges without any special treatment of the null space.

    The solver runs entirely on the device of *rhs* (MPS or CPU).

    Args:
        rhs: Right-hand side, shape (ni, nj).
        c: Positive diffusion coefficient (e.g. ``nu * dt``).
        dx: Cell width.
        dy: Cell height.
        tol: Absolute L2 residual tolerance.
        max_iter: Maximum number of CG iterations.

    Returns:
        Solution field x, shape (ni, nj).

    Raises:
        ValueError: For invalid solver parameters.
        RuntimeError: If CG does not converge within *max_iter* iterations.
    """
    if c <= 0.0:
        raise ValueError(f"c must be positive; got c={c}")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(f"dx and dy must be positive; got dx={dx}, dy={dy}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive; got tol={tol}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive; got max_iter={max_iter}")

    x = torch.zeros_like(rhs)
    r = rhs - _apply_helmholtz_interior(x, c, dx, dy)
    d = r.clone()
    r_dot = torch.dot(r.flatten(), r.flatten())

    for iteration in range(max_iter):
        if r_dot.item() < tol ** 2:
            logger.debug(
                "Helmholtz CG converged at iteration %d, residual²=%.3e",
                iteration,
                r_dot.item(),
            )
            return x

        Ad = _apply_helmholtz_interior(d, c, dx, dy)
        dAd = torch.dot(d.flatten(), Ad.flatten())

        if dAd.abs().item() < 1e-30:
            logger.warning(
                "Helmholtz CG stagnated at iteration %d (dAd ~ 0)", iteration
            )
            return x

        alpha = r_dot / dAd
        x = x + alpha * d
        r = r - alpha * Ad

        r_dot_new = torch.dot(r.flatten(), r.flatten())
        beta = r_dot_new / r_dot
        d = r + beta * d
        r_dot = r_dot_new

    raise RuntimeError(
        f"Helmholtz CG did not converge in {max_iter} iterations; "
        f"final residual²={r_dot.item():.3e}"
    )

