"""Boundary condition helpers for lid-driven cavity flow."""

import torch


def apply_boundary_conditions(
    u: torch.Tensor,
    v: torch.Tensor,
    lid_velocity: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply lid-driven cavity boundary conditions in-place.

    Staggered-grid layout (nx=ny=N):
      u shape: (nx+1, ny)  — x-velocity at vertical cell faces
      v shape: (nx, ny+1)  — y-velocity at horizontal cell faces

    Boundary conditions:
      - Top lid  (j = ny-1): u = lid_velocity, v = 0 at top wall
      - Bottom   (j = 0):    u = 0
      - Left     (i = 0):    v = 0
      - Right    (i = nx):   v = 0
      All wall-normal velocities set to zero on boundaries.

    Args:
        u: x-velocity tensor, shape (nx+1, ny).
        v: y-velocity tensor, shape (nx, ny+1).
        lid_velocity: velocity of the top lid (default 1.0).

    Returns:
        Tuple (u, v) with boundary values enforced.
    """
    # Set all wall (no-slip) velocities first, then apply lid last so that
    # the moving lid wins at the top corners (i=0, -1 ; j=-1).

    # Bottom wall: u = 0
    u[:, 0] = 0.0
    # Left wall: u = 0
    u[0, :] = 0.0
    # Right wall: u = 0
    u[-1, :] = 0.0

    # Left wall: v = 0
    v[0, :] = 0.0
    # Right wall: v = 0
    v[-1, :] = 0.0
    # Bottom wall: v = 0
    v[:, 0] = 0.0
    # Top wall: v = 0
    v[:, -1] = 0.0

    # Top lid (applied last so it overrides corner values set above)
    u[:, -1] = lid_velocity

    return u, v
