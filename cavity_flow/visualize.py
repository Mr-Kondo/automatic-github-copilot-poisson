"""Matplotlib visualisation helpers for cavity flow results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def plot_streamlines(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    nx: int,
    ny: int,
    output_path: Optional[Path] = None,
    title: str = "Lid-Driven Cavity Flow",
) -> None:
    """Plot streamlines and pressure contours for the cavity flow solution.

    Interpolates the staggered-grid velocities to cell centres before
    plotting. Saves to ``output_path`` if provided, otherwise shows
    the figure interactively.

    Args:
        u: x-velocity on staggered grid, shape (nx+1, ny).
        v: y-velocity on staggered grid, shape (nx, ny+1).
        p: Pressure at cell centres, shape (nx, ny).
        nx: Number of cells in x.
        ny: Number of cells in y.
        output_path: File path to save the figure (PNG/PDF). If None,
            calls plt.show() instead.
        title: Figure title string.
    """
    # Move to CPU numpy for matplotlib
    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()

    # Interpolate to cell centres
    u_center = 0.5 * (u_np[:-1, :] + u_np[1:, :])   # (nx, ny)
    v_center = 0.5 * (v_np[:, :-1] + v_np[:, 1:])   # (nx, ny)

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    # --- Streamlines ---
    ax_stream = axes[0]
    speed = np.sqrt(u_center ** 2 + v_center ** 2)
    ax_stream.streamplot(
        x,
        y,
        u_center.T,
        v_center.T,
        color=speed.T,
        cmap="viridis",
        density=1.5,
        linewidth=0.8,
    )
    ax_stream.set_title("Streamlines (coloured by speed)")
    ax_stream.set_xlabel("x")
    ax_stream.set_ylabel("y")
    ax_stream.set_aspect("equal")

    # --- Pressure contours ---
    ax_pres = axes[1]
    cf = ax_pres.contourf(X, Y, p_np, levels=20, cmap="RdBu_r")
    fig.colorbar(cf, ax=ax_pres, label="Pressure")
    ax_pres.set_title("Pressure field")
    ax_pres.set_xlabel("x")
    ax_pres.set_ylabel("y")
    ax_pres.set_aspect("equal")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure to %s", output_path)
    else:
        plt.show()

    plt.close(fig)
