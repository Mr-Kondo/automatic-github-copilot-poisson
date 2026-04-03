"""Entry point for running the lid-driven cavity flow simulation."""

import logging
from pathlib import Path

from cavity_flow.solver import CavityFlowSolver, SolverConfig
from cavity_flow.visualize import plot_streamlines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)


def main() -> None:
    config = SolverConfig(
        nx=300,
        ny=300,
        re=100.0,
        dt=2.5e-4,
        max_steps=10000,
    )

    solver = CavityFlowSolver(config)
    result = solver.run()

    converged = result["converged"].item()
    steps = result["steps"].item()
    print(f"\nDone — converged: {converged}, steps taken: {steps}")

    output_path = Path("output.png")
    plot_streamlines(
        result["u"],
        result["v"],
        result["p"],
        nx=config.nx,
        ny=config.ny,
        output_path=output_path,
    )
    print(f"Figure saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
