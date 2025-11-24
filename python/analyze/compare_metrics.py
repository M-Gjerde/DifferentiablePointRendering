#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt


def load_metrics_from_folder(runFolder: Path) -> Tuple[List[int], List[float], List[float]]:
    """
    Load iteration, loss_l2, and parameter_mse from metrics.csv
    in the given run folder.

    Returns:
        iterations, lossValues, parameterMseValues
    """
    metricsFilePath = runFolder / "metrics.csv"
    if not metricsFilePath.is_file():
        raise FileNotFoundError(f"metrics.csv not found in folder: {metricsFilePath}")

    iterations: List[int] = []
    lossValues: List[float] = []
    parameterMseValues: List[float] = []

    with metricsFilePath.open("r", newline="") as metricsFile:
        csvReader = csv.DictReader(metricsFile)
        requiredColumns = {"iteration", "loss_l2", "parameter_mse"}
        missingColumns = requiredColumns - set(csvReader.fieldnames or [])
        if missingColumns:
            raise ValueError(
                f"metrics.csv in {runFolder} is missing required columns: {missingColumns}"
            )

        for row in csvReader:
            try:
                iterations.append(int(row["iteration"]))
                lossValues.append(float(row["loss_l2"]))
                parameterMseValues.append(float(row["parameter_mse"]))
            except ValueError as valueError:
                raise ValueError(
                    f"Failed parsing row in {metricsFilePath}: {row}"
                ) from valueError

    return iterations, lossValues, parameterMseValues


def cap_by_max_iterations(
    iterations: List[int],
    values: List[float],
    maxIterations: int | None,
) -> Tuple[List[int], List[float]]:
    """
    Optionally cap the series at iteration <= maxIterations.
    If maxIterations is None, the input is returned unchanged.
    """
    if maxIterations is None:
        return iterations, values

    cappedIterations: List[int] = []
    cappedValues: List[float] = []
    for iteration, value in zip(iterations, values):
        if iteration <= maxIterations:
            cappedIterations.append(iteration)
            cappedValues.append(value)

    return cappedIterations, cappedValues

def create_comparison_plot(
    seriesList: List[Dict[str, Any]],
    yLabel: str,
    title: str,
    outputPath: Path,
) -> None:
    """
    Create a high-resolution comparison plot with visual differentiation
    for overlapping lines.
    """
    # High-resolution figure
    plt.figure(figsize=(5, 5), dpi=300)

    # Define distinct styles
    lineStyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^"]
    colorCycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, series in enumerate(seriesList):
        plt.plot(
            series["iterations"],
            series["values"],
            label=series["label"],
            linewidth=2.0,
            linestyle=lineStyles[i % len(lineStyles)],
            marker=markers[i % len(markers)],
            markersize=2.0,
            alpha=0.9,
            color=colorCycle[i % len(colorCycle)],
        )

    plt.xlabel("Iteration")
    plt.ylabel(yLabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outputPath, dpi=300)
    plt.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare metrics (loss_l2 and parameter_mse) between up to three training runs. "
            "Each run folder must contain a metrics.csv file."
        )
    )

    parser.add_argument(
        "--run-a",
        type=Path,
        required=True,
        help="Path to first run folder (containing metrics.csv).",
    )
    parser.add_argument(
        "--run-b",
        type=Path,
        required=True,
        help="Path to second run folder (containing metrics.csv).",
    )
    parser.add_argument(
        "--run-c",
        type=Path,
        required=False,
        help="Optional path to third run folder (containing metrics.csv).",
    )

    parser.add_argument(
        "--label-a",
        type=str,
        default=None,
        help="Label for first run in the plots (default: folder name).",
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default=None,
        help="Label for second run in the plots (default: folder name).",
    )
    parser.add_argument(
        "--label-c",
        type=str,
        default=None,
        help="Label for third run in the plots (default: folder name).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ComparisonPlots"),
        help="Directory where comparison plots will be saved.",
    )

    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="If set, only data with iteration <= max-iters is plotted.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    runFolderA: Path = arguments.run_a
    runFolderB: Path = arguments.run_b
    runFolderC: Path | None = arguments.run_c
    outputDirectory: Path = arguments.output_dir
    maxIterations: int | None = arguments.max_iters

    if arguments.label_a is not None:
        labelA = arguments.label_a
    else:
        labelA = runFolderA.name

    if arguments.label_b is not None:
        labelB = arguments.label_b
    else:
        labelB = runFolderB.name

    if runFolderC is not None:
        if arguments.label_c is not None:
            labelC = arguments.label_c
        else:
            labelC = runFolderC.name
    else:
        labelC = None

    print(f"Loading metrics from: {runFolderA}")
    iterationsA, lossValuesA, parameterMseValuesA = load_metrics_from_folder(runFolderA)

    print(f"Loading metrics from: {runFolderB}")
    iterationsB, lossValuesB, parameterMseValuesB = load_metrics_from_folder(runFolderB)

    iterationsA, lossValuesA = cap_by_max_iterations(iterationsA, lossValuesA, maxIterations)
    _, parameterMseValuesA = cap_by_max_iterations(iterationsA, parameterMseValuesA, maxIterations)

    iterationsB, lossValuesB = cap_by_max_iterations(iterationsB, lossValuesB, maxIterations)
    _, parameterMseValuesB = cap_by_max_iterations(iterationsB, parameterMseValuesB, maxIterations)

    seriesLoss: List[Dict[str, Any]] = [
        {
            "iterations": iterationsA,
            "values": lossValuesA,
            "label": labelA,
        },
        {
            "iterations": iterationsB,
            "values": lossValuesB,
            "label": labelB,
        },
    ]
    seriesParamMse: List[Dict[str, Any]] = [
        {
            "iterations": iterationsA,
            "values": parameterMseValuesA,
            "label": labelA,
        },
        {
            "iterations": iterationsB,
            "values": parameterMseValuesB,
            "label": labelB,
        },
    ]

    if runFolderC is not None:
        print(f"Loading metrics from: {runFolderC}")
        iterationsC, lossValuesC, parameterMseValuesC = load_metrics_from_folder(runFolderC)

        iterationsC, lossValuesC = cap_by_max_iterations(iterationsC, lossValuesC, maxIterations)
        _, parameterMseValuesC = cap_by_max_iterations(iterationsC, parameterMseValuesC, maxIterations)

        seriesLoss.append(
            {
                "iterations": iterationsC,
                "values": lossValuesC,
                "label": labelC,
            }
        )
        seriesParamMse.append(
            {
                "iterations": iterationsC,
                "values": parameterMseValuesC,
                "label": labelC,
            }
        )

    outputDirectory.mkdir(parents=True, exist_ok=True)

    # Loss comparison plot
    lossPlotPath = outputDirectory / "comparison_loss.png"
    print(f"Saving loss comparison plot to: {lossPlotPath}")
    create_comparison_plot(
        seriesList=seriesLoss,
        yLabel="L2 Loss",
        title="Loss Comparison",
        outputPath=lossPlotPath,
    )

    # Parameter MSE comparison plot
    parameterMsePlotPath = outputDirectory / "comparison_parameter_mse.png"
    print(f"Saving parameter MSE comparison plot to: {parameterMsePlotPath}")
    create_comparison_plot(
        seriesList=seriesParamMse,
        yLabel="Parameter MSE",
        title="Parameter MSE Comparison",
        outputPath=parameterMsePlotPath,
    )

    print("Done. Comparison plots generated.")


if __name__ == "__main__":
    main()
