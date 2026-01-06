from __future__ import annotations

from pathlib import Path
import csv
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parseFloatOrNone(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip().lower()
    if text == "" or text == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def loadCsvRows(csvPath: Path) -> List[Dict[str, str]]:
    with csvPath.open("r", newline="") as fileHandle:
        reader = csv.DictReader(fileHandle)
        return list(reader)


def extractAxisValues(row: Dict[str, str], prefix: str, axisNames: Sequence[str]) -> List[float]:
    values: List[float] = []
    for axisName in axisNames:
        value = parseFloatOrNone(row.get(f"{prefix}_{axisName}"))
        if value is not None:
            values.append(float(value))
    return values


def computeMeanGradient(csvPath: Path, axisNames: Sequence[str]) -> Tuple[float, float]:
    rows = loadCsvRows(csvPath)

    fdPerRowMeans: List[float] = []
    anPerRowMeans: List[float] = []

    for row in rows:
        fdAxisValues = extractAxisValues(row, "fd", axisNames)
        anAxisValues = extractAxisValues(row, "an", axisNames)

        # Require at least one axis for both FD and AN in this row
        if len(fdAxisValues) == 0 or len(anAxisValues) == 0:
            continue

        fdPerRowMeans.append(float(np.mean(fdAxisValues)))
        anPerRowMeans.append(float(np.mean(anAxisValues)))

    return float(np.sum(fdPerRowMeans)), float(np.sum(anPerRowMeans))


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
translationGradientsPath = Path(
    "output/initial/camera1/translation/translation_gradients_history.csv"
)

rotationGradientsPath = Path(
    "output/initial/camera1/rotation/rotation_gradients_history.csv"
)

scaleGradientsPath = Path(
    "output/initial/camera1/scale/scale_gradients_history.csv"
)
opacityGradientsPath = Path(
    "output/initial/camera1/opacity/opacity_gradients_history.csv"
)

albedoGradientsPath = Path(
    "output/initial/camera1/albedo/albedo_gradients_history.csv"
)

betaGradientsPath = Path(
    "output/initial/camera1/beta/beta_gradients_history.csv"
)

# ------------------------------------------------------------
# Aggregate values
# ------------------------------------------------------------
fdPosAvg, anPosAvg = computeMeanGradient(translationGradientsPath, axisNames=["x", "y", "z"])
fdRotAvg, anRotAvg = computeMeanGradient(rotationGradientsPath, axisNames=["x", "y", "z"])
fdScaleAvg, anScaleAvg = computeMeanGradient(scaleGradientsPath, axisNames=["x", "y"])  # only 2 axes
fdOpacityAvg, anOpacityAvg = computeMeanGradient(opacityGradientsPath, axisNames=["x"])  # only 2 axes
fdAlbedoAvg, anAlbedoAvg = computeMeanGradient(albedoGradientsPath, axisNames=["x", "y", "z"])  # only 2 axes
fdBetaAvg, anBetaAvg = computeMeanGradient(betaGradientsPath, axisNames=["x"])  # only 2 axes

labels = ["Position", "Rotation", "Scale", "Opacity", "Albedo", "Beta"]

fdValues = [fdPosAvg, fdRotAvg, fdScaleAvg, fdOpacityAvg, fdAlbedoAvg, fdBetaAvg]
anValues = [anPosAvg, anRotAvg, anScaleAvg, anOpacityAvg, anAlbedoAvg, anBetaAvg]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
x = np.arange(len(labels), dtype=np.float64)
barWidth = 0.35

plt.figure(figsize=(11.0, 4.5))

finiteDifferenceColor = "#F4A261"  # light orange
analyticColor = "#8FD3B6"          # light green


groupSpacingFactor = 0.55  # > 0.5 gives visible air between bars

plt.bar(
    x - groupSpacingFactor * barWidth,
    anValues,
    width=barWidth,
    label="Ours",
    color=analyticColor,
    edgecolor="black",
    linewidth=0.0,
    alpha=0.95,
)

plt.bar(
    x + groupSpacingFactor * barWidth,
    fdValues,
    width=barWidth,
    label="FD",
    color=finiteDifferenceColor,
    edgecolor="black",
    linewidth=0.0,
    alpha=0.95,
)

plt.xticks(x, labels, rotation=0, ha="center", fontsize=22)
plt.yscale("symlog", linthresh=1e-6)
plt.ylabel("Gradient Value", fontsize=22)
plt.legend(
    fontsize=20,
)
plt.tight_layout()
currentYMin, currentYMax = plt.ylim()
plt.ylim(currentYMin, currentYMax * 1.25)  # ~15% air on top

outputPath = "finite_difference_plot.pdf"

# Optional: embed fonts (avoids font substitution differences across viewers)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] =  42

plt.savefig(outputPath, format="pdf", bbox_inches="tight")
plt.show()
plt.close()

print(f"Wrote plot: {outputPath}")