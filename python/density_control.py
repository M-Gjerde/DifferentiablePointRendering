import numpy as np
import torch


def densify_points(
    positions: torch.nn.Parameter,
    tangent_u: torch.nn.Parameter,
    tangent_v: torch.nn.Parameter,
    scales: torch.nn.Parameter,
    colors: torch.nn.Parameter,
) -> dict | None:
    """
    Simple densification based on surfel scale.

    Does NOT modify the input tensors or any optimizer.

    Returns:
        None if no densification is requested.

        Otherwise a dict:
        {
            "updated": {
                "indices": np.ndarray [K],       # parent indices in point cloud
                "scale":   np.ndarray [K, 2],    # new scales for those parents
            },
            "new": {
                "position":  np.ndarray [M, 3],
                "tangent_u": np.ndarray [M, 3],
                "tangent_v": np.ndarray [M, 3],
                "scale":     np.ndarray [M, 2],
                "color":     np.ndarray [M, 3],
            }
        }
    """
    with torch.no_grad():
        positions_np = positions.detach().cpu().numpy()
        tangent_u_np = tangent_u.detach().cpu().numpy()
        tangent_v_np = tangent_v.detach().cpu().numpy()
        scales_np = scales.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy()

    numberOfPoints = positions_np.shape[0]
    if numberOfPoints == 0:
        return None

    # --------------------------------------------------------------
    # 1) Area-based densification rule
    # --------------------------------------------------------------
    gaussianArea = scales_np[:, 0] * scales_np[:, 1]
    areaThreshold = 0.05 * 0.05  # heuristic for "big" surfels

    densifyIndices = np.where(gaussianArea > areaThreshold)[0]
    if densifyIndices.size == 0:
        return None

    # --------------------------------------------------------------
    # 2) Define how to split parents
    # --------------------------------------------------------------
    parentScaleShrinkFactor = 1.0
    childScaleShrinkFactor = 1.0

    newPositions = []
    newTangentU = []
    newTangentV = []
    newScales = []
    newColors = []

    rng = np.random.default_rng()

    # For updated parents
    updatedParentIndices = []
    updatedParentScales = []

    for parentIndex in densifyIndices:
        parentPosition = positions_np[parentIndex]
        parentTangentU = tangent_u_np[parentIndex]
        parentTangentV = tangent_v_np[parentIndex]
        parentScale = scales_np[parentIndex]  # (su, sv)
        parentColor = colors_np[parentIndex]

        # New parent scale (shrink)
        parentNewScale = parentScale * parentScaleShrinkFactor

        # Random offset inside original footprint (use original scale for offset)
        randomU = rng.uniform(-0.5, 0.5) * 4.0
        randomV = rng.uniform(-0.5, 0.5) * 4.0
        offsetVector = (
            randomU * parentScale[0] * parentTangentU
            + randomV * parentScale[1] * parentTangentV
        )

        childPosition = parentPosition + offsetVector
        childTangentU = parentTangentU.copy()
        childTangentV = parentTangentV.copy()
        childScale = parentScale * childScaleShrinkFactor
        childColor = parentColor.copy()

        # Record updated parent
        updatedParentIndices.append(parentIndex)
        updatedParentScales.append(parentNewScale)

        # Record new child
        newPositions.append(childPosition)
        newTangentU.append(childTangentU)
        newTangentV.append(childTangentV)
        newScales.append(childScale)
        newColors.append(childColor)

    updatedParentIndicesNp = np.asarray(updatedParentIndices, dtype=np.int64)
    updatedParentScalesNp = np.asarray(updatedParentScales, dtype=np.float32)

    newPositionsNp = np.asarray(newPositions, dtype=np.float32)
    newTangentUNp = np.asarray(newTangentU, dtype=np.float32)
    newTangentVNp = np.asarray(newTangentV, dtype=np.float32)
    newScalesNp = np.asarray(newScales, dtype=np.float32)
    newColorsNp = np.asarray(newColors, dtype=np.float32)

    return {
        "updated": {
            "indices": updatedParentIndicesNp,
            "scale": updatedParentScalesNp,
        },
        "new": {
            "position": newPositionsNp,
            "tangent_u": newTangentUNp,
            "tangent_v": newTangentVNp,
            "scale": newScalesNp,
            "color": newColorsNp,
        },
    }
