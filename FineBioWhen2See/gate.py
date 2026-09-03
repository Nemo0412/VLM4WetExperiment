"""Viewpoint-compensated residual and raw frame-difference gates."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

ScoreKind = Literal["p90", "median"]


def _gray_small(rgb: np.ndarray, max_side: int = 320) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(
            rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def estimate_homography(prev_rgb: np.ndarray, curr_rgb: np.ndarray):
    g1 = _gray_small(prev_rgb)
    g2 = _gray_small(curr_rgb)
    orb = cv2.ORB_create(nfeatures=800)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 16:
        return None
    matches = sorted(matches, key=lambda match: match.distance)[:100]
    pts1 = np.float32([k1[match.queryIdx].pt for match in matches])
    pts2 = np.float32([k2[match.trainIdx].pt for match in matches])
    homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    if homography is None or mask is None or int(mask.sum()) < 10:
        return None
    return homography


def _residual_score(
    a: np.ndarray, b: np.ndarray, valid: np.ndarray, kind: ScoreKind
) -> float:
    if int(valid.sum()) < 200:
        return 1.0
    pixels = np.abs(a.astype(np.float32) - b.astype(np.float32))[valid]
    if kind == "median":
        return float(np.median(pixels) / 255.0)
    return float(np.percentile(pixels, 90) / 255.0)


def reproj_score(
    prev_rgb: np.ndarray, curr_rgb: np.ndarray, kind: ScoreKind = "p90"
) -> float:
    """Return 1 when alignment fails; low values indicate a reprojected copy."""
    homography = estimate_homography(prev_rgb, curr_rgb)
    if homography is None:
        return 1.0
    prev_s = _gray_small(prev_rgb)
    curr_s = _gray_small(curr_rgb)
    h, w = curr_s.shape[:2]
    warped = cv2.warpPerspective(prev_s, homography, (w, h))
    valid = (warped > 8) & (curr_s > 8)
    return _residual_score(warped, curr_s, valid, kind)


def framediff_score(
    prev_rgb: np.ndarray, curr_rgb: np.ndarray, kind: ScoreKind = "p90"
) -> float:
    a = _gray_small(prev_rgb)
    b = _gray_small(curr_rgb)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return _residual_score(a, b, np.ones(a.shape, dtype=bool), kind)


def select_indices(
    frames: list[np.ndarray],
    *,
    mode: Literal["reproj", "framediff"],
    tau: float,
    kind: ScoreKind = "p90",
) -> list[int]:
    """Keep frame zero and frames sufficiently different from the last kept frame."""
    if not frames:
        return [0]
    keep = [0]
    last = frames[0]
    scorer = reproj_score if mode == "reproj" else framediff_score
    for i in range(1, len(frames)):
        if scorer(last, frames[i], kind=kind) >= tau:
            keep.append(i)
            last = frames[i]
    return keep
