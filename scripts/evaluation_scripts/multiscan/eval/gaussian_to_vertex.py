#!/usr/bin/env python3
"""
Lift per-Gaussian scores to per-vertex (PLY) scores via nearest-neighbor.

Given:
    means          (G, 3)  float — Gaussian centers in the same world frame as the PLY
    vertex_xyz     (N, 3)  float — PLY vertex positions
    per_g_score    (G,)    float — score per Gaussian (e.g. LERF relevancy for one query)

Returns:
    per_v_score    (N,)    float — score per vertex, using nearest-Gaussian assignment.

This is the minimal bridge between H-SegSplat's per-Gaussian representation and
Search3D's per-vertex predicted-mask AP evaluation. Uses scikit-learn's NearestNeighbors
KD-tree under the hood (fast for the scales we care about: G ≈ 1.2M, N ≈ 80k–400k).

Also exposes a helper that combines a list of per-query scores into one prediction-set
in Search3D's expected format (pred_classes, pred_scores, pred_masks).
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_kdtree(means_xyz: np.ndarray) -> NearestNeighbors:
    """Build a kNN tree over Gaussian centers."""
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn.fit(means_xyz)
    return nn


def assign_gaussian_per_vertex(nn: NearestNeighbors,
                                vertex_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (assignment, distance) where assignment[i] is the index of the
    Gaussian closest to vertex i, distance[i] is the distance to that Gaussian
    in world units."""
    dist, idx = nn.kneighbors(vertex_xyz, n_neighbors=1, return_distance=True)
    return idx[:, 0].astype(np.int64), dist[:, 0].astype(np.float32)


def per_g_to_per_v(per_g: np.ndarray, gauss_per_v: np.ndarray) -> np.ndarray:
    """Scatter per-Gaussian scores onto per-vertex via the precomputed assignment."""
    return per_g[gauss_per_v].astype(np.float32)


def build_search3d_prediction(per_class_per_v_scores: dict[int, np.ndarray],
                              score_threshold: float = 0.5,
                              max_distance: np.ndarray | None = None,
                              max_distance_cutoff: float = 0.3) -> dict:
    """Build Search3D's eval-input dict from per-class per-vertex score maps.

    per_class_per_v_scores : {class_id (int joint semantic id): (N,) float relevancy}
    score_threshold        : binarize scores above this to form the instance mask
    max_distance           : optional (N,) float — distance from each vertex to its
                             nearest Gaussian; vertices farther than max_distance_cutoff
                             are zeroed out (no Gaussian to predict from).

    Returns:
        {
            "pred_classes": (N_inst,) int   — VALID_JOINT_SEMANTIC_IDS values
            "pred_scores" : (N_inst,) float — confidence per instance (max relevancy)
            "pred_masks"  : (N_verts, N_inst) bool — binary per-instance vertex masks
        }

    We emit ONE instance per class (the union of vertices passing threshold).
    This collapses instance-level recall (multiple physical instances of a class
    become one prediction) but is the most we can do with the v1/v2 cluster-based
    representation. Multi-instance separation would require connected components
    or per-mask-ID predictions (PROJECT_PLAN.md §6.5.7 future work).
    """
    # Find any classes that have at least one vertex above threshold.
    pred_classes_list = []
    pred_scores_list = []
    pred_masks_cols = []
    n_verts = None
    for cls_id, sc in per_class_per_v_scores.items():
        if n_verts is None:
            n_verts = sc.shape[0]
        assert sc.shape == (n_verts,)
        m = sc >= score_threshold
        if max_distance is not None:
            m &= (max_distance <= max_distance_cutoff)
        if not m.any():
            continue
        conf = float(sc[m].max())
        pred_classes_list.append(int(cls_id))
        pred_scores_list.append(conf)
        pred_masks_cols.append(m.astype(np.uint8))
    if not pred_classes_list:
        # No predictions — Search3D's evaluator handles empty preds fine.
        return {
            "pred_classes": np.zeros(0, dtype=np.int64),
            "pred_scores": np.zeros(0, dtype=np.float32),
            "pred_masks": np.zeros((n_verts or 0, 0), dtype=np.uint8),
        }
    return {
        "pred_classes": np.array(pred_classes_list, dtype=np.int64),
        "pred_scores": np.array(pred_scores_list, dtype=np.float32),
        "pred_masks": np.stack(pred_masks_cols, axis=1),  # (N, N_inst)
    }
