#!/usr/bin/env python3
"""
Minimal offline copy of viewer/server/serve.py's ServerState + v2 evaluator,
without the FastAPI server. Loads gaussians.pt once, encodes text terms with
SigLIP once, scores arbitrary v2 fluid-binding queries.

Used by the MultiScan AP eval driver to score each of Search3D's 47
(object, part) tuples per scene.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


CANONICAL_PHRASES = ["object", "things", "stuff"]
TAU = 100.0  # LERF temperature


class OfflineState:
    """Loads gaussians.pt (v1 + v2 payload) and runs per-Gaussian fluid-binding queries."""

    def __init__(self, gaussians_pt: Path, device: str = "cpu"):
        g = torch.load(gaussians_pt, map_location="cpu", weights_only=False)
        self.device = device

        self.means = g["means"].cpu().numpy().astype(np.float32)        # (G, 3)
        self.opacities = g["opacities"].cpu().numpy().astype(np.float32)
        self.levels = list(g["levels"])
        self.cluster_index = {int(lvl): g["cluster_index"][lvl].cpu().numpy().astype(np.int64)
                              for lvl in self.levels}
        self.banks = {int(lvl): g["banks"][lvl].cpu().numpy().astype(np.float32)
                      for lvl in self.levels}
        self.M = {int(lvl): int(g["M"][lvl]) for lvl in self.levels}
        self.G = int(self.cluster_index[self.levels[0]].shape[0])
        D = int(self.banks[self.levels[0]].shape[1])
        self.D = D

        # v2 payload
        self.has_v2 = "v2" in g
        if not self.has_v2:
            raise RuntimeError("gaussians.pt missing v2 payload — re-run inference.")
        v2 = g["v2"]
        self.finest_level = v2["finest_level"].cpu().numpy().astype(np.int8)
        self.finest_global = v2["finest_global_mask_id"].cpu().numpy().astype(np.int64)
        self.mask_features = {
            int(lvl): (v2["mask_features"][lvl].cpu().numpy().astype(np.float32)
                       if isinstance(v2["mask_features"][lvl], torch.Tensor)
                       else np.asarray(v2["mask_features"][lvl], dtype=np.float32))
            for lvl in self.levels
        }

        # Parent LUTs from parents.json
        self.parent_lut: dict[tuple[int, int], np.ndarray] = {}
        parents = v2.get("parents", None)
        if parents is not None:
            p6 = parents.get("parents", {}).get("level_6", {})
            p3 = parents.get("parents", {}).get("level_3", {})
            n6 = self.mask_features[6].shape[0]
            n3 = self.mask_features[3].shape[0]
            lut_6_3 = np.full(n6, -1, dtype=np.int64)
            lut_6_1 = np.full(n6, -1, dtype=np.int64)
            lut_3_1 = np.full(n3, -1, dtype=np.int64)
            for gid, links in p6.items():
                gid_i = int(gid)
                if links.get("level_3") is not None:
                    lut_6_3[gid_i] = int(links["level_3"])
                if links.get("level_1") is not None:
                    lut_6_1[gid_i] = int(links["level_1"])
            for gid, links in p3.items():
                gid_i = int(gid)
                if links.get("level_1") is not None:
                    lut_3_1[gid_i] = int(links["level_1"])
            self.parent_lut[(6, 3)] = lut_6_3
            self.parent_lut[(6, 1)] = lut_6_1
            self.parent_lut[(3, 1)] = lut_3_1

        print(f"[offline] Loaded G={self.G}, levels={self.levels}, D={D}; "
              f"finest 6/3/1/bg = "
              f"{int((self.finest_level==6).sum())}/"
              f"{int((self.finest_level==3).sum())}/"
              f"{int((self.finest_level==1).sum())}/"
              f"{int((self.finest_level==0).sum())}")


# ----- SigLIP text encoder shared across scenes -----

class SigLIPTextEncoder:
    def __init__(self, model_name="ViT-SO400M-14-SigLIP", pretrained="webli",
                 device: str = "cpu"):
        import open_clip
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
        self.model.eval().to(device)
        self.phi_canonical = self._encode(CANONICAL_PHRASES)  # (3, D) unit
        self._cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        emb = self.model.encode_text(tokens).float()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return emb.cpu().numpy()

    def __call__(self, term: str) -> np.ndarray:
        if term not in self._cache:
            self._cache[term] = self._encode([term])[0]
        return self._cache[term]


# ----- LERF + gather + fluid binding (lifted from serve.py, made stateless) -----

def lerf_per_mask(feats: np.ndarray, phi_q: np.ndarray, phi_canon: np.ndarray) -> np.ndarray:
    """feats: (N_L, D) unit-norm; phi_q: (D,); phi_canon: (C, D). Returns (N_L,) LERF."""
    sim_q = feats @ phi_q
    sim_c = feats @ phi_canon.T
    diff = TAU * (sim_q[:, None] - sim_c)
    pair = 1.0 / (1.0 + np.exp(-diff))
    return pair.min(axis=1).astype(np.float32)


def gather_per_gaussian(per_mask_score: np.ndarray, finest_global: np.ndarray,
                        finest_level: np.ndarray, restrict_to_level: int) -> np.ndarray:
    G = finest_level.shape[0]
    out = np.zeros(G, dtype=np.float32)
    mask = (finest_level == restrict_to_level)
    if not mask.any():
        return out
    gids = finest_global[mask]
    valid = gids >= 0
    gids_safe = np.where(valid, gids, 0)
    scores = per_mask_score[gids_safe]
    scores = np.where(valid, scores, 0.0)
    out[mask] = scores
    return out


def gather_through_parent(per_mask_parent_score: np.ndarray, finest_global: np.ndarray,
                          finest_level: np.ndarray, child_level: int,
                          parent_lut: np.ndarray) -> np.ndarray:
    G = finest_level.shape[0]
    out = np.zeros(G, dtype=np.float32)
    mask = (finest_level == child_level)
    if not mask.any():
        return out
    child_gids = finest_global[mask]
    parent_gids = parent_lut[child_gids]
    has_parent = parent_gids >= 0
    if not has_parent.any():
        return out
    parent_gids_safe = np.where(has_parent, parent_gids, 0)
    scores = per_mask_parent_score[parent_gids_safe]
    scores = np.where(has_parent, scores, 0.0)
    out[mask] = scores
    return out


def evaluate_v2_two_terms(state: OfflineState, encoder: SigLIPTextEncoder,
                          child: str, parent: str) -> np.ndarray:
    """Fluid two-term v2 query: tries 6→3, 6→1, 3→1 bindings; per-Gaussian max."""
    phi_c = encoder.phi_canonical
    phi_child = encoder(child)
    phi_parent = encoder(parent)

    out = np.zeros(state.G, dtype=np.float32)

    # 6→3
    if (6, 3) in state.parent_lut and 3 in state.mask_features and 6 in state.mask_features:
        r6 = lerf_per_mask(state.mask_features[6], phi_child, phi_c)
        r3 = lerf_per_mask(state.mask_features[3], phi_parent, phi_c)
        sc = gather_per_gaussian(r6, state.finest_global, state.finest_level, 6)
        sp = gather_through_parent(r3, state.finest_global, state.finest_level, 6,
                                   state.parent_lut[(6, 3)])
        out = np.maximum(out, np.minimum(sc, sp))

    # 6→1
    if (6, 1) in state.parent_lut and 1 in state.mask_features and 6 in state.mask_features:
        r6 = lerf_per_mask(state.mask_features[6], phi_child, phi_c)
        r1 = lerf_per_mask(state.mask_features[1], phi_parent, phi_c)
        sc = gather_per_gaussian(r6, state.finest_global, state.finest_level, 6)
        sp = gather_through_parent(r1, state.finest_global, state.finest_level, 6,
                                   state.parent_lut[(6, 1)])
        out = np.maximum(out, np.minimum(sc, sp))

    # 3→1
    if (3, 1) in state.parent_lut and 1 in state.mask_features and 3 in state.mask_features:
        r3 = lerf_per_mask(state.mask_features[3], phi_child, phi_c)
        r1 = lerf_per_mask(state.mask_features[1], phi_parent, phi_c)
        sc = gather_per_gaussian(r3, state.finest_global, state.finest_level, 3)
        sp = gather_through_parent(r1, state.finest_global, state.finest_level, 3,
                                   state.parent_lut[(3, 1)])
        out = np.maximum(out, np.minimum(sc, sp))

    return out
