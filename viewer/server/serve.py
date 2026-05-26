#!/usr/bin/env python3
"""
H-SegSplat localhost query server.

Loads a gaussians.pt (output of colab_hsegsplat_inference.py) once at startup,
exposes one POST endpoint that takes up to three text terms (child / parent /
grandparent), computes per-Gaussian LERF relevancy at each provided level
against the level's SigLIP bank, AND-combines across provided levels, and
returns a binary float32 array of length G (one relevancy per Gaussian).

Used by the Unity viewer in second_stage/viewer/unity_scripts/.

Run:
    python serve.py /path/to/gaussians.pt
"""

import argparse
import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import open_clip


CANONICAL_PHRASES = ["object", "things", "stuff"]
TAU = 100.0  # LERF temperature (SegSplat eq. 4)


# ----------------------------------------------------------------------------
# Server state
# ----------------------------------------------------------------------------

class ServerState:
    def __init__(self, gaussians_pt_path: str,
                 model_name: str = "ViT-SO400M-14-SigLIP",
                 pretrained: str = "webli"):
        print(f"[serve] Loading gaussians.pt: {gaussians_pt_path}")
        g = torch.load(gaussians_pt_path, map_location="cpu", weights_only=False)

        # Per-level cluster IDs and L2-normalized banks.
        # cluster_index[lvl] : (G,) int64 in [0, M_lvl]; row 0 is background.
        # banks[lvl]         : (M_lvl + 1, D); row 0 zeros; rows 1..M are unit vectors.
        self.levels = list(g["levels"])
        self.cluster_index = {int(lvl): g["cluster_index"][lvl].cpu().numpy().astype(np.int64)
                              for lvl in self.levels}
        self.banks = {int(lvl): g["banks"][lvl].cpu().numpy().astype(np.float32)
                      for lvl in self.levels}
        self.M = {int(lvl): int(g["M"][lvl]) for lvl in self.levels}
        self.G = int(self.cluster_index[self.levels[0]].shape[0])
        D = int(self.banks[self.levels[0]].shape[1])
        for lvl in self.levels:
            assert self.cluster_index[lvl].shape == (self.G,)
            assert self.banks[lvl].shape == (self.M[lvl] + 1, D)
        print(f"[serve] Loaded G={self.G}, levels={self.levels}, "
              f"M={ {l: self.M[l] for l in self.levels} }, D={D}")

        # --- v2 payload (optional) ---
        # finest_level: (G,) int8 — 0 = bg, 1/3/6 = the Gaussian's finest mask level
        # finest_local_mask_id: (G,) int32, -1 = bg
        # finest_global_mask_id: (G,) int32, -1 = bg
        # mask_features[lvl]: (N_L, D) float32, L2-normalized
        # parents: dict of {"parents": {"level_6": {gid: {level_3, level_1}}, "level_3": ...}}
        self.has_v2 = "v2" in g
        if self.has_v2:
            v2 = g["v2"]
            self.finest_level = v2["finest_level"].cpu().numpy().astype(np.int8)
            self.finest_local = v2["finest_local_mask_id"].cpu().numpy().astype(np.int64)
            self.finest_global = v2["finest_global_mask_id"].cpu().numpy().astype(np.int64)
            self.mask_features = {
                int(lvl): v2["mask_features"][lvl].cpu().numpy().astype(np.float32)
                for lvl in self.levels
            }
            # Parent gather tables: for each (child_level, parent_level), a numpy array
            # `parent_lut[child_global_id] = parent_global_id`, or -1 if no link.
            # Built once at load time so queries are pure gather + compare.
            self.parent_lut: dict[tuple[int, int], np.ndarray] = {}
            parents_blob = v2.get("parents")
            if parents_blob is not None:
                parents = parents_blob.get("parents", {})
                # 6 -> 3
                n6 = self.mask_features[6].shape[0]
                lut_6_3 = np.full(n6, -1, dtype=np.int64)
                lut_6_1 = np.full(n6, -1, dtype=np.int64)
                for gid_str, links in parents.get("level_6", {}).items():
                    gid = int(gid_str)
                    if links.get("level_3") is not None:
                        lut_6_3[gid] = int(links["level_3"])
                    if links.get("level_1") is not None:
                        lut_6_1[gid] = int(links["level_1"])
                self.parent_lut[(6, 3)] = lut_6_3
                self.parent_lut[(6, 1)] = lut_6_1
                # 3 -> 1
                n3 = self.mask_features[3].shape[0]
                lut_3_1 = np.full(n3, -1, dtype=np.int64)
                for gid_str, links in parents.get("level_3", {}).items():
                    gid = int(gid_str)
                    if links.get("level_1") is not None:
                        lut_3_1[gid] = int(links["level_1"])
                self.parent_lut[(3, 1)] = lut_3_1
                print(f"[serve] v2 enabled: finest level distribution "
                      f"6={int((self.finest_level == 6).sum())}, "
                      f"3={int((self.finest_level == 3).sum())}, "
                      f"1={int((self.finest_level == 1).sum())}, "
                      f"bg={int((self.finest_level == 0).sum())}; "
                      f"parent links 6→3={int((lut_6_3 >= 0).sum())}/{n6}, "
                      f"6→1={int((lut_6_1 >= 0).sum())}/{n6}, "
                      f"3→1={int((lut_3_1 >= 0).sum())}/{n3}")
            else:
                print("[serve] v2 enabled but parents dict is None — only 1-term queries usable")
        else:
            print("[serve] v2 payload not present in gaussians.pt — /query_combined_v2 disabled")

        # SigLIP for text encoding.
        print(f"[serve] Loading {model_name} ({pretrained})")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().to(device)
        self.D = D

        # Pre-encode canonicals once.
        print("[serve] Encoding canonical phrases")
        self.phi_canonical = self._encode_text(CANONICAL_PHRASES)  # (3, D) float32 unit

        # Tiny text cache to avoid re-encoding the same term across queries.
        self._text_cache: dict[str, np.ndarray] = {}

    def _encode_text(self, texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            emb = self.model.encode_text(tokens).float()
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return emb.cpu().numpy()

    def get_term_embedding(self, text: str) -> np.ndarray:
        text = text.strip()
        if text not in self._text_cache:
            self._text_cache[text] = self._encode_text([text])[0]
        return self._text_cache[text]


# ----------------------------------------------------------------------------
# Per-level LERF relevancy in cluster space, then scatter to Gaussian space.
# ----------------------------------------------------------------------------

def lerf_per_gaussian(state: ServerState, term: str, level: int) -> np.ndarray:
    """Returns relevancy in [0, 1] of shape (G,) for `term` at `level`.

    Background Gaussians (cluster id 0) get relevancy 0 — they don't belong to
    any cluster, so the query can't fire on them.
    """
    if level not in state.cluster_index:
        raise ValueError(f"unknown level {level}; have {state.levels}")

    bank = state.banks[level]                # (M+1, D), row 0 = zeros
    # Drop the background row; compute relevancy per real cluster only.
    real_bank = bank[1:]                     # (M, D)
    phi_q = state.get_term_embedding(term)   # (D,)

    sim_q = real_bank @ phi_q                # (M,)
    sim_c = real_bank @ state.phi_canonical.T  # (M, C)
    diff = TAU * (sim_q[:, None] - sim_c)    # (M, C)
    pair = 1.0 / (1.0 + np.exp(-diff))       # (M, C)
    cluster_rel = pair.min(axis=1)           # (M,)

    # Scatter to per-Gaussian. cluster_index = 0 (background) -> 0.0.
    rel_padded = np.concatenate([[0.0], cluster_rel]).astype(np.float32)  # (M+1,)
    return rel_padded[state.cluster_index[level]]                          # (G,)


def combine_terms(state: ServerState, terms: list[tuple[str, int]]) -> np.ndarray:
    """Per-Gaussian AND across provided (term, level) pairs by elementwise min."""
    if not terms:
        return np.zeros(state.G, dtype=np.float32)
    rels = [lerf_per_gaussian(state, t, lvl) for t, lvl in terms]
    out = rels[0]
    for r in rels[1:]:
        out = np.minimum(out, r)
    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# v2: per-mask LERF + finest-mask gather + parent walk
# ----------------------------------------------------------------------------

def lerf_per_mask(state: ServerState, term: str, level: int) -> np.ndarray:
    """Returns LERF relevancy in (0, 1) of shape (N_L,) — one score per mask at the level.
    Uses the per-mask SigLIP features (unit-norm) instead of bank centroids. This is the
    v2 scoring primitive."""
    feats = state.mask_features[level]                # (N_L, D), unit-norm
    phi_q = state.get_term_embedding(term)            # (D,)
    sim_q = feats @ phi_q                             # (N_L,)
    sim_c = feats @ state.phi_canonical.T             # (N_L, C)
    diff = TAU * (sim_q[:, None] - sim_c)             # (N_L, C)
    pair = 1.0 / (1.0 + np.exp(-diff))                # (N_L, C)
    return pair.min(axis=1).astype(np.float32)        # (N_L,)


def _gather_per_gaussian(per_mask_score: np.ndarray,
                         finest_global: np.ndarray,
                         finest_level: np.ndarray,
                         restrict_to_level: int) -> np.ndarray:
    """For each Gaussian whose finest_level == restrict_to_level, look up its per-mask
    score via finest_global_mask_id. Gaussians at any other level get 0 (failure for this
    binding). Returns (G,) float32."""
    G = finest_level.shape[0]
    out = np.zeros(G, dtype=np.float32)
    mask = (finest_level == restrict_to_level)
    if not mask.any():
        return out
    gids = finest_global[mask]
    # Defensive: invalid (-1) should not appear when finest_level matches a real level.
    valid = gids >= 0
    if not valid.all():
        # Shouldn't happen; clamp + zero-out the invalid entries.
        gids = np.where(valid, gids, 0)
    scores = per_mask_score[gids]
    if not valid.all():
        scores = np.where(valid, scores, 0.0)
    out[mask] = scores
    return out


def _gather_through_parent(per_mask_parent_score: np.ndarray,
                           finest_global: np.ndarray,
                           finest_level: np.ndarray,
                           child_level: int,
                           parent_lut: np.ndarray) -> np.ndarray:
    """For each Gaussian whose finest_level == child_level, walk to its parent via
    parent_lut[child_global_id]; gather the parent's per-mask score. Gaussians at other
    child levels or with no parent link get 0. Returns (G,) float32."""
    G = finest_level.shape[0]
    out = np.zeros(G, dtype=np.float32)
    mask = (finest_level == child_level)
    if not mask.any():
        return out
    child_gids = finest_global[mask]
    parent_gids = parent_lut[child_gids]              # -1 where no link
    has_parent = parent_gids >= 0
    if not has_parent.any():
        return out
    parent_gids_safe = np.where(has_parent, parent_gids, 0)
    scores = per_mask_parent_score[parent_gids_safe]
    scores = np.where(has_parent, scores, 0.0)
    out[mask] = scores
    return out


def evaluate_v2(state: ServerState,
                child: str | None,
                parent: str | None,
                grandparent: str | None) -> np.ndarray:
    """Implements the fluid level-binding rule (PROJECT_PLAN.md §6.5.5).
    Returns (G,) float32 relevancy in [0, 1].
    """
    if not state.has_v2:
        raise RuntimeError("v2 payload not loaded — re-run colab_hsegsplat_inference.py "
                           "with the v2-extended build outputs.")
    # Normalize: treat empty/whitespace strings as missing terms.
    child = child.strip() if child and child.strip() else None
    parent = parent.strip() if parent and parent.strip() else None
    grandparent = grandparent.strip() if grandparent and grandparent.strip() else None
    n_terms = sum(1 for t in (child, parent, grandparent) if t)
    if n_terms == 0:
        return np.zeros(state.G, dtype=np.float32)
    if child is None:
        # v2 requires child to be the leading term (see §6.5.5).
        raise RuntimeError("v2 requires `child` to be set; parent/grandparent without child "
                           "are not supported.")

    # Pre-compute per-mask LERF scores for the provided terms at all levels they might
    # bind to. (We only encode each text term once thanks to the term cache.)
    def per_mask_at(term: str, level: int) -> np.ndarray:
        return lerf_per_mask(state, term, level)

    if n_terms == 1:
        # Child only — match against the Gaussian's own finest mask, regardless of level.
        # Compute per-mask LERF at each level and gather using finest_global at that level.
        out = np.zeros(state.G, dtype=np.float32)
        for lvl in state.levels:
            r = per_mask_at(child, lvl)
            out = np.maximum(out, _gather_per_gaussian(
                r, state.finest_global, state.finest_level, lvl))
        return out

    if n_terms == 3:
        # Locked binding: child=6, parent=3, grandparent=1.
        # Strict — no fallbacks.
        r_child = per_mask_at(child, 6)
        r_parent = per_mask_at(parent, 3)
        r_grand = per_mask_at(grandparent, 1)

        s_child = _gather_per_gaussian(r_child, state.finest_global, state.finest_level, 6)
        # Parent: Gaussian at lvl 6 walks via 6→3 link.
        s_parent = _gather_through_parent(r_parent, state.finest_global, state.finest_level,
                                          6, state.parent_lut[(6, 3)])
        # Grandparent: Gaussian at lvl 6 walks via 6→1 link directly.
        s_grand = _gather_through_parent(r_grand, state.finest_global, state.finest_level,
                                         6, state.parent_lut[(6, 1)])
        # Soft AND inside the binding: min of the three scores.
        out = np.minimum(np.minimum(s_child, s_parent), s_grand)
        return out.astype(np.float32)

    # n_terms == 2 → child + parent. Try three sub-bindings, take max.
    # (a) 6→3
    # (b) 6→1
    # (c) 3→1
    bindings: list[np.ndarray] = []
    # (a) child at lvl 6, parent at lvl 3 via 6→3
    r_child_6 = per_mask_at(child, 6)
    r_parent_3 = per_mask_at(parent, 3)
    s_a_child = _gather_per_gaussian(r_child_6, state.finest_global, state.finest_level, 6)
    s_a_parent = _gather_through_parent(r_parent_3, state.finest_global, state.finest_level,
                                        6, state.parent_lut[(6, 3)])
    bindings.append(np.minimum(s_a_child, s_a_parent))

    # (b) child at lvl 6, parent at lvl 1 via 6→1
    r_parent_1 = per_mask_at(parent, 1)
    s_b_child = s_a_child  # same gather, reused
    s_b_parent = _gather_through_parent(r_parent_1, state.finest_global, state.finest_level,
                                        6, state.parent_lut[(6, 1)])
    bindings.append(np.minimum(s_b_child, s_b_parent))

    # (c) child at lvl 3, parent at lvl 1 via 3→1
    r_child_3 = per_mask_at(child, 3)
    s_c_child = _gather_per_gaussian(r_child_3, state.finest_global, state.finest_level, 3)
    s_c_parent = _gather_through_parent(r_parent_1, state.finest_global, state.finest_level,
                                        3, state.parent_lut[(3, 1)])
    bindings.append(np.minimum(s_c_child, s_c_parent))

    out = bindings[0]
    for b in bindings[1:]:
        out = np.maximum(out, b)
    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# FastAPI
# ----------------------------------------------------------------------------

app = FastAPI()


def get_state() -> ServerState:
    return app.state.server_state


class QueryInput(BaseModel):
    child: str
    parent: Optional[str] = None
    grandparent: Optional[str] = None
    child_level: int = 6
    parent_level: int = 3
    grandparent_level: int = 1


def collect_terms(q: QueryInput) -> list[tuple[str, int]]:
    """Returns the (term, level) list with only non-empty terms."""
    out: list[tuple[str, int]] = []
    if q.child and q.child.strip():
        out.append((q.child.strip(), q.child_level))
    if q.parent and q.parent.strip():
        out.append((q.parent.strip(), q.parent_level))
    if q.grandparent and q.grandparent.strip():
        out.append((q.grandparent.strip(), q.grandparent_level))
    levels = [lvl for _, lvl in out]
    if len(set(levels)) != len(levels):
        raise ValueError(f"levels collide: {levels}")
    return out


@app.post("/query_combined")
def query_combined(q: QueryInput, state: ServerState = Depends(get_state)):
    try:
        terms = collect_terms(q)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    if not terms:
        return JSONResponse(status_code=400, content={"error": "no terms provided"})

    print(f"[query] terms={terms}")
    rel = combine_terms(state, terms)  # (G,) float32 in [0, 1]
    print(f"[query]   min={rel.min():.4f} max={rel.max():.4f} "
          f"mean={rel.mean():.4f}  N>=0.5={int((rel >= 0.5).sum())}")

    buf = rel.tobytes()
    return StreamingResponse(io.BytesIO(buf), media_type="application/octet-stream")


@app.post("/query_combined_v2")
def query_combined_v2(q: QueryInput, state: ServerState = Depends(get_state)):
    """v2 query: fluid level binding by query shape (see PROJECT_PLAN.md §6.5.5).
    Ignores `*_level` fields — bindings are determined by which terms are provided.
    """
    if not state.has_v2:
        return JSONResponse(status_code=400, content={
            "error": "v2 payload not loaded; gaussians.pt was built without v2 outputs."
        })
    child = (q.child or "").strip()
    parent = (q.parent or "").strip()
    grandparent = (q.grandparent or "").strip()
    if not child and not parent and not grandparent:
        return JSONResponse(status_code=400, content={"error": "no terms provided"})
    # v2 requires `child` to be the leading term. A query of (parent only) or (grandparent
    # only) doesn't have a well-defined binding under the fluid rules.
    if not child:
        return JSONResponse(status_code=400, content={
            "error": "v2 requires `child` to be set; parent/grandparent without child are not supported"
        })

    try:
        rel = evaluate_v2(state,
                          child or None,
                          parent or None,
                          grandparent or None)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    n_terms = sum(1 for t in (child, parent, grandparent) if t)
    print(f"[query/v2] n_terms={n_terms}  child={child!r} parent={parent!r} grandparent={grandparent!r}")
    print(f"[query/v2]   min={rel.min():.4f} max={rel.max():.4f} "
          f"mean={rel.mean():.4f}  N>=0.5={int((rel >= 0.5).sum())}")
    buf = rel.tobytes()
    return StreamingResponse(io.BytesIO(buf), media_type="application/octet-stream")


@app.post("/extrema")
def extrema(q: QueryInput, state: ServerState = Depends(get_state)):
    try:
        terms = collect_terms(q)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    if not terms:
        return JSONResponse(status_code=400, content={"error": "no terms provided"})
    rel = combine_terms(state, terms)
    return {"min": float(rel.min()), "max": float(rel.max()),
            "n_gaussians": int(rel.size),
            "n_above_0_5": int((rel >= 0.5).sum())}


@app.get("/info")
def info(state: ServerState = Depends(get_state)):
    out = {
        "G": state.G,
        "levels": state.levels,
        "M": state.M,
        "D": state.D,
        "has_v2": state.has_v2,
    }
    if state.has_v2:
        out["v2"] = {
            "finest_level_counts": {
                "0": int((state.finest_level == 0).sum()),
                "1": int((state.finest_level == 1).sum()),
                "3": int((state.finest_level == 3).sum()),
                "6": int((state.finest_level == 6).sum()),
            },
            "mask_counts": {lvl: int(state.mask_features[lvl].shape[0])
                            for lvl in state.levels},
            "parent_links": {
                "6->3": int((state.parent_lut[(6, 3)] >= 0).sum()) if (6, 3) in state.parent_lut else 0,
                "6->1": int((state.parent_lut[(6, 1)] >= 0).sum()) if (6, 1) in state.parent_lut else 0,
                "3->1": int((state.parent_lut[(3, 1)] >= 0).sum()) if (3, 1) in state.parent_lut else 0,
            },
        }
    return out


# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gaussians", help="Path to gaussians.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app.state.server_state = ServerState(args.gaussians)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
