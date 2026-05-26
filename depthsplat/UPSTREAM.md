# DepthSplat — vendored upstream snapshot

This directory is a frozen, unmodified copy of [cvg/depthsplat](https://github.com/cvg/depthsplat).

| Field | Value |
|---|---|
| Source | https://github.com/cvg/depthsplat |
| Commit SHA | `2dad25a` ("Add ReSplat update") |
| Branch | `main` |
| Vendored on | 2026-05-26 |
| Modifications | **None.** Working tree clean at vendor time, zero ahead/behind upstream `main`. |

## Why vendored

DepthSplat is used as a frozen geometric backbone (see `docs/PROJECT_PLAN.md` §4).
Vendoring keeps everything in one `git clone` and removes the network dependency at setup time.

## Re-vendoring (if upstream updates and we ever want to pick them up)

```bash
rm -rf depthsplat
git clone https://github.com/cvg/depthsplat.git
rm -rf depthsplat/.git
# Update the SHA above.
```

## Do not edit files in this directory

If we ever need to patch DepthSplat (we currently don't — see §4 of the plan), do it as a
clearly-marked patch file under `scripts/patches/` rather than by editing files here.
That keeps the vendored copy diffable against upstream.
