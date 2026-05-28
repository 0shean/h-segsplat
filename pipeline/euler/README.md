# Running H-SegSplat on ETH Euler

SLURM wrappers around the existing `pipeline/run_pipeline.sh`. All jobs request
`rtx_4090:1` and load `eth_proxy` so they have internet access (Euler compute
nodes are offline by default).

## One-time setup

```bash
# On laptop: upload staged zips
scp data/3D-OVS/staged_zips/*.zip data/Multiscan/staged_zips/*.zip \
    sergejsz@eu-login-27:/cluster/project/cvg/students/sergejsz/h-segsplat/data/

# On cluster:
ssh sergejsz@euler.ethz.ch
cd /cluster/project/cvg/students/sergejsz
git clone https://github.com/0shean/h-segsplat.git
cd h-segsplat/data
for z in *.zip; do unzip -q "$z" && rm "$z"; done
ls -d */         # verify 20 scene directories
cd ..
```

## Submit everything

```bash
bash pipeline/euler/submit_all.sh
```

This submits:

1. `setup_sam.sbatch` — builds `envs/sam/venv` (~10 min)
2. `setup_siglip.sbatch` — builds `envs/siglip/venv` (~5 min)
3. `setup_hsegsplat.sbatch` — builds `envs/hsegsplat/venv` (~10 min)
4. `download_checkpoints.sbatch` — `swinl_only_sam_many2many.pth` + DepthSplat (~2 min)
5. 20 × `run_scene.sbatch data/<scene>` with `afterok` dependency on the above

Monitor:

```bash
squeue -u $USER
watch squeue -u $USER     # ctrl-c to stop
```

## Re-running individual pieces

```bash
# Re-run a single scene (after the venvs + ckpts exist):
sbatch --job-name=hseg_3dovs_bed pipeline/euler/run_scene.sbatch data/3dovs_bed

# Submit only the scene jobs (skip the four setup/ckpt jobs):
bash pipeline/euler/submit_all.sh skip_setup
```

## After everything finishes

Per-stage timings land in:

```
data/<scene>/pipeline_timings.csv     # one row per stage, per scene
data/pipeline_timings.csv             # aggregate, appended across all runs
```

Pull the aggregate to your laptop:

```bash
scp sergejsz@eu-login-27:/cluster/project/cvg/students/sergejsz/h-segsplat/data/pipeline_timings.csv .
column -t -s, pipeline_timings.csv
```

Pull all gaussians + rendered feature maps:

```bash
rsync -avh --include='*/' \
    --include='gaussians.pt' \
    --include='rendered_*.npy' \
    --exclude='*' \
    sergejsz@eu-login-27:/cluster/project/cvg/students/sergejsz/h-segsplat/data/ \
    ./euler_outputs/
```

## Module set

All scripts use the same module load block (Euler stack 2024-06):

```
module load stack/2024-06
module load python/3.10.13
module load cuda/12.1.1
module load gcc/12.2.0
module load eth_proxy
```

The h-segsplat venv installs torch 2.4.0+cu124 wheels which are forward-compatible
with the cu121 module at runtime.

## Notes

- **Logs**: `logs/<jobname>_<jobid>.log` (stdout) and `.err` (stderr) in the repo root.
- **Internet on compute nodes**: Euler requires `module load eth_proxy` for any
  outbound traffic (pip, wget, huggingface). Every script here loads it.
- **Updating the pipeline**: edit locally, push to GitHub, then on the cluster:
  ```bash
  cd /cluster/project/cvg/students/sergejsz/h-segsplat
  git pull
  ```
