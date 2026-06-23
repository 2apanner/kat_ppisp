# Katada PPISP engine

Fork of [nv-tlabs/ppisp](https://github.com/nv-tlabs/ppisp) with Katada pilot integration.

## What it adds

- `katada/splatfacto_ppisp.py` — Nerfstudio method **`splatfacto-ppisp`**
- Applies PPISP (exposure, vignetting, color, CRF) during splatfacto training
- Single-camera default (`num_cameras=1`) for DJI drone orbits

## Local layout

```
production/dev/engines/kat_ppisp/
  ppisp/           # upstream CUDA PPISP module
  katada/          # Katada nerfstudio integration
  KATADA.md
```

## Colab / pilot

Pilot runner installs this repo and runs:

```bash
ns-train splatfacto-ppisp --data ... --output-dir ...
```

Select **VGGT + 3DGS + PPISP** in the pilot web UI (`model=ppisp`).

## Mac workflow

```bash
cd production/dev/engines/kat_ppisp
# edit katada/ or ppisp/
./scripts/push_ppisp.sh   # from pilot/colab/scripts
```

Colab clones `https://github.com/2apanner/kat_ppisp.git` on next run.

## Build

```bash
pip install . --no-build-isolation
```

Requires CUDA PyTorch (Colab T4/A100).
