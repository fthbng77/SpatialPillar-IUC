<div align="center">

# SpatialPillar-IUC

**Spatially-Enhanced 4D Radar 3D Object Detection on View-of-Delft**

*Developed at İstanbul Üniversitesi-Cerrahpaşa (IUC).*

</div>

---

## Headline

| Method | Car | Ped | Cyc | mAP_3D (R11) |
|---|:---:|:---:|:---:|:---:|
| MAFF-Net (PV-RCNN, 2025) | 42.3 | 46.8 | 74.7 | 54.6 |
| SCKD (2025) | 41.9 | 43.5 | 70.8 | 52.1 |
| **RadarPillars (ours, multi-seed best)** | **41.58** | **44.78** | 71.31 | **52.56** |
| RadarPillars (ours, 3-seed mean) | 41.02 | 43.15 | 70.12 | 51.43 ± 0.99 |
| SMURF (2023) | 42.3 | 39.1 | 71.5 | 51.0 |
| RadarPillars (paper, 2024) | 41.1 | 38.6 | 72.6 | 50.70 |
| CenterPoint (baseline) | 33.9 | 39.0 | 66.9 | 46.6 |
| PointPillars (baseline) | 37.9 | 31.2 | 65.7 | 45.0 |

VoD validation set, 3D AP (%) at IoU: Car=0.50, Ped/Cyc=0.25 (R11). The "(ours)" rows are a paper-faithful reproduction of RadarPillars (Section IV + the augmentor velocity fix from this repo + rotation augmentation), trained with three different random seeds; we report the best seed and the 3-seed mean ± std. This is the **baseline** that the SpatialPillar-IUC modules below are built on top of.

---

## Architecture

```
Radar pcd (N,7)
  → PillarVFE (voxelize + Doppler decomp: vx, vy via atan2)
  → PillarAttention (masked self-attention, C=E=32)
  → PointPillarScatter (320×320×32 BEV)
  → BaseBEVBackbone (3-block 2D CNN, uniform C=32)
  → AnchorHeadSingle (Car / Pedestrian / Cyclist)
```

Optional SpatialPillar modules stacked on top of the baseline:

- **GeoSPA** — Lalonde geometric descriptors (scatter/linear/surface) from KNN covariance, appended pre-voxelization.
- **CQCA** — DBSCAN velocity clustering + cluster-query cross-attention on pillar features.
- **DCNBEVBackbone** — deformable convolutions replacing the first conv in each BEV stage.
- **KDE Branch** — Gaussian kernel density side branch fused with BEV features.
- **CenterHead** — anchor-free heatmap detection head.

---

## Key Implementation Details

- Velocity decomposition in VFE: `vx = v_r_comp·cos(φ)`, `vy = v_r_comp·sin(φ)`, `φ = atan2(y, x)`.
- Physics-consistent augmentation: velocity vectors rotated/flipped with point coordinates (fixes a bug in OpenPCDet that assumes the nuScenes column layout).
- PillarAttention with key-padding mask so empty pillars do not poison attention.
- `FFN_CHANNELS` config-driven in `pillar_attention.py` (was hardcoded `*2` before).

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
python setup.py develop
```

Requirements: Python 3.8+, PyTorch 2.4+, CUDA 12.x, spconv 2.3.6.

---

## Data

```
data/VoD/view_of_delft_PUBLIC/radar_5frames/
  ├── ImageSets/{train,val,test}.txt
  ├── training/{velodyne,label_2,calib,image_2}/
  └── testing/velodyne/
```

Generate info pkl + GT db:
```bash
python -m pcdet.datasets.vod.vod_dataset create_vod_infos \
    tools/cfgs/dataset_configs/vod_dataset_radar.yaml
```

---

## Train

RadarPillars baseline reproduction (the 52.56 row above):
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --batch_size 8 --extra_tag <run_name> --workers 4
```

SpatialPillar variants share the same launcher, only the config changes:
```bash
--cfg_file tools/cfgs/vod_models/spatialpillar_geospa.yaml
--cfg_file tools/cfgs/vod_models/spatialpillar_cqca.yaml
--cfg_file tools/cfgs/vod_models/spatialpillar_dcn.yaml
--cfg_file tools/cfgs/vod_models/spatialpillar_kde.yaml
--cfg_file tools/cfgs/vod_models/spatialpillar_centerhead_geospa.yaml
--cfg_file tools/cfgs/vod_models/spatialpillar_full.yaml
```

---

## Eval

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --ckpt output/cfgs/vod_models/vod_radarpillar_rot/<run_name>/ckpt/checkpoint_best.pth
```

---

## Configs

| File | Purpose |
|---|---|
| `vod_radarpillar.yaml` | paper-faithful baseline (no rotation) |
| **`vod_radarpillar_rot.yaml`** | **rotation-augmented baseline — produced the headline 52.56** |
| `spatialpillar_geospa.yaml` | + GeoSPA |
| `spatialpillar_cqca.yaml` | + CQCA |
| `spatialpillar_dcn.yaml` | + DCN BEV backbone |
| `spatialpillar_kde.yaml` | + KDE density branch |
| `spatialpillar_centerhead.yaml` | + CenterHead detection head |
| `spatialpillar_centerhead_geospa.yaml` | + GeoSPA + CenterHead |
| `spatialpillar_centerhead_cqca.yaml` | + CQCA + CenterHead |
| `spatialpillar_full.yaml` | full SpatialPillar-IUC (all modules) |
| `spatialpillar_distill.yaml` | + LiDAR-to-radar distillation |

> The SpatialPillar ablation numbers previously reported in this README were measured on an older, weaker baseline (48–50 mAP). They are being re-run on top of the current 52.56 baseline; updated numbers will replace the old table once available.

---

## Citation

```bibtex
@inproceedings{gillen2024radarpillars,
  title     = {RadarPillars: Efficient Object Detection from 4D Radar Point Clouds},
  author    = {Gillen, Julius and Bieder, Manuel and Stiller, Christoph},
  booktitle = {Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)},
  year      = {2024}
}

@misc{openpcdet2020,
  title  = {OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
  author = {OpenPCDet Development Team},
  year   = {2020},
  url    = {https://github.com/open-mmlab/OpenPCDet}
}
```

---

## License

Released under the Apache 2.0 License — see [LICENSE](LICENSE). Built on top of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), which is itself Apache 2.0 licensed.
