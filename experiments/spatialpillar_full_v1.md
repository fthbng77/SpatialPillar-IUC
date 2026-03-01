# SpatialPillar-IUC Full v1 - Experiment Results

## Config
- **File:** `tools/cfgs/vod_models/spatialpillar_full.yaml`
- **Modules:** CenterHead + DCN BEV Backbone + GeoSPA + KDE Density + CQCA
- **Epochs:** 60
- **Batch Size:** 16
- **Optimizer:** adam_onecycle, LR=0.003, weight_decay=0.01
- **Dataset:** VoD radar_5frames (1296 samples)

## Results (3D AP R40)

### Per-Epoch (last 11 epochs)
| Epoch | Car    | Ped    | Cyc    | mAP   |
|-------|--------|--------|--------|-------|
| 50    | 35.19  | 39.94  | 69.72  | 48.29 |
| 51    | 31.16  | 36.69  | 64.47  | 44.11 |
| 52    | 31.90  | 38.68  | 65.15  | 45.24 |
| 53    | 34.42  | 38.61  | 70.40  | 47.81 |
| 54    | 34.49  | 39.54  | 68.00  | 47.34 |
| 55    | 29.25  | 38.14  | 62.75  | 43.38 |
| 56    | 34.35  | 39.58  | 66.07  | 46.67 |
| 57    | 34.97  | 39.81  | 67.17  | 47.32 |
| 58    | 34.47  | 39.65  | 67.25  | 47.12 |
| 59    | 34.43  | 39.53  | 66.96  | 46.97 |
| 60    | 34.49  | 39.56  | 67.03  | 47.03 |

### Best Epoch: 50 (mAP: 48.29)

## Comparison vs RadarPillar Baseline (Epoch 60)

| Class      | RadarPillar | SpatialPillar-IUC | Diff   |
|------------|-------------|-------------------|--------|
| Car        | 36.13       | 34.49             | -1.64  |
| Pedestrian | 40.95       | 39.56             | -1.39  |
| Cyclist    | 68.62       | 67.03             | -1.59  |
| **mAP**    | **48.57**   | **47.03**         | **-1.54** |

## Notes
- Performance dropped ~1.5 mAP compared to baseline
- All classes show consistent decrease
- High variance between epochs (43.38 - 48.29) suggests training instability
- Possible causes: CenterHead harder to converge on sparse radar, too many modules at once, CQCA detach() cutting gradients
- Next steps: ablation studies (test modules individually)
