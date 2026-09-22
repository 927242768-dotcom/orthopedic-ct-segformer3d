# v7 Final Model Freeze

## Freeze decision
- Final engineering model: **v7**
- Status: **FROZEN**
- Training line: **CLOSED**
- v7.1: archived negative residual-FP refinement; it does not replace v7.
- No new v8/v9-style continuation is permitted on the same validation197 loop.

## Canonical artifacts
- Config: `configs/orthopedic_ct_full_large_scale_v7_segformer3d_hr_hard_negative_precision.yaml`
- Experiment: `experiments/20260917_185941_ctspine1k_v7_segformer3d_hr_hard_negative_precision`
- Checkpoint: `.../checkpoint/best.pt`
- Validation197: `.../full_volume_validation197`
- Main inference protocol: ROI 96, overlap 0.5, BatchNorm running mode, component filter 8192 voxels, connectivity 3.

## v7 validation197
| Metric | Mean |
|---|---:|
| Dice | 0.7933430437 |
| IoU | 0.7346736465 |
| Precision | 0.7862013885 |
| Recall | 0.8244633247 |
| HD95 | 37.946454 mm |
| ASSD | 8.810012 mm |
| Prediction / target FG ratio | 1.6849015 |
| Component count error | 1.1522843 |
| False merge | 0.3147208 |
| False break | 0.0659898 |

The fixed 32768-voxel component filter is validation-only secondary evidence: Dice 0.7980874500, Precision 0.7969522683, Recall 0.8215012833, FG ratio 1.6512499118. It is not a reason to revisit test or start a new threshold sweep.

## Why v7.1 was rejected
v7.1 completed training and 197/197 full-volume validation. Its main protocol produced Dice 0.7026965254, Precision 0.6121711390, Recall 0.8796528098, HD95 132.534697 mm, ASSD 24.185827 mm, and FG ratio 2.8933022548.

Although recall increased and Dice<0.5 cases decreased from 32 to 29, mean Dice, precision, foreground control, HD95, ASSD, Dice<0.7 count, Precision<0.5 count, and FG-ratio>2 count worsened. The fixed 32768 comparison reached Dice 0.7836175414 and still did not exceed v7.

## Boundary after freeze
Do not retrain v7/v7.1, rerun validation197, access test for model selection, reopen residual-FP tuning, or continue an incremental v7.2/v8/v9 chain. Future algorithm work must be a separately defined study such as nnU-Net / Residual Encoder nnU-Net baseline, domain-generalization analysis, cross-source analysis, or external/clinical validation.

Historical v13 / 10-case pilot records remain in the repository for traceability but are not the current engineering-model definition.