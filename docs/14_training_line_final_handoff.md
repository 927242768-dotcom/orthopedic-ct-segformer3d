# Training Line Final Handoff

## Final state
- Final model: **v7**
- Training state: **TRAINING LINE CLOSED**
- v7.1: failed final residual-FP hard-case refinement; retained as negative evidence.
- No test access or new full-volume validation was performed during this finalization.

## v7
- Config: `configs/orthopedic_ct_full_large_scale_v7_segformer3d_hr_hard_negative_precision.yaml`
- Experiment: `experiments/20260917_185941_ctspine1k_v7_segformer3d_hr_hard_negative_precision`
- Checkpoint: `checkpoint/best.pt`
- Validation: `full_volume_validation197`, 197/197 cases complete.
- Main protocol: ROI=96, overlap=0.5, BatchNorm running, min_component_voxels=8192, connectivity=3.
- Metrics: Dice=0.7933430437, IoU=0.7346736465, Precision=0.7862013885, Recall=0.8244633247, HD95=37.946454 mm, ASSD=8.810012 mm, FG ratio=1.6849015.

## v7.1
- Experiment: `experiments/20260921_130417_ctspine1k_v7_1_segformer3d_hr_residual_fp_refinement`
- Guidance: 610/610; audit missing=0, extra=0, validation-overlap=0, shape_bad=0, affine_bad=0, bad_count=0.
- Training: early-stopped at epoch 5; best checkpoint epoch 2; patch selector Dice=0.6680869599572375.
- Full-volume validation: 197/197.
- Main metrics: Dice=0.7026965254, Precision=0.6121711390, Recall=0.8796528098, HD95=132.534697 mm, ASSD=24.185827 mm, FG ratio=2.8933022548.
## Decision rationale
v7.1 improved recall and slightly reduced the count of Dice<0.5 cases, but degraded the primary quality profile: mean Dice and precision fell, foreground overprediction increased, surface distances worsened sharply, and several hard-failure counts worsened. The fixed 32768 validation-only comparison also stayed below v7. Therefore v7 remains the sole final engineering model.

## Prohibited continuation
Do not reopen v7/v7.1 training, start v8/v9 as incremental tuning, rerun validation197, access test for model selection, sweep thresholds again, or use test outcomes to alter loss, sampling, augmentation, post-processing, or architecture.

## Next phase
1. Competition Demo and teaching-product refinement.
2. 3D reconstruction and presentation quality.
3. Results-review, uncertainty, and failure-case visualization.
4. Defense materials and paper consolidation.
5. Independent nnU-Net / Residual Encoder nnU-Net baselines.
6. External or legally authorized clinical validation.
7. Publication formatting and delivery packaging.

## Research limitations
The frozen result is validation evidence, not a clinical-grade claim. Cross-architecture baselines, external/multicenter validation, legally authorized clinical data, robust subgroup evidence, and statistical significance analysis remain open research requirements.

## Git handoff
The final commit hash and remote-sync result are filled by the final Git verification step. Repository identity must remain `927242768-dotcom <927242768@qq.com>`, and model weights, NIfTI volumes, prediction volumes, caches, virtual environments, and the experiments directory must remain outside Git.