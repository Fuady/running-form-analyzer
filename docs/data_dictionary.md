# Data Dictionary

## biomech_features.csv
Per-frame biomechanical features. One row per video frame.

| Column | Type | Description |
|---|---|---|
| `video_stem` | str | Source video filename |
| `form_class` | str | good_form / overstriding / forward_lean / arm_crossing |
| `frame` | int | Frame index (0-based) |
| `timestamp_ms` | float | Timestamp in ms |
| `trunk_lean_angle` | float | Forward trunk tilt from vertical (°). Ideal: 3–12° |
| `max_overstride` | float | Foot x position ahead of hip (torso units). Ideal: ≤ 0.05 |
| `arm_swing_symmetry` | float | L/R elbow angle difference (°). Ideal: < 15° |
| `hip_drop_angle` | float | Lateral pelvic tilt during stance (°). Ideal: < 5° |
| `knee_drive_angle` | float | Front knee lift angle from hip vertical (°). Ideal: > 60° |
| `stride_angle` | float | Rear leg extension angle at push-off (°) |
| `rear_knee_angle` | float | Rear knee bend (°) |
| `front_knee_angle` | float | Front knee bend (°) |
| `head_alignment` | float | Head-to-trunk axis deviation (°). Ideal: < 10° |
| `vertical_oscillation` | float | Rolling std of hip height (torso units). Ideal: < 0.08 |
| `left_elbow_angle` | float | Left elbow bend (°). Ideal: 85–95° |
| `right_elbow_angle` | float | Right elbow bend (°). Ideal: 85–95° |
| `left_arm_cross` | float | Left wrist x relative to midline. Ideal: ≈ 0 |
| `right_arm_cross` | float | Right wrist x relative to midline. Ideal: ≈ 0 |
| `cadence_proxy` | float | Ankle vertical velocity magnitude |
| `*_vel` | float | Angular velocity of each angle (degrees/sec) |
| `hip_height` | float | Mid-hip y position (normalized) |

## stride_metrics.csv
Per-clip stride-level summary. One row per video.

| Column | Type | Description |
|---|---|---|
| `cadence_spm` | float | Estimated steps per minute |
| `vertical_oscillation` | float | Mean hip vertical movement |
| `trunk_lean_mean` | float | Mean trunk lean across clip |
| `trunk_lean_std` | float | Variability of trunk lean |
| `arm_swing_symmetry_mean` | float | Mean arm asymmetry |
| `max_overstride_mean` | float | Mean foot overstride |
| `hip_drop_mean` | float | Mean hip drop |

## form_labels.csv
Per-clip class labels.

| Column | Description |
|---|---|
| `video_stem` | Video filename (without extension) |
| `csv_path` | Path to normalized keypoint CSV |
| `form_class` | good_form / overstriding / forward_lean / arm_crossing |
| `label_source` | auto_folder / manual / synthetic |

## Form Class Reference

| Class | Key Biomechanical Signature | Primary Correction |
|---|---|---|
| `good_form` | Upright trunk 3–12°, foot near hip, symmetric arms | Maintain |
| `overstriding` | Foot > 0.05 torso units ahead of hip | Shorten stride, increase cadence |
| `forward_lean` | Trunk lean > 12° | Stand tall, engage core |
| `arm_crossing` | Wrist crosses body midline (> 0.05 units) | Swing fore-aft, not across |
