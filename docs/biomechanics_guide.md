# Running Biomechanics Guide

A reference for understanding the features extracted by this system.

---

## 1. The Running Gait Cycle

One complete stride = two steps (left + right). Phases:
```
Initial Contact → Loading Response → Mid-Stance → Terminal Stance
→ Pre-Swing → Initial Swing → Mid-Swing → Terminal Swing → [repeat]
```

## 2. Key Features & Thresholds

### Trunk Lean (3–12° forward)
- **Too upright (< 3°)**: Runner may be "sitting" in hips, inefficient
- **Excessive lean (> 12°)**: Shifts load to lower back, increases braking
- **Physics**: Slight forward lean uses gravity to assist propulsion

### Overstriding (foot ≤ 0.05 torso units ahead of hip)
- **Overstriding**: Heel lands ahead of center of mass → braking impulse
- **Impact**: Each braking step increases injury risk 2–3×
- **Fix**: Increase cadence by 5–10% (target 170–180 steps/min)

### Arm Swing Symmetry (< 15° asymmetry)
- Arms counterbalance leg rotation; asymmetry wastes energy
- Crossing the midline causes trunk rotation → energy waste
- **Cue**: "Thumbs up, elbows back"

### Hip Drop (< 5°)
- Contralateral hip drops during single-leg stance
- Indicates weak hip abductors (gluteus medius)
- **Injury risk**: Iliotibial band syndrome, patellofemoral pain

### Knee Drive (> 60° lift angle)
- Higher knee lift = longer stride without overstriding
- Creates stored elastic energy in hip flexors
- Insufficient drive = shuffling gait, slow cadence

### Vertical Oscillation (< 0.08 torso units per stride)
- Excessive bouncing wastes energy going UP vs. FORWARD
- Well-trained runners: ~6–8cm at marathon pace
- **Cue**: "Run along the ground, not above it"

## 3. Form Score Interpretation

| Score | Meaning |
|---|---|
| 90–100 | Excellent — minimal deviations |
| 75–89  | Good — minor tweaks recommended |
| 60–74  | Moderate — 1–2 significant faults |
| 40–59  | Poor — multiple faults affecting efficiency |
| < 40   | Major issues — fundamental technique work needed |

## 4. Form Class Signatures

| Class | Trunk Lean | Overstride | Arm Cross | Hip Drop |
|---|---|---|---|---|
| good_form | 3–12° | ≤ 0.05 | ≤ 0.05 | < 5° |
| overstriding | Any | **> 0.10** | Any | Any |
| forward_lean | **> 12°** | Any | Any | Any |
| arm_crossing | Any | Any | **> 0.08** | Any |

## 5. References

- Heiderscheit et al. (2011). Effects of step rate manipulation on joint mechanics. *MSSE*.
- Dorn et al. (2012). Muscular strategy shift in running. *J Experimental Biology*.
- Napier et al. (2018). Kinematic predictors of running injury. *BJSM*.
