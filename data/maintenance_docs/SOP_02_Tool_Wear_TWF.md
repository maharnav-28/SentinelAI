> **[NOTICE: ILLUSTRATIVE REFERENCE DOCUMENT]**  
> *This document is an illustrative technical procedure formulated for demonstration and educational benchmarking with the synthetic UCI AI4I 2020 Predictive Maintenance dataset (Matzka, 2020). It incorporates realistic tool wear progression limits, but is NOT an official cutting tool OEM manual. Do not use for physical repairs on live factory hardware.*

# Illustrative SOP 02: Cutting Tool Wear & Replacement Criteria
**Document ID:** SOP-MNT-002 (Illustrative)  
**Target Mechanism:** Tool Wear Failure (TWF) & Cutting Insert Degradation  
**Applicability:** Milling Cutters, Face Mills, and End Mills (Insert Grades L, M, H)

---

## 1. Tool Wear Progression & Limits
Continuous cutting produces progressive flank wear ($V_B$) and crater wear on carbide inserts due to friction, high localized shear stress, and thermal cycling.

- **Wear Limit Criteria:**
  - For standard Low-variant (L) tools: Maximum continuous service limit is **200 minutes** of accumulated cut time.
  - For Medium-variant (M) tools: Limit is **220 minutes**.
  - For High-variant (H) premium coated carbide tools: Limit is **240 minutes**.
- **Critical Threshold:** Any tool exceeding **200 minutes** without microscopic inspection is operating in the accelerated catastrophic wear phase.
- **Physical Symptom:** Exceeding tool wear limits causes burr formation, severe surface roughness ($R_a > 3.2\ \mu\text{m}$), increased spindle torque, and eventual insert tip fracture.

## 2. Mandatory Inspection Steps (At 180 Minutes)
1. Stop the cycle at a safe retract point.
2. Engage Lockout/Tagout (LOTO) on the spindle drive.
3. Use a 10x illuminated optical loupe or shop microscope to inspect cutting edges:
   - Measure flank wear land width ($V_B$). If $V_B \ge 0.30\text{ mm}$, index the insert.
   - Inspect for edge chipping, micro-flaking, or built-up edge (BUE).
   - Check tool holder clamping pocket for fretting corrosion or chip jamming.

## 3. Tool Replacement & Indexing Protocol
1. Completely halt spindle and wait for stationary zero-speed confirmation.
2. Clean chips from pocket with a soft brass brush.
3. Loosen Torx clamping screw using a calibrated torque wrench (typically 2.5 to 3.0 Nm).
4. Index to a fresh cutting edge or replace insert with genuine specification inserts.
5. Apply anti-seize paste lightly to the clamping screw thread.
6. Reset the CNC control software tool wear timer (`Tool wear [min] = 0`). Never resume production without resetting the register.
