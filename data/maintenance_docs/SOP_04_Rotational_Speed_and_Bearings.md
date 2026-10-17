> **[NOTICE: ILLUSTRATIVE REFERENCE DOCUMENT]**  
> *This document is an illustrative technical procedure formulated for demonstration and educational benchmarking with the synthetic UCI AI4I 2020 Predictive Maintenance dataset (Matzka, 2020). It incorporates realistic lubrication intervals and vibration boundaries, but is NOT an official spindle bearing OEM manual. Do not use for physical repairs on live factory hardware.*

# Illustrative SOP 04: Spindle Bearings, Lubrication & Speed Limits
**Document ID:** SOP-MNT-004 (Illustrative)  
**Target Mechanism:** Bearing Degradation, Speed Fluctuations & Random Failure (RNF) Mitigation  
**Applicability:** Main Spindle Assemblies, Angular Contact Ceramic/Steel Bearings

---

## 1. Operating Envelope & Speed Guidelines
High-precision machining spindles are rated for continuous service between **1,200 rpm and 2,800 rpm**.

- **Under-speed Hazard:**
  - Operating under high cutting load (< 1,200 rpm with Torque > 50 Nm) subjects electric motor stator windings to high I²R thermal dissipation while integral fan cooling capacity is minimal.
- **Over-speed Hazard:**
  - Exceeding 2,800 rpm without balanced tooling (ISO 1940 Grade G 2.5) induces excessive centrifugal force and dynamic bearing radial runout.

## 2. Spindle Bearing Lubrication Protocol
1. **Lubricant Specification:**
   - Use high-speed synthetic ester spindle grease (viscosity grade ISO VG 15 to VG 32).
   - Never mix lithium-soap grease with polyurea-based high-speed grease.
2. **Re-lubrication Interval:**
   - Every **1,000 operating hours** or semi-annually.
   - Exact volume control: Inject strictly 2.0 cm³ per bearing chamber.
   - *Caution:* Over-greasing causes viscous churning, leading to rapid temperature escalation > 65°C and thermal expansion seizure.
3. **Run-in Cycle after Re-greasing:**
   - Run spindle at 500 rpm for 20 minutes.
   - Increase by 500 rpm steps every 15 minutes up to maximum rated speed, monitoring bearing cap temperature.

## 3. Vibration & Runout Diagnostics
- **Vibration Limit:**
  - Spindle housing peak-to-peak vibration velocity must remain below **1.8 mm/s RMS** in the 10 Hz to 1,000 Hz frequency range.
- **Runout Limit:**
  - Total radial runout measured inside the spindle taper with a 0.001 mm dial test indicator (DTI) must not exceed **0.005 mm (5 µm)**.
- **Unexplained / Random Fluctuations (RNF):**
  - If random torque or speed spikes occur without tool wear or thermal anomalies, verify ground cable bonding, electrical noise suppression on VFD cables, and encoder shield termination.
