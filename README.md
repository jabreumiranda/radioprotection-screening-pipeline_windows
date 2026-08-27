# High-Throughput Small-Molecule Radioprotection Screening Pipeline

An automated chemoinformatics and structure-based virtual screening framework designed to identify, evaluate, and prioritize drug-like small molecules for **topical radioprotection**.

## 1. Multi-Objective Screening Architecture
- **Aim 1 (Delta G_bind):** Real molecular docking free energy (kcal/mol) in the B-DNA minor groove (PDB ID: 1BNA) via AutoDock Vina.
- **Aim 2 (RSI):** Radical Scavenging Index, quantifying thermodynamic density of redox centers normalized by MW.
- **Aim 3 (DARS):** Dual-Action Radioprotection Score: DARS = |Delta G_bind| * RSI.

## 2. Quantitative Screening Summary (90 Compounds Evaluated)
- **Aim 3 (Dual-Action Leads):** 4 candidates
- **Aim 1 (DNA Minor Groove Specialists):** 18 candidates
- **Aim 2 (ROS Scavengers):** 16 candidates
- **Basal / Moderate Profile:** 52 candidates

### Top 10 Prioritized Radioprotective Hits
| ChEMBL ID | Functional Class | MW (Da) | Delta G_bind (kcal/mol) | RSI | DARS | Dermal 500Da |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `CHEMBL511458` | Aim 2: ROS Scavenger Specialist | 318.24 | -6.7 | 5.656 | **37.9** | PASS |
| `CHEMBL247484` | Aim 2: ROS Scavenger Specialist | 302.24 | -6.7 | 4.963 | **33.25** | PASS |
| `CHEMBL151` | Aim 2: ROS Scavenger Specialist | 286.24 | -6.7 | 4.192 | **28.09** | PASS |
| `CHEMBL3896909` | Aim 2: ROS Scavenger Specialist | 302.31 | -6.7 | 3.969 | **26.59** | PASS |
| `CHEMBL4847220` | Aim 3: Dual-Action Lead | 538.46 | -7.9 | 3.343 | **26.41** | FAIL |
| `CHEMBL1779470` | Aim 3: Dual-Action Lead | 538.46 | -7.9 | 3.343 | **26.41** | FAIL |
| `CHEMBL469752` | Aim 2: ROS Scavenger Specialist | 316.27 | -6.7 | 3.794 | **25.42** | PASS |
| `CHEMBL459583` | Aim 2: ROS Scavenger Specialist | 332.31 | -6.3 | 3.611 | **22.75** | PASS |
| `CHEMBL243677` | Aim 2: ROS Scavenger Specialist | 270.24 | -6.7 | 3.33 | **22.31** | PASS |
| `CHEMBL457821` | Aim 2: ROS Scavenger Specialist | 270.24 | -6.7 | 3.33 | **22.31** | PASS |