content = """# High-Throughput Small-Molecule Radioprotection Screening Pipeline

An automated chemoinformatics and structure-based virtual screening framework designed to identify, evaluate, and prioritize drug-like small molecules for **topical radioprotection**.

This workflow translates the biomimetic mechanisms of radiotolerant organisms (such as tardigrade **Dsup**) and established chemical radioprotectors into a high-throughput, multi-objective computational pipeline targeting DNA stabilization and free radical neutralization.

## 1. Multi-Objective Screening Architecture

- **Aim 1 (Delta G_bind):** Predicted binding free energy (kcal/mol) in the minor groove of B-DNA (PDB ID: 1BNA).
- **Aim 2 (RSI):** Radical Scavenging Index, quantifying thermodynamic density of redox centers normalized by molecular weight:
  `RSI = (Sum(w_i * N_i) / MW) * 100`
- **Aim 3 (DARS):** Dual-Action Radioprotection Score:
  `DARS = |Delta G_bind| * RSI`

## 2. Benchmark Results (Full Curated Set)

| Compound ID | Target Mechanism | MW (Da) | Delta G_bind (kcal/mol) | RSI Score | DARS Score | Dermal Filter (500 Da) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `CHEMBL511458` | Aim 2: Polyphenol Scavenger | 318.24 | -6.20 | 5.656 | 35.07 | PASS |
| `CHEMBL469752` | Aim 2: Polyphenol Scavenger | 316.27 | -6.20 | 3.794 | 23.52 | PASS |
| `CHEMBL459583` | Aim 2: Polyphenol Scavenger | 332.31 | -6.20 | 3.611 | 22.39 | PASS |
| `CHEMBL1683055` | Aim 1: Minor Groove Binder | 311.35 | -8.71 | 0.482 | 4.20 | PASS |
| `CHEMBL214612` | Aim 2: Thiol/Disulfide | 185.32 | -6.20 | 0.540 | 3.35 | PASS |
| `CHEMBL176543` | Aim 1: Intercalator/Shield | 293.41 | -8.10 | 0.341 | 2.76 | PASS |
| `CHEMBL1962789` | Aim 1: Minor Groove Binder | 354.37 | -8.50 | 0.282 | 2.40 | PASS |
| `CHEMBL1927181` | Aim 3: Dual-Action Lead | 429.48 | -8.68 | 0.233 | 2.02 | PASS |

## 3. Repository Structure

```text
├── 01_Mining_and_Curation/
├── 02_Chemoinformatics_2D/
│   └── curated_radioprotection_library.csv
├── 03_DNA_Docking_Aim1/
│   ├── 1BNA_clean_DNA.pdb
│   ├── 1BNA_receptor.pdbqt
│   ├── ranking_dna_binding_aim1.xlsx
│   └── docking_results/
├── 04_ROS_Scavenging_Aim2/
│   └── ranking_ros_aim2_final.xlsx
├── 05_Final_Hit_List/
│   ├── hitlist_priorizada_radioprotecao.xlsx
│   └── ranking_dual_action_aim3_final.xlsx
├── run_pipeline.py
├── .gitignore
└── README.md
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(content)

print("README.md gerado com sucesso!")
