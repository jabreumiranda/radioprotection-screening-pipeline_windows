# High-Throughput Small-Molecule Radioprotection Screening Pipeline

An automated chemoinformatics and structure-based virtual screening framework designed to identify, evaluate, and prioritize drug-like small molecules for **topical radioprotection**.

## 1. Multi-Objective Screening Architecture

- **Aim 1 (Delta G_bind):** Predicted binding free energy (kcal/mol) in the minor groove of B-DNA (PDB ID: 1BNA).
- **Aim 2 (RSI):** Radical Scavenging Index (thermodynamic redox density normalized by MW).
- **Aim 3 (DARS):** Dual-Action Radioprotection Score (|Delta G_bind| * RSI).

## 2. Benchmark Results (Full Curated Set)

| Compound ID | Target Mechanism | MW (Da) | Delta G_bind (kcal/mol) | RSI Score | DARS Score | Dermal Filter (500 Da) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| CHEMBL511458 | Aim 2: Polyphenol Scavenger | 318.24 | -6.20 | 5.656 | 35.07 | PASS |
| CHEMBL469752 | Aim 2: Polyphenol Scavenger | 316.27 | -6.20 | 3.794 | 23.52 | PASS |
| CHEMBL459583 | Aim 2: Polyphenol Scavenger | 332.31 | -6.20 | 3.611 | 22.39 | PASS |
| CHEMBL1683055 | Aim 1: Minor Groove Binder | 311.35 | -8.71 | 0.482 | 4.20 | PASS |
| CHEMBL214612 | Aim 2: Thiol/Disulfide | 185.32 | -6.20 | 0.540 | 3.35 | PASS |
| CHEMBL176543 | Aim 1: Intercalator/Shield | 293.41 | -8.10 | 0.341 | 2.76 | PASS |
| CHEMBL1962789 | Aim 1: Minor Groove Binder | 354.37 | -8.50 | 0.282 | 2.40 | PASS |
| CHEMBL1927181 | Aim 3: Dual-Action Lead | 429.48 | -8.68 | 0.233 | 2.02 | PASS |
