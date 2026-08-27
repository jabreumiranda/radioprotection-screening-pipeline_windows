import os
import pandas as pd

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project')
hitlist_path = os.path.join(base_dir, '05_Final_Hit_List', 'hitlist_priorizada_radioprotecao.xlsx')
hitlist = pd.read_excel(hitlist_path)

counts = hitlist['Functional_Mechanism'].value_counts()
top10 = hitlist.head(10)

lines = [
    "# High-Throughput Small-Molecule Radioprotection Screening Pipeline",
    "",
    "An automated chemoinformatics and structure-based virtual screening framework designed to identify, evaluate, and prioritize drug-like small molecules for **topical radioprotection**.",
    "",
    "## 1. Multi-Objective Screening Architecture",
    "- **Aim 1 (Delta G_bind):** Real molecular docking free energy (kcal/mol) in the B-DNA minor groove (PDB ID: 1BNA) via AutoDock Vina.",
    "- **Aim 2 (RSI):** Radical Scavenging Index, quantifying thermodynamic density of redox centers normalized by MW.",
    "- **Aim 3 (DARS):** Dual-Action Radioprotection Score: DARS = |Delta G_bind| * RSI.",
    "",
    "## 2. Quantitative Screening Summary (90 Compounds Evaluated)",
    f"- **Aim 3 (Dual-Action Leads):** {counts.get('Aim 3: Dual-Action Lead', 0)} candidates",
    f"- **Aim 1 (DNA Minor Groove Specialists):** {counts.get('Aim 1: Minor Groove Specialist', 0)} candidates",
    f"- **Aim 2 (ROS Scavengers):** {counts.get('Aim 2: ROS Scavenger Specialist', 0)} candidates",
    f"- **Basal / Moderate Profile:** {counts.get('Moderate / Basal Profile', 0)} candidates",
    "",
    "### Top 10 Prioritized Radioprotective Hits",
    "| ChEMBL ID | Functional Class | MW (Da) | Delta G_bind (kcal/mol) | RSI | DARS | Dermal 500Da |",
    "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
]

for _, r in top10.iterrows():
    cid = r['ChEMBL_ID']
    mech = r['Functional_Mechanism']
    mw = r['MW_Da']
    dg = r['DeltaG_kcal_mol']
    rsi = r['RSI_Score']
    dars = r['DARS_Score']
    derm = r['Dermal_500Da']
    lines.append(f"| `{cid}` | {mech} | {mw} | {dg} | {rsi} | **{dars}** | {derm} |")

readme_path = os.path.join(base_dir, 'README.md')
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("README.md atualizado com sucesso!")
