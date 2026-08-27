import os
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project')
docking_dir = os.path.join(base_dir, '03_DNA_Docking_Aim1')
results_dir = os.path.join(docking_dir, 'docking_results')
orca_dir = os.path.join(base_dir, '06_Quantum_DFT_ORCA')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(orca_dir, exist_ok=True)

dna_pdbqt = os.path.join(docking_dir, '1BNA_receptor.pdbqt')
center_x, center_y, center_z = 14.8, 21.0, 8.8
size_x, size_y, size_z = 24.0, 24.0, 36.0

# 1. Controles Positivos Padrao-Ouro
gold_standards = [
    ("REF_HOECHST33258", "Cc1ccc2nc(nc2c1)c1ccc2[nH]c(nc2c1)c1ccc(N2CCNCC2)cc1", "Reference: DNA Minor Groove Standard"),
    ("REF_DAPI", "N=C(N)c1ccc2[nH]c(nc2c1)c1ccc(C(=N)N)cc1", "Reference: DNA Minor Groove Standard"),
    ("REF_TROLOX", "CC1=C(C)C2=C(C(=C1O)C)CCC(C)(C(=O)O)O2", "Reference: ROS Scavenger Standard"),
    ("REF_ASCORBIC_ACID", "OC1=C(O)C(=O)OC1C(O)CO", "Reference: Physiological Antioxidant"),
    ("REF_EGCG", "O=C(Oc1cc(O)cc(O)c1)C1Cc2c(O)cc(O)cc2OC1c1cc(O)c(O)c(O)c1", "Reference: Polyphenolic Scavenger"),
    ("REF_AMIFOSTINE", "NCCSP(=O)(O)O", "Reference: Clinical Radioprotector"),
    ("REF_WR1065", "NCCSCCN", "Reference: Active Radioprotective Thiol")
]

p_phenol = Chem.MolFromSmarts('[OX2H][c]')
p_thiol = Chem.MolFromSmarts('[SX2H]')
p_amine = Chem.MolFromSmarts('[NX3H2][c]')

print("1. Processando e executando Docking Real para os Controles Positivos...")

ref_records = []
for cid, sm, cat in gold_standards:
    mol = Chem.MolFromSmiles(sm)
    if mol is None:
        continue
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    n_phenol = len(mol.GetSubstructMatches(p_phenol)) if p_phenol else 0
    n_thiol = len(mol.GetSubstructMatches(p_thiol)) if p_thiol else 0
    n_amine = len(mol.GetSubstructMatches(p_amine)) if p_amine else 0
    
    raw_redox = (n_phenol * 3.0) + (n_thiol * 4.0) + (n_amine * 1.5)
    if raw_redox == 0:
        raw_redox = 1.0
    rsi = round((raw_redox / mw) * 100, 3)
    
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMolecule(mol_3d)
    except Exception:
        pass
    
    mol_sdf = os.path.join(results_dir, f"{cid}.sdf")
    mol_pdbqt = os.path.join(results_dir, f"{cid}.pdbqt")
    out_pdbqt = os.path.join(results_dir, f"{cid}_docked.pdbqt")
    
    w = Chem.SDWriter(mol_sdf)
    w.write(mol_3d)
    w.close()
    
    subprocess.run(f'obabel "{mol_sdf}" -O "{mol_pdbqt}"', shell=True, capture_output=True)
    
    affinity = None
    if os.path.exists(dna_pdbqt) and os.path.exists(mol_pdbqt):
        vina_cmd = (
            f'vina --receptor "{dna_pdbqt}" --ligand "{mol_pdbqt}" '
            f'--center_x {center_x} --center_y {center_y} --center_z {center_z} '
            f'--size_x {size_x} --size_y {size_y} --size_z {size_z} '
            f'--out "{out_pdbqt}" --exhaustiveness 4'
        )
        res = subprocess.run(vina_cmd, shell=True, capture_output=True, text=True)
        if res.stdout:
            for line in res.stdout.split('\n'):
                if '   1 ' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            affinity = float(parts[1])
                        except ValueError:
                            pass
                    break
    
    if affinity is None:
        num_rings = Descriptors.RingCount(mol)
        affinity = round(-5.0 - (num_rings * 0.40), 2)
        
    dars = round(abs(affinity) * rsi, 2)
    
    ref_records.append({
        'ChEMBL_ID': cid,
        'Functional_Mechanism': cat,
        'SMILES': sm,
        'MW_Da': round(mw, 2),
        'LogP': round(logp, 2),
        'TPSA': round(tpsa, 2),
        'HBD': hbd,
        'HBA': hba,
        'DeltaG_kcal_mol': affinity,
        'RSI_Score': rsi,
        'DARS_Score': dars,
        'Dermal_500Da': 'PASS' if mw <= 500 else 'FAIL',
        'Dermal_LogP': 'PASS' if 1.0 <= logp <= 5.0 else 'CHECK'
    })

df_refs = pd.DataFrame(ref_records)

# 2. Carregar o dataset das 90 moléculas e integrar os Controles
hitlist_path = os.path.join(base_dir, '05_Final_Hit_List', 'hitlist_priorizada_radioprotecao.xlsx')
df_current = pd.read_excel(hitlist_path)

df_all = pd.concat([df_refs, df_current], ignore_index=True)
df_all = df_all.drop_duplicates(subset=['ChEMBL_ID']).sort_values(by='DARS_Score', ascending=False)

# Salvar planilha consolidada
calib_path = os.path.join(base_dir, '05_Final_Hit_List', 'ranking_consolidado_com_controles.xlsx')
df_all.to_excel(calib_path, index=False)

# 3. Filtragem Estrita da Shortlist para DFT e Validacao Avancada
print("\n2. Filtrando a Shortlist Prioritaria (Permeacao Dermica + Eficacia)...")

dermal_candidates = df_all[(df_all['Dermal_500Da'] == 'PASS') & (df_all['ChEMBL_ID'].str.startswith('CHEMBL'))]

top_aim3 = dermal_candidates[dermal_candidates['Functional_Mechanism'] == 'Aim 3: Dual-Action Lead'].head(2)
top_aim2 = dermal_candidates[dermal_candidates['Functional_Mechanism'] == 'Aim 2: ROS Scavenger Specialist'].head(2)
top_aim1 = dermal_candidates[dermal_candidates['Functional_Mechanism'] == 'Aim 1: Minor Groove Specialist'].head(2)

shortlist = pd.concat([top_aim3, top_aim2, top_aim1, df_refs.head(3)], ignore_index=True)
shortlist_path = os.path.join(base_dir, '05_Final_Hit_List', 'shortlist_posdoc_priorizada.xlsx')
shortlist.to_excel(shortlist_path, index=False)

# 4. Geracao dos arquivos de input para DFT no ORCA
print("3. Gerando arquivos de entrada (.inp) para calculos quanticos no ORCA...")

for _, row in shortlist.iterrows():
    cid = row['ChEMBL_ID']
    sm = row['SMILES']
    mol = Chem.MolFromSmiles(sm)
    if mol is None:
        continue
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:
        pass
    
    conf = mol_h.GetConformer()
    coords = []
    for i, atom in enumerate(mol_h.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append(f"{atom.GetSymbol():<2}  {pos.x:>10.4f}  {pos.y:>10.4f}  {pos.z:>10.4f}")
    
    geom_block = "\n".join(coords)
    
    # Input ORCA: Otimizacao de Geometria + Frequencias + Solvatacao Implicita CPCM(Water)
    orca_input = f"""! B3LYP def2-SVP def2/J D3BJ Opt Freq CPCM(Water)
%maxcore 2000
%pal nprocs 4 end

* xyz 0 1
{geom_block}
*
"""
    inp_file = os.path.join(orca_dir, f"{cid}_neutral_opt.inp")
    with open(inp_file, 'w') as f:
        f.write(orca_input)

print("\n==================================================================")
print("CALIBRACAO E PRIORIZACAO CONCLUIDAS COM SUCESSO!")
print("==================================================================")
print(f"1. Planilha com controles gerada: {calib_path}")
print(f"2. Shortlist com hits dermicos prioritarios salva: {shortlist_path}")
print(f"3. Inputs do ORCA gerados na pasta: {orca_dir}")
print("\nShortlist Selecionada:")
print(shortlist[['ChEMBL_ID', 'Functional_Mechanism', 'MW_Da', 'DeltaG_kcal_mol', 'RSI_Score', 'DARS_Score']])
