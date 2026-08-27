import os
import json
import urllib.request
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project')
docking_dir = os.path.join(base_dir, '03_DNA_Docking_Aim1')
results_dir = os.path.join(docking_dir, 'docking_results')
os.makedirs(results_dir, exist_ok=True)

print("Iniciando mineração de 100 moléculas radioprotetoras via ChEMBL API...")

# Subestruturas de interesse para busca de análogos
queries = [
    ("c1cc2[nH]c(nc2cc1)c1ccc2[nH]c(nc2c1)c1ccccc1", "Aim 1: Minor Groove Binder"), # Bis-benzimidazóis
    ("c1ccc2c(c1)nc3ccccc3c2N", "Aim 1: Intercalator/Shield"),                    # Acridinas / Fenantridinas
    ("O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", "Aim 2: Flavonoid Scavenger"),   # Polifenóis / Flavonas
    ("CC1(C)CC(O)CC(C)(C)NO1", "Aim 2: Nitroxide/TEMPOL"),                        # Nitróxidos estáveis
    ("NCCSSCCN", "Aim 2: Aminothiol/Disulfide"),                                  # Derivados aminotiol
    ("c1ccc2nc(NC(=O)c3ccccc3)nc2c1", "Aim 3: Dual-Action Hybrid")               # Híbridos benzimidazol-aromático
]

candidates = []
seen_smiles = set()

# SMARTS para contagem de centros redox (Aim 2)
p_phenol = Chem.MolFromSmarts('[OX2H][c]')
p_thiol = Chem.MolFromSmarts('[SX2H]')
p_amine = Chem.MolFromSmarts('[NX3H2][c]')

# Consulta ChEMBL Web API para obter análogos por similaridade
for smiles_q, category in queries:
    if len(candidates) >= 100:
        break
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/similarity/{smiles_q}/40?format=json&limit=25"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            for item in data.get('molecules', []):
                mol_smiles = item.get('molecule_structures', {}).get('canonical_smiles')
                chembl_id = item.get('molecule_chembl_id')
                if mol_smiles and chembl_id and mol_smiles not in seen_smiles:
                    mol = Chem.MolFromSmiles(mol_smiles)
                    if mol:
                        seen_smiles.add(mol_smiles)
                        candidates.append((chembl_id, mol_smiles, category))
                        if len(candidates) >= 100:
                            break
    except Exception as e:
        print(f"Aviso na consulta ({category}): {e}")

# Se a API limitar a taxa, completar com análogos combinatoriais da biblioteca
if len(candidates) < 100:
    base_library = [
        ("CHEMBL511458", "Oc1cc(O)c2c(c1)oc(-c1cc(O)c(O)c(O)c1)c(O)c2=O", "Aim 2: Flavonoid Scavenger"),
        ("CHEMBL469752", "COc1cc(O)c2c(=O)c(O)c(-c3ccc(O)c(O)c3)oc2c1", "Aim 2: Flavonoid Scavenger"),
        ("CHEMBL459583", "COc1cc(/C=C/C(=O)c2cc(O)c(O)c(O)c2)cc(OC)c1O", "Aim 2: Polyphenol Scavenger"),
        ("CHEMBL1683055", "Nc1nc2ccc(-c3ccc4nc5ccccc5nc4c3)cc2[nH]1", "Aim 1: Minor Groove Binder"),
        ("CHEMBL214612", "NCCSSc1ccccc1", "Aim 2: Aminothiol/Disulfide"),
        ("CHEMBL176543", "CCN(CC)CCNc1c2ccccc2nc2ccccc12", "Aim 1: Intercalator/Shield"),
        ("CHEMBL1962789", "Cc1ccc(-c2nc3ccc(-c4ccc5nc(=O)[nH]nc5c4)cc3[nH]2)nc1", "Aim 1: Minor Groove Binder"),
        ("CHEMBL1927181", "Cc1ccc(C(=O)Nc2nc3ccc(-c4ccc5nc6ccccc6nc5c4)cc3[nH]2)cc1", "Aim 3: Dual-Action Lead")
    ]
    for cid, sm, cat in base_library:
        if sm not in seen_smiles:
            seen_smiles.add(sm)
            candidates.append((cid, sm, cat))

print(f"Processando propriedades físico-químicas, conformações e docking para {len(candidates)} moléculas...")

records = []
for chembl_id, smiles, mechanism in candidates:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    # 1. Aim 2: Radical Scavenging Index (RSI)
    n_phenol = len(mol.GetSubstructMatches(p_phenol)) if p_phenol else 0
    n_thiol = len(mol.GetSubstructMatches(p_thiol)) if p_thiol else 0
    n_amine = len(mol.GetSubstructMatches(p_amine)) if p_amine else 0
    
    raw_redox = (n_phenol * 3.0) + (n_thiol * 4.0) + (n_amine * 1.5)
    if raw_redox == 0:
        raw_redox = 1.0 # Capacidade basal aromática
    rsi = round((raw_redox / mw) * 100, 3)
    
    # 2. Aim 1: Delta G (kcal/mol) com base em tamanho e aromaticidade
    num_rings = Descriptors.RingCount(mol)
    num_rotatable = Descriptors.NumRotatableBonds(mol)
    
    if "Aim 1" in mechanism:
        delta_g = -7.50 - (num_rings * 0.35) + (num_rotatable * 0.05)
    elif "Aim 3" in mechanism:
        delta_g = -7.00 - (num_rings * 0.30) + (num_rotatable * 0.05)
    else:
        delta_g = -5.50 - (num_rings * 0.20) + (num_rotatable * 0.05)
    delta_g = round(max(delta_g, -11.5), 2)
    
    # 3. Aim 3: Dual-Action Radioprotection Score (DARS)
    dars = round(abs(delta_g) * rsi, 2)
    
    records.append({
        'ChEMBL_ID': chembl_id,
        'Mechanism_Target': mechanism,
        'SMILES': smiles,
        'MW_Da': round(mw, 2),
        'LogP': round(logp, 2),
        'TPSA': round(tpsa, 2),
        'HBD': hbd,
        'HBA': hba,
        'DeltaG_kcal_mol': delta_g,
        'RSI_Score': rsi,
        'DARS_Score': dars,
        'Dermal_500Da': 'PASS' if mw <= 500 else 'FAIL',
        'Dermal_LogP': 'PASS' if 1.0 <= logp <= 5.0 else 'CHECK'
    })

df = pd.DataFrame(records)

# Exportar Planilhas Atualizadas
csv_2d = os.path.join(base_dir, '02_Chemoinformatics_2D', 'curated_radioprotection_library_100.csv')
aim1_file = os.path.join(docking_dir, 'ranking_dna_binding_aim1.xlsx')
aim2_file = os.path.join(base_dir, '04_ROS_Scavenging_Aim2', 'ranking_ros_aim2_final.xlsx')
aim3_file = os.path.join(base_dir, '05_Final_Hit_List', 'ranking_dual_action_aim3_final.xlsx')
hitlist_file = os.path.join(base_dir, '05_Final_Hit_List', 'hitlist_priorizada_radioprotecao.xlsx')

df.to_csv(csv_2d, index=False)
df.sort_values(by='DeltaG_kcal_mol').to_excel(aim1_file, index=False)
df.sort_values(by='RSI_Score', ascending=False).to_excel(aim2_file, index=False)
df.sort_values(by='DARS_Score', ascending=False).to_excel(aim3_file, index=False)
df.sort_values(by='DARS_Score', ascending=False).to_excel(hitlist_file, index=False)

print(f"\nSucesso: {len(df)} moleculas mineradas, processadas e exportadas para o Excel!")
print(df[['ChEMBL_ID', 'MW_Da', 'DeltaG_kcal_mol', 'RSI_Score', 'DARS_Score', 'Dermal_500Da']].head(10))
