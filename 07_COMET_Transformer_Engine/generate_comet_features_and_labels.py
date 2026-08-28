import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project\07_COMET_Transformer_Engine')
in_csv = os.path.join(base_dir, 'comet_large_library_raw.csv')
out_xlsx = os.path.join(base_dir, 'comet_training_dataset_labeled.xlsx')
out_emb = os.path.join(base_dir, 'comet_embeddings_248.npy')

print("=" * 70)
print("GERACAO DE FEATURES 3D E ROTULOS MULTITAREFA (DATASET COMET)")
print("=" * 70)

df = pd.read_csv(in_csv)
print(f"Processando {len(df)} moleculas...")

embeddings = []
labeled_rows = []

# Padroes SMARTS para calculo estequiometrico de RSI
SMARTS_REDOX = {
    "Phenol_OH": Chem.MolFromSmarts("c[OH]"),
    "Catechol": Chem.MolFromSmarts("c1c([OH])c([OH])ccc1"),
    "Free_Thiol": Chem.MolFromSmarts("[SX2H]"),
    "Disulfide": Chem.MolFromSmarts("[#16]-[#16]"),
    "Selenium": Chem.MolFromSmarts("[#34]"),
    "Aliphatic_Amine": Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"),
    "Aromatic_Amine": Chem.MolFromSmarts("c[NX3;H2,H1]"),
    "Hydroquinone": Chem.MolFromSmarts("Oc1ccc(O)cc1"),
    "Guanidine_Amidine": Chem.MolFromSmarts("C(=[NH,N])[NH2,NH]")
}

for idx, row in df.iterrows():
    smi = row['SMILES']
    cid = row['ChEMBL_ID']
    name = row['Compound_Name']
    mol_class = row['Class']
    
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
        
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, randomSeed=42)
    
    # 1. Feature Vector Estrutural (ECFP4 2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.array(fp, dtype=np.float32)
    
    # 2. Descritores Fisico-Quimicos
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
    aromatic_rings = Lipinski.NumAromaticRings(mol)
    
    # 3. Calculo Estequiometrico de RSI (Aim 2)
    rsi = 0.0
    for motif_name, smarts_mol in SMARTS_REDOX.items():
        if smarts_mol is not None:
            matches = len(mol.GetSubstructMatches(smarts_mol))
            if motif_name == "Free_Thiol": rsi += matches * 2.5
            elif motif_name == "Selenium": rsi += matches * 3.0
            elif motif_name == "Catechol": rsi += matches * 3.5
            elif motif_name == "Hydroquinone": rsi += matches * 3.5
            elif motif_name == "Phenol_OH": rsi += matches * 1.5
            elif motif_name == "Aliphatic_Amine": rsi += matches * 1.0
            elif motif_name == "Aromatic_Amine": rsi += matches * 1.2
            elif motif_name == "Disulfide": rsi += matches * 1.0
            elif motif_name == "Guanidine_Amidine": rsi += matches * 1.0
            
    # 4. Estimativa Termodinamica de Afinidade com B-DNA (Aim 1 Proxy deltaG)
    # Modelagem linear biofisica baseada no numero de aneis aromaticos, carga cationica e pontes H
    dna_affinity_est = -(3.5 + 0.9 * aromatic_rings + 1.2 * max(0, charge) + 0.3 * hbd)
    # Limitador biofisico realista
    dna_affinity_est = round(float(np.clip(dna_affinity_est, -13.0, -3.0)), 2)
    
    # 5. Comet Radioprotection Index (Score Integrado DARS)
    dars_score = round(float((abs(dna_affinity_est) / 10.0) * 0.5 + (min(rsi, 10.0) / 10.0) * 0.5), 3)
    
    # 6. Safety / Viability Index (Lipinski + Baixa toxicidade)
    safety_score = round(float(1.0 - (max(0, logp - 5.0)*0.1 + max(0, mw - 500)*0.001)), 3)
    safety_score = float(np.clip(safety_score, 0.1, 1.0))
    
    physchem_vec = np.array([mw, logp, tpsa, hbd, hba, rot_bonds, charge, aromatic_rings], dtype=np.float32)
    combined_embedding = np.concatenate([fp_array, physchem_vec])
    embeddings.append(combined_embedding)
    
    labeled_rows.append({
        'ChEMBL_ID': cid,
        'Compound_Name': name,
        'Class': mol_class,
        'SMILES': smi,
        'MW': mw,
        'LogP': logp,
        'TPSA': tpsa,
        'Task0_DNA_Affinity_kcal_mol': dna_affinity_est,
        'Task1_RSI_Score': rsi,
        'Task2_Comet_Radioprotection_Score': dars_score,
        'Task3_Safety_Viability_Score': safety_score
    })

df_labeled = pd.DataFrame(labeled_rows)
X_mat = np.array(embeddings)

df_labeled.to_excel(out_xlsx, index=False)
np.save(out_emb, X_mat)

print("\n" + "=" * 70)
print(f"MATRIZ DE EMBEDDINGS GERADA: {X_mat.shape}")
print(f"DATASET COM ROTULOS MULTITAREFA SALVO EM: {out_xlsx}")
print("=" * 70 + "\n")
print(df_labeled[['ChEMBL_ID', 'Compound_Name', 'Task0_DNA_Affinity_kcal_mol', 'Task1_RSI_Score', 'Task2_Comet_Radioprotection_Score', 'Task3_Safety_Viability_Score']].head(10))
