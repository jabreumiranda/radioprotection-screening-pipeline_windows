import os
import time
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem
from chembl_webresource_client.new_client import new_client

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project\07_COMET_Transformer_Engine')
os.makedirs(base_dir, exist_ok=True)
out_csv = os.path.join(base_dir, 'comet_large_library_raw.csv')

print("=" * 70)
print("MINERACAO ROBUSTA DE ~500 MOLECULAS NO CHEMBL (AIM 1 & AIM 2)")
print("=" * 70)

molecule_api = new_client.molecule
curated_molecules = []
seen_smiles = set()

# 1. Controles Padrao-Ouro (Ground Truth Essencial)
core_controls = [
    {"chembl_id": "REF_AMIFOSTINE", "smiles": "NCCCSCP(=O)(O)O", "pref_name": "Amifostine", "class": "Thiol_Prodrug"},
    {"chembl_id": "REF_WR1065", "smiles": "NCCCS", "pref_name": "WR-1065", "class": "Free_Thiol"},
    {"chembl_id": "REF_TROLOX", "smiles": "CC1=C(C(=C2CCC(O)(C(=O)O)Oc2c1C)C)O", "pref_name": "Trolox", "class": "Phenolic_Antioxidant"},
    {"chembl_id": "REF_ASCORBIC_ACID", "smiles": "OCC(O)C1OC(=O)C(O)=C1O", "pref_name": "Ascorbic_Acid", "class": "Enediol_Antioxidant"},
    {"chembl_id": "REF_EGCG", "smiles": "Oc1cc(O)c2c(c1)OC(c3cc(O)c(O)c(O)c3)C(OC(=O)c4cc(O)c(O)c(O)c4)C2", "pref_name": "EGCG", "class": "Polyphenol"},
    {"chembl_id": "REF_HOECHST33258", "smiles": "CN1CCN(CC1)c2ccc3nc4cc(nc4c3c2)c5ccc(O)cc5", "pref_name": "Hoechst_33258", "class": "MGB_Bisbenzimidazole"},
    {"chembl_id": "REF_DAPI", "smiles": "N=C(N)c1ccc2[nH]c(nc2c1)c3ccc(cc3)C(=N)N", "pref_name": "DAPI", "class": "MGB_Bisamidine"},
    {"chembl_id": "REF_EBSELEN", "smiles": "O=C1c2ccccc2[Se]N1c3ccccc3", "pref_name": "Ebselen", "class": "Organoselenium"},
    {"chembl_id": "REF_TEMPOL", "smiles": "CC1(C)CC(O)CC(C)(C)N1[O]", "pref_name": "Tempol", "class": "Nitroxide_Radical_Scavenger"},
    {"chembl_id": "REF_NETROPSIN", "smiles": "NC(=N)CC(=O)NC1=CC(=NC1)C(=O)NC2=CC(=NC2)C(=O)NCCC(=N)N", "pref_name": "Netropsin", "class": "MGB_Polyamide"},
    {"chembl_id": "REF_DISTAMYCIN", "smiles": "CNC(=O)c1cc(NC(=O)c2cc(NC(=O)c3cc(NC=O)n(C)c3)n(C)c2)n(C)c1", "pref_name": "Distamycin_A", "class": "MGB_Polyamide"},
    {"chembl_id": "REF_PENTAMIDINE", "smiles": "N=C(N)c1ccc(OCCCCCOc2ccc(cc2)C(=N)N)cc1", "pref_name": "Pentamidine", "class": "MGB_Diamidine"}
]

for ctrl in core_controls:
    mol = Chem.MolFromSmiles(ctrl['smiles'])
    can_smi = Chem.MolToSmiles(mol)
    seen_smiles.add(can_smi)
    curated_molecules.append({
        'ChEMBL_ID': ctrl['chembl_id'],
        'Compound_Name': ctrl['pref_name'],
        'SMILES': can_smi,
        'Class': ctrl['class']
    })

def fetch_by_terms(terms, class_label, target_count=130):
    print(f"-> Minerando: {class_label}...")
    collected = 0
    for term in terms:
        if collected >= target_count:
            break
        try:
            res = molecule_api.filter(molecule_synonyms__molecule_synonym__icontains=term).only(
                ['molecule_chembl_id', 'pref_name', 'molecule_structures']
            )[:150]
            
            for item in res:
                if collected >= target_count:
                    break
                smi = item.get('molecule_structures', {}).get('canonical_smiles') if item.get('molecule_structures') else None
                if not smi:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                mw = Descriptors.MolWt(mol)
                if 120 <= mw <= 650:
                    can_smi = Chem.MolToSmiles(mol)
                    if can_smi not in seen_smiles:
                        seen_smiles.add(can_smi)
                        curated_molecules.append({
                            'ChEMBL_ID': item.get('molecule_chembl_id'),
                            'Compound_Name': item.get('pref_name') or item.get('molecule_chembl_id'),
                            'SMILES': can_smi,
                            'Class': class_label
                        })
                        collected += 1
        except Exception as e:
            continue
        time.sleep(0.5)
    print(f"   Recuperados {collected} compostos para {class_label}.")

# 2. Consultas por Famílias Mecanísticas do Projeto
fetch_by_terms(['amidine', 'benzimidazole', 'furamidine', 'diminazene', 'acridine'], 'DNA_MGB_Intercalator', 130)
fetch_by_terms(['thiol', 'cysteine', 'mercapto', 'disulfide', 'diselenide', 'seleno'], 'Thiol_Selenium_Redox', 130)
fetch_by_terms(['quercetin', 'catechin', 'flavone', 'resveratrol', 'phenol', 'hydroxy'], 'Polyphenol_Antioxidant', 130)
fetch_by_terms(['nitroxide', 'ascorb', 'tocopherol', 'coumarin', 'quinone'], 'Radical_Scavenger_Control', 130)

df_all = pd.DataFrame(curated_molecules)
df_all.to_csv(out_csv, index=False)

print("\n" + "=" * 70)
print(f"SUCESSO: {len(df_all)} MOLECULAS CURADAS E PRONTAS PARA O TREINAMENTO!")
print(f"Arquivo consolidado em: {out_csv}")
print("=" * 70 + "\n")
