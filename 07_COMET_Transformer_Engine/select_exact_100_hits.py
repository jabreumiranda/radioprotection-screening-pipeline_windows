import os
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans

RDLogger.DisableLog('rdApp.*')

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project\07_COMET_Transformer_Engine')
in_screened_xlsx = os.path.join(base_dir, 'comet_smallmol_virtual_library_screened.xlsx')
out_100_hits_xlsx = os.path.join(base_dir, 'comet_smallmol_top_100_exploratory_hits.xlsx')

print("=" * 75)
print("OTIMIZACAO MULTIOBJETIVO E SELECAO DOS 100 EXPLORATORY HITS (K-MEANS)")
print("=" * 75)

df = pd.read_excel(in_screened_xlsx)

# 1. Normalizacao Min-Max para Construcao da Fronteira Multiobjetivo
def min_max_norm(s, invert=False):
    mn, mx = s.min(), s.max()
    norm = (s - mn) / (mx - mn + 1e-8)
    return 1.0 - norm if invert else norm

df['Norm_DNA_Affinity'] = min_max_norm(df['Task0_DNA_Binding_Kd_uM'], invert=True)  # Menor Kd = Maior escore
df['Norm_NC_Ratio'] = min_max_norm(df['Task1_Nuclear_Cytoplasmic_Ratio'])
df['Norm_Uptake'] = min_max_norm(df['Task2_Cellular_Uptake_4h_Pct'])
df['Norm_IC50_Safety'] = min_max_norm(df['Task3_Cytotoxicity_IC50_mg_mL'])
df['Norm_Radioprotection'] = min_max_norm(df['Task4_Radioprotection_Damage_Reduction_Pct'])
df['Norm_Certainty'] = min_max_norm(df['Ensemble_Uncertainty_Std'], invert=True) # Menor incerteza = Maior escore

# 2. Escore Global Multiobjetivo (Pesos Alinhados com os Objetivos do Grant)
df['Grant_Multiobjective_Score'] = (
    df['Norm_Radioprotection'] * 0.30 +
    df['Norm_DNA_Affinity'] * 0.25 +
    df['Norm_NC_Ratio'] * 0.20 +
    df['Norm_Uptake'] * 0.15 +
    df['Norm_IC50_Safety'] * 0.05 +
    df['Norm_Certainty'] * 0.05
)

# 3. K-Means Clustering (K=100) para Garantir Diversidade Estrutural
print("-> Calculando fingerprints Morgan ECFP4 para agrupamento de diversidade...")
fps = []
for smi in df['SMILES']:
    m = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)
    fps.append(np.array(fp, dtype=np.float32))
fps = np.array(fps)

n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Cluster_ID'] = kmeans.fit_predict(fps)

# 4. Selecao do Composto Lider (Medoide) de Cada um dos 100 Clusters
top_100_list = []
for cid in range(n_clusters):
    cluster_subset = df[df['Cluster_ID'] == cid]
    best_candidate = cluster_subset.sort_values(by='Grant_Multiobjective_Score', ascending=False).iloc[0]
    top_100_list.append(best_candidate)

df_top100 = pd.DataFrame(top_100_list).sort_values(by='Grant_Multiobjective_Score', ascending=False).reset_index(drop=True)
df_top100['Rank'] = range(1, len(df_top100) + 1)

# Salvar planilha consolidada dos 100 Hits
df_top100.to_excel(out_100_hits_xlsx, index=False)

print("\n" + "=" * 75)
print(f"SUCESSO: EXATAMENTE {len(df_top100)} EXPLORATORY HITS DIVERSOS SELECIONADOS!")
print(f"Arquivo final salvo em: {out_100_hits_xlsx}")
print("=" * 75)

cols_p = [
    'Rank', 'Candidate_ID', 'Scaffold', 'Mod_A', 'Mod_B',
    'Task0_DNA_Binding_Kd_uM', 'Task1_Nuclear_Cytoplasmic_Ratio',
    'Task2_Cellular_Uptake_4h_Pct', 'Task4_Radioprotection_Damage_Reduction_Pct',
    'Grant_Multiobjective_Score'
]
print("\nTOP 10 CANDIDATOS LIDERES:")
print(df_top100[cols_p].head(10).to_string(index=False))
print("=" * 75 + "\n")
