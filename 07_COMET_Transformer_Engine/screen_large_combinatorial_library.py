import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski
from sklearn.cluster import KMeans

RDLogger.DisableLog('rdApp.*')

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project\07_COMET_Transformer_Engine')
models_dir = os.path.join(base_dir, 'trained_ensemble_models')
out_100_hits_xlsx = os.path.join(base_dir, 'comet_top_100_exploratory_hits.xlsx')
out_full_screen_xlsx = os.path.join(base_dir, 'comet_virtual_library_screened.xlsx')

print("=" * 75)
print("MINERACAO E TRIAGEM COMBINATORIA EM LARGA ESCALA VIA ENSEMBLE COMET")
print("=" * 75)

# 1. ENUMERACAO COMBINATORIA DE BIBLIOTECA VIRTUAL DE RADIOPROTETORES
# Esqueletos de Ancoragem em DNA (Aim 1) x Grupos Funcionais Redox (Aim 2)
scaffolds = [
    # Bis-heterociclos e Intercaladores
    {"name": "Bisbenzimidazole_Core", "smiles": "c1cc2nc([*])nc2cc1-c1ccc2[nH]c([*])nc2c1"},
    {"name": "Furamidine_Core", "smiles": "c1cc([*])ccc1-c1oc(-c2ccc([*])cc2)cc1"},
    {"name": "Acridine_Core", "smiles": "c1cc2c(cc1)nc3ccccc3c2[*]"},
    {"name": "Carbazole_Core", "smiles": "c1ccc2c(c1)[nH]c1ccccc12"},
    {"name": "Quinoline_Amide", "smiles": "c1ccc2nc([*])ccc2c1"},
    {"name": "Flavone_Core", "smiles": "O=c1cc(-c2ccc([*])cc2)oc2cc([*])ccc12"},
    {"name": "Isoflavone_Core", "smiles": "O=c1c(-c2ccc([*])cc2)coc2cc([*])ccc12"},
    {"name": "Stilbene_Core", "smiles": "c1cc([*])ccc1/C=C/c1ccc([*])cc1"},
    {"name": "Chalcone_Core", "smiles": "O=C(/C=C/c1ccc([*])cc1)c1ccc([*])cc1"},
    {"name": "Coumarin_Core", "smiles": "O=c1oc2cc([*])ccc2cc1[*]"},
    {"name": "Naphthoquinone_Core", "smiles": "O=C1C=C([*])C(=O)c2ccccc12"},
    {"name": "Anthraquinone_Core", "smiles": "O=C1c2ccccc2C(=O)c2cc([*])ccc12"}
]

side_chains = [
    # Grupos Redox & Varredores de ROS (Aim 2)
    {"name": "Cysteamine_Thiol", "smiles": "NCCS"},
    {"name": "Mercaptoethanol", "smiles": "OCCS"},
    {"name": "Lipoic_Dithiol", "smiles": "CCCC1CCSS1"},
    {"name": "Ebselen_Organoseleno", "smiles": "N1C(=O)c2ccccc2[Se]1"},
    {"name": "Selenocysteine_Amide", "smiles": "NC(C[SeH])C(=O)N"},
    {"name": "Catechol_Hydroxyl", "smiles": "c1c(O)c(O)ccc1"},
    {"name": "Pyrogallol_Trihydroxy", "smiles": "c1c(O)c(O)c(O)cc1"},
    {"name": "Gallic_Acid_Ester", "smiles": "C(=O)c1cc(O)c(O)c(O)c1"},
    {"name": "Trolox_Chroman", "smiles": "CC1(C)CCc2c(O)c(C)c(C)c(C)c2O1"},
    {"name": "Tempol_Nitroxide", "smiles": "CC1(C)CC(O)CC(C)(C)N1[O]"},
    # Grupos de Afinidade Cationica e Sulco de DNA (Aim 1)
    {"name": "Guanidino_Propyl", "smiles": "NCCC(=N)N"},
    {"name": "Amidino_Phenyl", "smiles": "c1ccc(C(=N)N)cc1"},
    {"name": "Spermine_Polyamine", "smiles": "NCCCNCCCCNCCCN"},
    {"name": "Dimethylamino_Ethyl", "smiles": "N(C)CCN(C)C"},
    {"name": "Methylpiperazine", "smiles": "N1CCN(C)CC1"}
]

print("-> Enumerando espaco quimico combinatorio...")
generated_candidates = []
seen_smiles = set()

for scaf in scaffolds:
    for sc1 in side_chains:
        for sc2 in side_chains:
            # Construcao modular de candidatos bi-funcionais
            s_smi = scaf['smiles']
            # Substituicao do primeiro ponto de acoplamento [*]
            s_smi = s_smi.replace("[*]", sc1['smiles'], 1)
            # Substituicao do segundo ponto de acoplamento [*]
            if "[*]" in s_smi:
                s_smi = s_smi.replace("[*]", sc2['smiles'], 1)
            
            mol = Chem.MolFromSmiles(s_smi)
            if mol is not None:
                can_smi = Chem.MolToSmiles(mol)
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                if 200 <= mw <= 900 and -2.0 <= logp <= 6.5 and can_smi not in seen_smiles:
                    seen_smiles.add(can_smi)
                    generated_candidates.append({
                        'Candidate_ID': f'VIRT_MOL_{len(generated_candidates)+1:05d}',
                        'Scaffold': scaf['name'],
                        'SideChain_A': sc1['name'],
                        'SideChain_B': sc2['name'],
                        'SMILES': can_smi,
                        'MW': mw,
                        'LogP': logp,
                        'TPSA': Descriptors.TPSA(mol),
                        'HBD': Descriptors.NumHDonors(mol),
                        'HBA': Descriptors.NumHAcceptors(mol),
                        'RotBonds': Descriptors.NumRotatableBonds(mol),
                        'AromaticRings': Lipinski.NumAromaticRings(mol)
                    })

df_virtual = pd.DataFrame(generated_candidates)
print(f"   Total de moleculas virtuais unicas sintetizaveis geradas: {len(df_virtual)}")

# 2. EXTRACAO DE FEATURES E EMBEDDINGS 3D/ECFP4
print("\n-> Extraindo embeddings de alta dimensao (2056D)...")
feature_list = []
fps_for_clustering = []

for idx, row in df_virtual.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.array(fp, dtype=np.float32)
    fps_for_clustering.append(fp_array)
    
    phys_vec = np.array([
        row['MW'], row['LogP'], row['TPSA'], row['HBD'],
        row['HBA'], row['RotBonds'], 0.0, row['AromaticRings']
    ], dtype=np.float32)
    
    feature_list.append(np.concatenate([fp_array, phys_vec]))

X_virtual = np.array(feature_list)
X_virtual_tensor = torch.tensor(X_virtual, dtype=torch.float32)

# 3. CARREGAR ENSEMBLE COMET (5 FOLDS) E EXECUTAR INFERENCIA
class COMETTransformer(nn.Module):
    def __init__(self, input_dim=2056, embed_dim=256, n_heads=4, num_layers=2, num_tasks=4):
        super(COMETTransformer, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1))
            for _ in range(num_tasks)
        ])
        
    def forward(self, x):
        b = x.size(0)
        proj = self.input_proj(x).unsqueeze(1)
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat((cls, proj), dim=1)
        out = self.transformer(seq)
        cls_feat = out[:, 0, :]
        preds = [h(cls_feat) for h in self.heads]
        return torch.cat(preds, dim=1)

print("\n-> Executando inferencia pelo Ensemble COMET (5 Folds)...")
models = []
for fold in range(1, 6):
    m = COMETTransformer(input_dim=2056, embed_dim=256, n_heads=4, num_layers=2, num_tasks=4)
    m.load_state_dict(torch.load(os.path.join(models_dir, f'comet_fold_{fold}.pt')))
    m.eval()
    models.append(m)

all_preds = []
with torch.no_grad():
    for m in models:
        preds = m(X_virtual_tensor).cpu().numpy()
        all_preds.append(preds)

all_preds = np.array(all_preds) # [5, N, 4]
mean_preds = np.mean(all_preds, axis=0)
std_preds = np.std(all_preds, axis=0) # Incerteza Epistemica

df_virtual['Pred_DNA_Affinity_Score'] = mean_preds[:, 0]
df_virtual['Pred_ROS_Scavenging_Score'] = mean_preds[:, 1]
df_virtual['Pred_Comet_Radioprotection'] = mean_preds[:, 2]
df_virtual['Pred_Safety_Index'] = mean_preds[:, 3]
df_virtual['Ensemble_Uncertainty_Std'] = np.mean(std_preds, axis=1)

# Score Integrado Global COMET
df_virtual['COMET_Integrated_Score'] = (
    df_virtual['Pred_Comet_Radioprotection'] * 0.40 +
    df_virtual['Pred_DNA_Affinity_Score'] * 0.25 +
    df_virtual['Pred_ROS_Scavenging_Score'] * 0.25 +
    df_virtual['Pred_Safety_Index'] * 0.10
)

# 4. FILTRAGEM MULTIOBJETIVO (TOP 20% EM PROPRIEDADES-CHAVE)
q_dna = df_virtual['Pred_DNA_Affinity_Score'].quantile(0.60)
q_ros = df_virtual['Pred_ROS_Scavenging_Score'].quantile(0.60)
q_comet = df_virtual['Pred_Comet_Radioprotection'].quantile(0.60)
q_uncert = df_virtual['Ensemble_Uncertainty_Std'].quantile(0.80) # Penaliza alta incerteza

df_filtered = df_virtual[
    (df_virtual['Pred_DNA_Affinity_Score'] >= q_dna) &
    (df_virtual['Pred_ROS_Scavenging_Score'] >= q_ros) &
    (df_virtual['Pred_Comet_Radioprotection'] >= q_comet) &
    (df_virtual['Ensemble_Uncertainty_Std'] <= q_uncert)
].copy()

print(f"\n-> Candidatos aprovados no filtro multiobjetivo (Top %): {len(df_filtered)}")

# 5. K-MEANS CLUSTERING (K=100) PARA GARANTIR DIVERSIDADE QUIMICA DOS 100 EXPLORATORY HITS
print("-> Executando K-means Clustering (K=100) para garantir diversidade quimica...")
idx_filtered = df_filtered.index.tolist()
fps_filtered = np.array([fps_for_clustering[i] for i in idx_filtered])

n_clusters = min(100, len(df_filtered))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(fps_filtered)
df_filtered['Cluster_ID'] = cluster_labels

# Selecionar o melhor composto de cada cluster pelo COMET_Integrated_Score
top_100_hits = []
for cid in range(n_clusters):
    cluster_df = df_filtered[df_filtered['Cluster_ID'] == cid]
    best_mol = cluster_df.sort_values(by='COMET_Integrated_Score', ascending=False).iloc[0]
    top_100_hits.append(best_mol)

df_top100 = pd.DataFrame(top_100_hits).sort_values(by='COMET_Integrated_Score', ascending=False).reset_index(drop=True)
df_top100['Rank'] = range(1, len(df_top100) + 1)

# Salvar Planilhas
df_virtual.to_excel(out_full_screen_xlsx, index=False)
df_top100.to_excel(out_100_hits_xlsx, index=False)

print("\n" + "=" * 75)
print(f"SUCESSO: {len(df_top100)} EXPLORATORY HITS DIVERSOS SELECIONADOS COM EXITO!")
print(f"Planilha dos 100 Hits: {out_100_hits_xlsx}")
print(f"Planilha da Triagem Completa: {out_full_screen_xlsx}")
print("=" * 75)
print("\nTOP 10 EXPLORATORY HITS (REPRESENTATIVOS DOS MELHORES CLUSTERS):")
cols_p = ['Rank', 'Candidate_ID', 'Scaffold', 'SideChain_A', 'SideChain_B', 'Pred_DNA_Affinity_Score', 'Pred_ROS_Scavenging_Score', 'COMET_Integrated_Score', 'Ensemble_Uncertainty_Std']
print(df_top100[cols_p].head(10).to_string(index=False))
print("=" * 75 + "\n")
