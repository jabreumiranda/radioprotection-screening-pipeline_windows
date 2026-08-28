import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski
from unimol_tools import UniMolRepr

RDLogger.DisableLog('rdApp.*')

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project\07_COMET_Transformer_Engine')
models_dir = os.path.join(base_dir, 'trained_smallmol_comet_models')
os.makedirs(models_dir, exist_ok=True)

out_100_hits_xlsx = os.path.join(base_dir, 'comet_smallmol_top_100_exploratory_hits.xlsx')
out_full_screen_xlsx = os.path.join(base_dir, 'comet_smallmol_virtual_library_screened.xlsx')

print("=" * 75)
print("COMET PIPELINE ADAPTADO: SMALL MOLECULES MULTITAREFA (PHASE 1 GRANT)")
print("=" * 75)

# 1. ESQUELETOS BASE (SCAFFOLDS DE ANCORAGEM NO DNA - AIM 1)
scaffolds = [
    {"name": "Bisbenzimidazole", "smiles": "c1cc2nc([*])nc2cc1-c1ccc2[nH]c([*])nc2c1", "dna_core": 0.85},
    {"name": "Furamidine", "smiles": "c1cc([*])ccc1-c1oc(-c2ccc([*])cc2)cc1", "dna_core": 0.80},
    {"name": "Acridine", "smiles": "c1cc2c(cc1)nc3ccccc3c2[*]", "dna_core": 0.75},
    {"name": "Carbazole", "smiles": "c1ccc2c(c1)[nH]c1ccccc12", "dna_core": 0.60},
    {"name": "Quinoline", "smiles": "c1ccc2nc([*])ccc2c1", "dna_core": 0.55},
    {"name": "Flavone", "smiles": "O=c1cc(-c2ccc([*])cc2)oc2cc([*])ccc12", "dna_core": 0.70},
    {"name": "Isoflavone", "smiles": "O=c1c(-c2ccc([*])cc2)coc2cc([*])ccc12", "dna_core": 0.65},
    {"name": "Stilbene", "smiles": "c1cc([*])ccc1/C=C/c1ccc([*])cc1", "dna_core": 0.50},
    {"name": "Chalcone", "smiles": "O=C(/C=C/c1ccc([*])cc1)c1ccc([*])cc1", "dna_core": 0.45},
    {"name": "Coumarin", "smiles": "O=c1oc2cc([*])ccc2cc1[*]", "dna_core": 0.40},
    {"name": "Pyrene_Aromatic", "smiles": "c1cc2cccc3cccc4cccc(c1)c4c32", "dna_core": 0.70},
    {"name": "Anthraquinone", "smiles": "O=C1c2ccccc2C(=O)c2cc([*])ccc12", "dna_core": 0.65}
]

# 2. GRUPOS FUNCIONAIS LATERAIS (AIM 1: CATIONICO/NLS + AIM 2: REDOX/ANTIOXIDANTE)
modifications = [
    # Cationicos e Miméticos de Dsup (DNA binding)
    {"name": "Tri_Arginine_Mimic", "smiles": "NCCC(=N)N", "cat": 1.0, "nls": 0.6, "redox": 0.1, "tox": 0.2},
    {"name": "Primary_Amine", "smiles": "NCCCN", "cat": 0.8, "nls": 0.3, "redox": 0.0, "tox": 0.1},
    {"name": "Guanidinium", "smiles": "NC(=N)N", "cat": 0.9, "nls": 0.4, "redox": 0.0, "tox": 0.1},
    {"name": "Dimethylamino", "smiles": "CCN(C)C", "cat": 0.7, "nls": 0.2, "redox": 0.0, "tox": 0.1},
    {"name": "Spermine_Polyamine", "smiles": "NCCCNCCCCN", "cat": 1.0, "nls": 0.7, "redox": 0.1, "tox": 0.3},
    # NLS e Penetração
    {"name": "NLS_Mimic_ProLys", "smiles": "CC(C)C(N)C(=O)N1CCCC1C(=O)O", "cat": 0.4, "nls": 0.9, "redox": 0.0, "tox": 0.0},
    {"name": "Benzylamine_Hydrophobic", "smiles": "NCc1ccccc1", "cat": 0.2, "nls": 0.3, "redox": 0.2, "tox": 0.1},
    {"name": "Naphthyl_Hydrophobic", "smiles": "NCc1cccc2ccccc12", "cat": 0.2, "nls": 0.4, "redox": 0.3, "tox": 0.2},
    # Antioxidantes / Scavengers de Radicais Livres (Aim 2)
    {"name": "Cysteamine_Thiol", "smiles": "NCCS", "cat": 0.5, "nls": 0.2, "redox": 0.9, "tox": 0.1},
    {"name": "Ebselen_Organoseleno", "smiles": "N1C(=O)c2ccccc2[Se]1", "cat": 0.1, "nls": 0.4, "redox": 0.95, "tox": 0.1},
    {"name": "Catechol_Hydroxyl", "smiles": "c1c(O)c(O)ccc1", "cat": 0.0, "nls": 0.2, "redox": 0.90, "tox": 0.0},
    {"name": "Pyrogallol_Trihydroxy", "smiles": "c1c(O)c(O)c(O)cc1", "cat": 0.0, "nls": 0.1, "redox": 0.95, "tox": 0.1},
    {"name": "Trolox_Chroman", "smiles": "CC1(C)CCc2c(O)c(C)c(C)c(C)c2O1", "cat": 0.0, "nls": 0.3, "redox": 0.85, "tox": 0.0},
    {"name": "Tempol_Nitroxide", "smiles": "CC1(C)CC(O)CC(C)(C)N1[O]", "cat": 0.1, "nls": 0.4, "redox": 0.88, "tox": 0.1}
]

print("1. Enumerando biblioteca combinatorial de Small Molecules...")
library = []
seen_smiles = set()

for scaf in scaffolds:
    for m1 in modifications:
        for m2 in modifications:
            s_smi = scaf['smiles'].replace("[*]", m1['smiles'], 1)
            if "[*]" in s_smi:
                s_smi = s_smi.replace("[*]", m2['smiles'], 1)
            
            mol = Chem.MolFromSmiles(s_smi)
            if mol is not None:
                can_smi = Chem.MolToSmiles(mol)
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                if 220 <= mw <= 850 and -1.5 <= logp <= 6.0 and can_smi not in seen_smiles:
                    seen_smiles.add(can_smi)
                    
                    # Biofísica calibrada para as 5 tarefas do grant:
                    # Task 0: DNA binding Kd (uM) - alvo 1 - 10 uM
                    kd = float(np.clip(14.0 - 8.0 * scaf['dna_core'] - 3.5 * ((m1['cat'] + m2['cat']) / 2.0) + np.random.normal(0, 0.4), 0.5, 25.0))
                    
                    # Task 1: Nuclear:Cytoplasmic Ratio - alvo > 2.0
                    nls_score = (m1['nls'] + m2['nls']) / 2.0
                    nc_ratio = float(np.clip(0.8 + 2.5 * nls_score + 0.3 * (logp > 1.5 and logp < 3.5) + np.random.normal(0, 0.1), 0.4, 4.5))
                    
                    # Task 2: Cellular Uptake % em 4h - alvo > 60%
                    uptake_pct = float(np.clip(30.0 + 45.0 * ((m1['cat'] + m2['cat'] + m1['nls'] + m2['nls']) / 4.0) + 15.0 * (logp > 1.0) + np.random.normal(0, 2.0), 15.0, 98.0))
                    
                    # Task 3: IC50 Cytotoxicity (mg/mL) - alvo > 1.0 mg/mL
                    tox_penalty = (m1['tox'] + m2['tox']) / 2.0
                    ic50 = float(np.clip(2.8 - 1.2 * tox_penalty - 0.2 * (mw > 650) + np.random.normal(0, 0.1), 0.15, 3.5))
                    
                    # Task 4: Radioprotection % no Ensaio Cometa - alvo > 40% (Sinergia Aim 1 + Aim 2)
                    redox_power = (m1['redox'] + m2['redox']) / 2.0
                    rad_prot_pct = float(np.clip((12.0 / kd) * 1.8 + (redox_power * 35.0) + (nc_ratio * 8.0) + np.random.normal(0, 1.5), 10.0, 92.0))
                    
                    library.append({
                        'Candidate_ID': f'SMOL_{len(library)+1:05d}',
                        'Scaffold': scaf['name'],
                        'Mod_A': m1['name'],
                        'Mod_B': m2['name'],
                        'SMILES': can_smi,
                        'MW': round(mw, 1),
                        'LogP': round(logp, 2),
                        'TPSA': round(Descriptors.TPSA(mol), 1),
                        'HBD': Descriptors.NumHDonors(mol),
                        'HBA': Descriptors.NumHAcceptors(mol),
                        'Task0_DNA_Binding_Kd_uM': round(kd, 2),
                        'Task1_Nuclear_Cytoplasmic_Ratio': round(nc_ratio, 2),
                        'Task2_Cellular_Uptake_4h_Pct': round(uptake_pct, 1),
                        'Task3_Cytotoxicity_IC50_mg_mL': round(ic50, 2),
                        'Task4_Radioprotection_Damage_Reduction_Pct': round(rad_prot_pct, 1)
                    })

df_lib = pd.DataFrame(library)
print(f"-> Biblioteca combinatorial de Small Molecules gerada: {len(df_lib)} compostos.")

# 2. EXTRACAO DE EMBEDDINGS 3D VIA UNI-MOL
print("\n2. Extraindo representações 3D via Uni-Mol (512D)...")
clf = UniMolRepr(data_type='molecule', remove_hs=False)
unimol_repr = clf.get_repr(df_lib['SMILES'].tolist(), return_atomic_reprs=False)

if isinstance(unimol_repr, dict):
    X_unimol = np.array(unimol_repr['cls_repr'], dtype=np.float32)
else:
    X_unimol = np.array(unimol_repr, dtype=np.float32)

task_cols = [
    'Task0_DNA_Binding_Kd_uM',
    'Task1_Nuclear_Cytoplasmic_Ratio',
    'Task2_Cellular_Uptake_4h_Pct',
    'Task3_Cytotoxicity_IC50_mg_mL',
    'Task4_Radioprotection_Damage_Reduction_Pct'
]
Y_all = df_lib[task_cols].values.astype(np.float32)
# Para Kd, menor = melhor, entao invertemos o sinal para ranqueamento pareado
Y_all[:, 0] = -Y_all[:, 0]

# 3. ARQUITETURA TRANSFORMER DO COMET (5 TASKS)
class SmallMolCOMET(nn.Module):
    def __init__(self, input_dim=512, embed_dim=256, n_heads=4, num_layers=2, num_tasks=5):
        super(SmallMolCOMET, self).__init__()
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

def pairwise_ranking_loss(preds, targets, lambda_margin=0.01):
    total_loss = 0.0
    num_tasks = preds.size(1)
    for t in range(num_tasks):
        p = preds[:, t]
        y = targets[:, t]
        diff_p = p.unsqueeze(1) - p.unsqueeze(0)
        diff_y = y.unsqueeze(1) - y.unsqueeze(0)
        mask = diff_y > 0
        if mask.sum() > 0:
            loss_t = -torch.log(torch.sigmoid(diff_p[mask] - lambda_margin * diff_y[mask]) + 1e-8).mean()
            total_loss += loss_t
    return total_loss / num_tasks

# 4. TREINAMENTO DO ENSEMBLE DE 5 MODELOS (5-FOLD CROSS-VALIDATION)
print("\n3. Treinando Ensemble COMET (5 Folds)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
trained_models = []
metrics_spearman = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_unimol)):
    X_train, Y_train = torch.tensor(X_unimol[train_idx]), torch.tensor(Y_all[train_idx])
    X_val, Y_val = torch.tensor(X_unimol[val_idx]), torch.tensor(Y_all[val_idx])
    
    model = SmallMolCOMET(input_dim=512, embed_dim=256, n_heads=4, num_layers=2, num_tasks=5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model.train()
    for epoch in range(60):
        optimizer.zero_grad()
        noise = torch.randn_like(X_train) * 0.02
        preds = model(X_train + noise)
        loss = pairwise_ranking_loss(preds, Y_train, lambda_margin=0.01)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).cpu().numpy()
        val_targets = Y_val.cpu().numpy()
        
    spearmans = [spearmanr(val_preds[:, t], val_targets[:, t])[0] for t in range(5)]
    metrics_spearman.append(spearmans)
    trained_models.append(model)
    torch.save(model.state_dict(), os.path.join(models_dir, f'comet_smallmol_fold_{fold+1}.pt'))

mean_sp = np.mean(metrics_spearman, axis=0)
print(f"-> Desempenho Medio de Validacao (Spearman r_s):")
print(f"   Task 0 (DNA Binding Kd):      r_s = {mean_sp[0]:.3f}")
print(f"   Task 1 (N:C Ratio > 2.0):     r_s = {mean_sp[1]:.3f}")
print(f"   Task 2 (Cellular Uptake %):   r_s = {mean_sp[2]:.3f}")
print(f"   Task 3 (Cytotoxicity IC50):   r_s = {mean_sp[3]:.3f}")
print(f"   Task 4 (Radioprotection %):   r_s = {mean_sp[4]:.3f}")

# 5. INFERENCIA, FILTRAGEM MULTIOBJETIVO (TOP 20%) E K-MEANS CLUSTERING (100 HITS)
print("\n4. Executando Inferência, Filtro Multiobjetivo e K-Means (K=100)...")
tensor_X_all = torch.tensor(X_unimol, dtype=torch.float32)

all_preds = []
with torch.no_grad():
    for m in trained_models:
        all_preds.append(m(tensor_X_all).cpu().numpy())

all_preds = np.array(all_preds) # [5, N, 5]
mean_preds = np.mean(all_preds, axis=0)
std_preds = np.std(all_preds, axis=0)

df_lib['Pred_DNA_Binding_Kd_Score'] = mean_preds[:, 0]
df_lib['Pred_NC_Ratio_Score'] = mean_preds[:, 1]
df_lib['Pred_Cellular_Uptake_Score'] = mean_preds[:, 2]
df_lib['Pred_Safety_IC50_Score'] = mean_preds[:, 3]
df_lib['Pred_Radioprotection_Score'] = mean_preds[:, 4]
df_lib['Ensemble_Uncertainty_Std'] = np.mean(std_preds, axis=1)

# Escore Global Ponderado do Grant
df_lib['COMET_Global_Score'] = (
    df_lib['Pred_Radioprotection_Score'] * 0.35 +
    df_lib['Pred_DNA_Binding_Kd_Score'] * 0.25 +
    df_lib['Pred_NC_Ratio_Score'] * 0.20 +
    df_lib['Pred_Cellular_Uptake_Score'] * 0.10 +
    df_lib['Pred_Safety_IC50_Score'] * 0.10
)

# Filtro Top 20% Multiobjetivo
q_uncert = df_lib['Ensemble_Uncertainty_Std'].quantile(0.80)
df_candidates = df_lib[
    (df_lib['Task0_DNA_Binding_Kd_uM'] <= 8.0) &
    (df_lib['Task1_Nuclear_Cytoplasmic_Ratio'] >= 2.0) &
    (df_lib['Task2_Cellular_Uptake_4h_Pct'] >= 60.0) &
    (df_lib['Task3_Cytotoxicity_IC50_mg_mL'] >= 1.0) &
    (df_lib['Task4_Radioprotection_Damage_Reduction_Pct'] >= 40.0) &
    (df_lib['Ensemble_Uncertainty_Std'] <= q_uncert)
].copy()

print(f"-> Candidatos que cumpriram TODOS os 5 critérios do grant: {len(df_candidates)}")

# K-Means Clustering (K=100) sobre Morgan Fingerprints para garantir diversidade química
fps = []
for smi in df_candidates['SMILES']:
    m = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)
    fps.append(np.array(fp))
fps = np.array(fps)

n_clusters = min(100, len(df_candidates))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_candidates['Cluster_ID'] = kmeans.fit_predict(fps)

# Selecionar o melhor representante de cada cluster
top_100_hits = []
for cid in range(n_clusters):
    c_df = df_candidates[df_candidates['Cluster_ID'] == cid]
    top_100_hits.append(c_df.sort_values(by='COMET_Global_Score', ascending=False).iloc[0])

df_top100 = pd.DataFrame(top_100_hits).sort_values(by='COMET_Global_Score', ascending=False).reset_index(drop=True)
df_top100['Rank'] = range(1, len(df_top100) + 1)

# Exportar Planilhas Finais
df_lib.to_excel(out_full_screen_xlsx, index=False)
df_top100.to_excel(out_100_hits_xlsx, index=False)

print("\n" + "=" * 75)
print(f"SUCESSO: {len(df_top100)} EXPLORATORY HITS DIVERSOS SELECIONADOS PELO COMET!")
print(f"Arquivo dos 100 Hits: {out_100_hits_xlsx}")
print("=" * 75)
cols_show = ['Rank', 'Candidate_ID', 'Scaffold', 'Mod_A', 'Mod_B', 'Task0_DNA_Binding_Kd_uM', 'Task1_Nuclear_Cytoplasmic_Ratio', 'Task2_Cellular_Uptake_4h_Pct', 'Task4_Radioprotection_Damage_Reduction_Pct', 'COMET_Global_Score']
print(df_top100[cols_show].head(10).to_string(index=False))
print("=" * 75 + "\n")
