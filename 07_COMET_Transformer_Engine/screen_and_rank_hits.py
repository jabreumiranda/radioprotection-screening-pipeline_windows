import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski

RDLogger.DisableLog('rdApp.*')

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project\07_COMET_Transformer_Engine')
models_dir = os.path.join(base_dir, 'trained_ensemble_models')
out_ranked_xlsx = os.path.join(base_dir, 'comet_virtual_screening_top_hits.xlsx')

print("=" * 70)
print("TRIAGEM VIRTUAL MULTIOBJETIVO VIA ENSEMBLE COMET (5 FOLDS)")
print("=" * 70)

# 1. Carregar Dados de Treinamento e Definir Arquitetura
df_train = pd.read_excel(os.path.join(base_dir, 'comet_training_dataset_labeled.xlsx'))
input_dim = 2056

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
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(num_tasks)
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

# 2. Carregar os 5 Modelos do Ensemble
models = []
for fold in range(1, 6):
    model_path = os.path.join(models_dir, f'comet_fold_{fold}.pt')
    m = COMETTransformer(input_dim=input_dim, embed_dim=256, n_heads=4, num_layers=2, num_tasks=4)
    m.load_state_dict(torch.load(model_path))
    m.eval()
    models.append(m)

print(f"-> Ensemble de {len(models)} modelos carregado com sucesso.")

# 3. Executar Predicoes por Ensemble sobre a Biblioteca
X_all = np.load(os.path.join(base_dir, 'comet_embeddings_248.npy'))
tensor_X = torch.tensor(X_all, dtype=torch.float32)

all_preds = []
with torch.no_grad():
    for m in models:
        pred = m(tensor_X).cpu().numpy()
        all_preds.append(pred)

all_preds = np.array(all_preds) # [5, N, 4]
mean_preds = np.mean(all_preds, axis=0) # [N, 4]
std_preds = np.std(all_preds, axis=0)  # [N, 4] Incerteza Epistemica

df_train['COMET_Pred_DNA_Affinity'] = mean_preds[:, 0]
df_train['COMET_Pred_ROS_Scavenging'] = mean_preds[:, 1]
df_train['COMET_Pred_Radioprotection'] = mean_preds[:, 2]
df_train['COMET_Pred_Safety'] = mean_preds[:, 3]
df_train['COMET_Uncertainty_Std'] = np.mean(std_preds, axis=1)

# Calculo do Score Final Integrado COMET (Multiobjetivo Balanceado)
df_train['COMET_Global_Score'] = (
    df_train['COMET_Pred_Radioprotection'] * 0.40 +
    df_train['COMET_Pred_DNA_Affinity'] * 0.25 +
    df_train['COMET_Pred_ROS_Scavenging'] * 0.25 +
    df_train['COMET_Pred_Safety'] * 0.10
)

# Ranqueamento decrescente
df_ranked = df_train.sort_values(by='COMET_Global_Score', ascending=False).reset_index(drop=True)
df_ranked.to_excel(out_ranked_xlsx, index=False)

print("\n" + "=" * 70)
print("TOP 10 CANDIDATOS RANQUEADOS PELO MODELO COMET (AIM 1 + AIM 2):")
print("=" * 70)
cols_show = ['ChEMBL_ID', 'Compound_Name', 'Class', 'COMET_Pred_DNA_Affinity', 'COMET_Pred_ROS_Scavenging', 'COMET_Pred_Radioprotection', 'COMET_Global_Score', 'COMET_Uncertainty_Std']
print(df_ranked[cols_show].head(10).to_string(index=False))
print("\n" + "=" * 70)
print(f"Ranking completo salvo em: {out_ranked_xlsx}")
print("=" * 70 + "\n")
