import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project\07_COMET_Transformer_Engine')
models_dir = os.path.join(base_dir, 'trained_unimol_ensemble_models')
os.makedirs(models_dir, exist_ok=True)

# Carregar Embeddings Uni-Mol Oficiais e Rótulos
X_data = np.load(os.path.join(base_dir, 'comet_unimol_3d_embeddings_248.npy'))
df_meta = pd.read_excel(os.path.join(base_dir, 'comet_training_dataset_labeled.xlsx'))

task_cols = [
    'Task0_DNA_Affinity_kcal_mol',
    'Task1_RSI_Score',
    'Task2_Comet_Radioprotection_Score',
    'Task3_Safety_Viability_Score'
]
Y_data = df_meta[task_cols].values.astype(np.float32)
Y_data[:, 0] = np.abs(Y_data[:, 0]) # Afinidade positiva

# Arquitetura Transformer COMET (Adaptada para entrada 512D do Uni-Mol)
class OfficialCOMETHead(nn.Module):
    def __init__(self, input_dim=512, embed_dim=256, n_heads=4, num_layers=2, num_tasks=4):
        super(OfficialCOMETHead, self).__init__()
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
            p_sub = diff_p[mask]
            y_sub = diff_y[mask]
            loss_t = -torch.log(torch.sigmoid(p_sub - lambda_margin * y_sub) + 1e-8).mean()
            total_loss += loss_t
    return total_loss / num_tasks

print("=" * 70)
print("TREINANDO ENSEMBLE COMET SOBRE EMBEDDINGS 3D OFICIAIS DO UNI-MOL")
print("=" * 70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
ensemble_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_data)):
    X_train, Y_train = torch.tensor(X_data[train_idx]), torch.tensor(Y_data[train_idx])
    X_val, Y_val = torch.tensor(X_data[val_idx]), torch.tensor(Y_data[val_idx])
    
    model = OfficialCOMETHead(input_dim=512, embed_dim=256, n_heads=4, num_layers=2, num_tasks=4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model.train()
    for epoch in range(70):
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
        
    spearman_scores = [spearmanr(val_preds[:, t], val_targets[:, t])[0] for t in range(4)]
    pearson_scores = [pearsonr(val_preds[:, t], val_targets[:, t])[0] for t in range(4)]
    
    print(f"\n---> Fold {fold+1}/5:")
    print(f"   Task 0 (DNA Binding)       : Spearman r_s = {spearman_scores[0]:.3f} | Pearson r = {pearson_scores[0]:.3f}")
    print(f"   Task 1 (ROS Scavenging)    : Spearman r_s = {spearman_scores[1]:.3f} | Pearson r = {pearson_scores[1]:.3f}")
    print(f"   Task 2 (Comet Radioprot)   : Spearman r_s = {spearman_scores[2]:.3f} | Pearson r = {pearson_scores[2]:.3f}")
    print(f"   Task 3 (Safety Index)      : Spearman r_s = {spearman_scores[3]:.3f} | Pearson r = {pearson_scores[3]:.3f}")
    
    ensemble_metrics.append(spearman_scores)
    torch.save(model.state_dict(), os.path.join(models_dir, f'comet_unimol_fold_{fold+1}.pt'))

mean_spearman = np.mean(ensemble_metrics, axis=0)

print("\n" + "=" * 70)
print("PERFORMANCE MEDIA DO ENSEMBLE COMET + UNI-MOL (5-FOLD CV):")
print(f"-> Task 0 (DNA Binding):       Spearman r_s = {mean_spearman[0]:.3f}")
print(f"-> Task 1 (ROS Scavenging):    Spearman r_s = {mean_spearman[1]:.3f}")
print(f"-> Task 2 (Comet Radioprot):   Spearman r_s = {mean_spearman[2]:.3f}")
print(f"-> Task 3 (Safety Index):      Spearman r_s = {mean_spearman[3]:.3f}")
print(f"Modelos salvos em: {models_dir}")
print("=" * 70 + "\n")
