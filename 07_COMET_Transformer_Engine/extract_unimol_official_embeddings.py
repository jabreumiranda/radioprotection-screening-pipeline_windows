import os
import numpy as np
import pandas as pd
from unimol_tools import UniMolRepr

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project\07_COMET_Transformer_Engine')
in_xlsx = os.path.join(base_dir, 'comet_training_dataset_labeled.xlsx')
out_unimol_npy = os.path.join(base_dir, 'comet_unimol_3d_embeddings_248.npy')

print("=" * 70)
print("EXTRACAO DE EMBEDDINGS 3D NATIVOS VIA UNI-MOL PRE-TREINADO (512D)")
print("=" * 70)

df = pd.read_excel(in_xlsx)
smiles_list = df['SMILES'].tolist()
print(f"Processando {len(smiles_list)} moleculas atraves do Uni-Mol...")

clf = UniMolRepr(data_type='molecule', remove_hs=False)
unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=False)

# Trata retorno como lista direta de embeddings [CLS] de 512D
if isinstance(unimol_repr, dict):
    cls_repr = np.array(unimol_repr['cls_repr'], dtype=np.float32)
else:
    cls_repr = np.array(unimol_repr, dtype=np.float32)

print("\n" + "=" * 70)
print(f"MATRIZ UNI-MOL EXTRAIDA COM SUCESSO: {cls_repr.shape}")
print(f"Salvando em: {out_unimol_npy}")
print("=" * 70 + "\n")

np.save(out_unimol_npy, cls_repr)
