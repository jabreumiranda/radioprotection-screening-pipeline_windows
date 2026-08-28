import os
import subprocess
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project')
dock_dir = os.path.join(base_dir, '09_Real_DNA_Docking')
tools_dir = os.path.join(base_dir, 'tools')
vina_exe = os.path.join(tools_dir, 'vina.exe')

os.makedirs(dock_dir, exist_ok=True)

# Receptor PDBQT
receptor_pdbqt = os.path.join(dock_dir, '1bna_dna_clean.pdbqt')

# Carregar os 100 candidatos
in_opt_xlsx = os.path.join(base_dir, '07_COMET_Transformer_Engine', 'comet_lead_optimization_100_candidates.xlsx')
df_opt = pd.read_excel(in_opt_xlsx)

print("=" * 75)
print(f"INICIANDO DOCKING FISICO EM LOTE (100 COMPOSTOS) NO B-DNA (1BNA)")
print("=" * 75)
print(f"Executavel Vina: {vina_exe}")
print(f"Receptor:        {receptor_pdbqt}")
print(f"Total de Ligantes: {len(df_opt)}")

# Grid Box no Sulco Menor
center_x, center_y, center_z = 14.7, 20.9, 8.8
size_x, size_y, size_z = 24.0, 24.0, 28.0

docking_results = []
total = len(df_opt)

for idx, row in df_opt.iterrows():
    cand_id = row['Optimized_ID']
    smi = row['SMILES']
    lig_pdbqt = os.path.join(dock_dir, f"{cand_id}.pdbqt")
    out_pdbqt = os.path.join(dock_dir, f"{cand_id}_docked.pdbqt")
    out_log = os.path.join(dock_dir, f"{cand_id}_vina.log")
    
    # 1. Preparação do PDBQT Canônico com Cargas de Gasteiger
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
        AllChem.ComputeGasteigerCharges(mol)
        
        conf = mol.GetConformer()
        atom_lines = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            elem = atom.GetSymbol()
            ad_type = elem
            if elem == 'C' and atom.GetIsAromatic():
                ad_type = 'A'
            elif elem == 'O':
                ad_type = 'OA'
            elif elem == 'N':
                ad_type = 'NA'
            elif elem == 'S':
                ad_type = 'SA'
            elif elem == 'H':
                neigh = [n.GetSymbol() for n in atom.GetNeighbors()]
                ad_type = 'HD' if any(x in ['O', 'N', 'S'] for x in neigh) else 'H'
                
            charge = float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0
            if np.isnan(charge):
                charge = 0.0
                
            atom_name = f"{elem}{i+1}"
            line = f"ATOM  {i+1:5d} {atom_name:<4s} LIG A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00 20.00    {charge:+6.3f} {ad_type:<2s}"
            atom_lines.append(line)
            
        with open(lig_pdbqt, 'w') as f:
            f.write("ROOT\n")
            f.write("\n".join(atom_lines) + "\n")
            f.write("ENDROOT\n")
            f.write("TORSDOF 0\n")
            
        # 2. Executar AutoDock Vina (exhaustiveness = 8)
        vina_cmd = [
            vina_exe,
            "--receptor", receptor_pdbqt,
            "--ligand", lig_pdbqt,
            "--center_x", str(center_x),
            "--center_y", str(center_y),
            "--center_z", str(center_z),
            "--size_x", str(size_x),
            "--size_y", str(size_y),
            "--size_z", str(size_z),
            "--exhaustiveness", "8",
            "--out", out_pdbqt
        ]
        
        res = subprocess.run(vina_cmd, capture_output=True, text=True)
        with open(out_log, 'w') as flog:
            flog.write(res.stdout + "\n" + res.stderr)
            
        affinity = None
        for line in res.stdout.split('\n'):
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == '1':
                try:
                    affinity = float(parts[1])
                    break
                except ValueError:
                    pass
                    
        status = "Sucesso" if affinity is not None else "Sem Score"
    except Exception as e:
        affinity = None
        status = f"Erro: {e}"
        
    docking_results.append({
        'Opt_Rank': row['Opt_Rank'],
        'Candidate_ID': cand_id,
        'Parent_Lead_ID': row['Parent_Lead_ID'],
        'Scaffold': row['Scaffold'],
        'Functional_Strategy': row['Optimization_Strategy'],
        'MW_Da': row['MW'],
        'LogP': row['LogP'],
        'TPSA_A2': row['TPSA'],
        'Pred_Kd_uM': row['Pred_DNA_Binding_Kd_uM'],
        'Pred_Radioprotection_Pct': row['Pred_Radioprotection_Pct'],
        'Real_Vina_Affinity_kcal_mol': affinity,
        'Status': status
    })
    
    if (idx + 1) % 10 == 0 or (idx + 1) == total:
        print(f"-> Progresso: {idx + 1}/{total} compostos processados...")

df_vina_full = pd.DataFrame(docking_results)
df_vina_full.sort_values(by='Real_Vina_Affinity_kcal_mol', ascending=True, inplace=True)
df_vina_full['Vina_Rank'] = range(1, len(df_vina_full) + 1)

out_all_xlsx = os.path.join(dock_dir, 'autodock_vina_physical_results_100_leads.xlsx')
df_vina_full.to_excel(out_all_xlsx, index=False)

print("\n" + "=" * 75)
print("DOCKING FISICO DOS 100 CANDIDATOS FINALIZADO!")
print("=" * 75)
print(f"Média de Afinidade: {df_vina_full['Real_Vina_Affinity_kcal_mol'].mean():.2f} kcal/mol")
print(f"Melhor Afinidade:  {df_vina_full['Real_Vina_Affinity_kcal_mol'].min():.2f} kcal/mol ({df_vina_full.iloc[0]['Candidate_ID']})")
print(f"\nTop 10 Melhores Afinidades Fisicas:")
print(df_vina_full[['Vina_Rank', 'Candidate_ID', 'Scaffold', 'Functional_Strategy', 'Real_Vina_Affinity_kcal_mol']].head(10).to_string(index=False))
print(f"\nPlanilha consolidada dos 100 compostos salva em:\n{out_all_xlsx}")
print("=" * 75 + "\n")
