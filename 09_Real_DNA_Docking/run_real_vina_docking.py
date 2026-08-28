import os
import subprocess
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project')
modeling_dir = os.path.join(base_dir, '08_DNA_Molecular_Modeling')
dock_dir = os.path.join(base_dir, '09_Real_DNA_Docking')
tools_dir = os.path.join(base_dir, 'tools')
vina_exe = os.path.join(tools_dir, 'vina.exe')

os.makedirs(dock_dir, exist_ok=True)

# 1. Preparacao Canônica Estrita do Receptor B-DNA (1BNA)
dna_pdb = os.path.join(base_dir, '03_DNA_Docking_Aim1', '1BNA_clean_DNA.pdb')
receptor_pdbqt = os.path.join(dock_dir, '1bna_dna_clean.pdbqt')

with open(dna_pdb, 'r') as fpdb, open(receptor_pdbqt, 'w') as fpdbqt:
    for line in fpdb:
        if line.startswith(('ATOM', 'HETATM')):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21] if len(line) > 21 else 'A'
            res_seq = line[22:26]
            x_str, y_str, z_str = line[30:38], line[38:46], line[46:54]
            
            elem = line[76:78].strip() if len(line) >= 78 else ''
            if not elem:
                elem = atom_name[0]
                
            ad_type = elem
            charge = "+0.000"
            if elem == 'P':
                ad_type = 'P'
                charge = "+1.200"
            elif elem == 'O':
                ad_type = 'OA'
                if 'OP' in atom_name or 'O1P' in atom_name or 'O2P' in atom_name:
                    charge = "-0.600"
            elif elem == 'N':
                ad_type = 'NA'
            elif elem == 'C':
                ad_type = 'A' if ('DA' in res_name or 'DT' in res_name or 'DC' in res_name or 'DG' in res_name) and ('C' in atom_name or 'N' in atom_name) else 'C'
            elif elem == 'H':
                ad_type = 'HD'
                
            line_out = f"ATOM  {line[6:11]} {atom_name:<4s} {res_name:>3s} {chain_id}{res_seq}    {x_str}{y_str}{z_str}  1.00 20.00    {charge} {ad_type:<2s}\n"
            fpdbqt.write(line_out)

print("=" * 75)
print("DOCKING FISICO REAL (AUTODOCK VINA 1.2.7) NO B-DNA (1BNA)")
print("=" * 75)
print(f"Executavel Vina: {vina_exe}")
print(f"Receptor PDBQT:  {receptor_pdbqt}")

in_opt_xlsx = os.path.join(base_dir, '07_COMET_Transformer_Engine', 'comet_lead_optimization_100_candidates.xlsx')
df_opt = pd.read_excel(in_opt_xlsx)
top10 = df_opt.head(10).copy()

# Grid Box no Sulco Menor (Região Central AATT)
center_x, center_y, center_z = 14.7, 20.9, 8.8
size_x, size_y, size_z = 24.0, 24.0, 28.0

docking_results = []

for idx, row in top10.iterrows():
    cand_id = row['Optimized_ID']
    smi = row['SMILES']
    lig_pdbqt = os.path.join(dock_dir, f"{cand_id}.pdbqt")
    out_pdbqt = os.path.join(dock_dir, f"{cand_id}_docked.pdbqt")
    out_log = os.path.join(dock_dir, f"{cand_id}_vina.log")
    
    # 2. Geração do Ligante PDBQT Canônico com Cargas de Gasteiger
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
        
    print(f"\n-> Rodando AutoDock Vina para: {cand_id}...")
    
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
                
    if affinity is not None:
        print(f"   Score Vina REAL: {affinity} kcal/mol")
        status = "Sucesso"
    else:
        err_snippet = res.stderr.strip().split('\n')[0] if res.stderr else "Erro desconhecido"
        print(f"   Falha: {err_snippet}")
        status = f"Falha: {err_snippet}"
        
    docking_results.append({
        'Opt_Rank': row['Opt_Rank'],
        'Candidate_ID': cand_id,
        'Scaffold': row['Scaffold'],
        'Functional_Strategy': row['Optimization_Strategy'],
        'Pred_Kd_uM': row['Pred_DNA_Binding_Kd_uM'],
        'Real_Vina_Affinity_kcal_mol': affinity,
        'Status': status
    })

df_vina = pd.DataFrame(docking_results)
out_vina_xlsx = os.path.join(dock_dir, 'autodock_vina_physical_results.xlsx')
df_vina.to_excel(out_vina_xlsx, index=False)

print("\n" + "=" * 75)
print("TABELA CONSOLIDADA: SCORES FISICOS REAIS DE DOCKING (AUTODOCK VINA 1.2.7):")
print("=" * 75)
print(df_vina[['Opt_Rank', 'Candidate_ID', 'Scaffold', 'Real_Vina_Affinity_kcal_mol', 'Status']].to_string(index=False))
print(f"\nResultados salvos em: {out_vina_xlsx}")
print("=" * 75 + "\n")
