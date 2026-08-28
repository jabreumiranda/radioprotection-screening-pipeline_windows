import os
import glob
import subprocess
import pandas as pd

base_dir = os.path.expandvars(r'%USERPROFILE%\Radioprotection_Pipeline_Project')
orca_dir = os.path.join(base_dir, '06_Quantum_DFT_ORCA')
out_xlsx = os.path.join(orca_dir, 'dft_electronic_properties.xlsx')
orca_bin = r'C:\ORCA_6.1.1\orca.exe'

# Limpeza de logs corrompidos anteriores
for old_out in glob.glob(os.path.join(orca_dir, "*.out")):
    try:
        os.remove(old_out)
    except Exception:
        pass

inp_files = glob.glob(os.path.join(orca_dir, "*_neutral_opt.inp"))
print(f"Iniciando calculos DFT para {len(inp_files)} moleculas via ORCA 6.1.1...\n")

results = []

for inp_path in inp_files:
    file_name = os.path.basename(inp_path)
    cid = file_name.replace("_neutral_opt.inp", "")
    out_path = inp_path.replace(".inp", ".out")
    
    print(f"-> [ORCA DFT] Otimizando e calculando orbitais: {cid}...")
    
    cmd = f'"{orca_bin}" "{inp_path}"'
    with open(out_path, "w", encoding="utf-8") as out_f:
        subprocess.run(cmd, shell=True, stdout=out_f, stderr=subprocess.STDOUT)
    
    final_energy_hartree = None
    homo_ev = None
    lumo_ev = None
    converged = False
    
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
            for line in lines:
                if "FINAL SINGLE POINT ENERGY" in line:
                    parts = line.split()
                    try:
                        final_energy_hartree = float(parts[-1])
                    except ValueError:
                        pass
                if "THE OPTIMIZATION HAS CONVERGED" in line or "HURRAY" in line:
                    converged = True
            
            for i, line in enumerate(lines):
                if "ORBITAL ENERGIES" in line:
                    for sub_line in lines[i+4:i+350]:
                        if len(sub_line.strip()) == 0 or ("---" in sub_line and sub_line.strip() != "------------------"):
                            break
                        parts = sub_line.split()
                        if len(parts) >= 4:
                            try:
                                occ = float(parts[1])
                                ev = float(parts[3])
                                if occ > 0.0:
                                    homo_ev = ev
                                elif occ == 0.0 and lumo_ev is None:
                                    lumo_ev = ev
                            except ValueError:
                                continue

    gap_ev = round(lumo_ev - homo_ev, 3) if (homo_ev is not None and lumo_ev is not None) else None
    eta = round(gap_ev / 2.0, 3) if gap_ev else None
    mu = round((homo_ev + lumo_ev) / 2.0, 3) if (homo_ev and lumo_ev) else None
    omega = round((mu**2) / (2.0 * eta), 3) if (eta and mu and eta > 0) else None
    
    status_str = "Converged" if converged else ("Finished" if homo_ev is not None else "Failed/Check Output")
    print(f"   Status: {status_str} | HOMO: {homo_ev} eV | LUMO: {lumo_ev} eV | GAP: {gap_ev} eV\n")
    
    results.append({
        'ChEMBL_ID': cid,
        'DFT_Status': status_str,
        'E_SCF_Hartree': final_energy_hartree,
        'E_HOMO_eV': homo_ev,
        'E_LUMO_eV': lumo_ev,
        'GAP_eV': gap_ev,
        'Chemical_Hardness_eta_eV': eta,
        'Electrophilicity_omega_eV': omega
    })

df_dft = pd.DataFrame(results)
df_dft = df_dft.sort_values(by='E_HOMO_eV', ascending=False)
df_dft.to_excel(out_xlsx, index=False)

print("==================================================================")
print("PROPRIEDADES ELETRONICAS DFT CONCLUIDAS COM SUCESSO!")
print("==================================================================")
print(f"Planilha exportada: {out_xlsx}\n")
print(df_dft[['ChEMBL_ID', 'DFT_Status', 'E_HOMO_eV', 'E_LUMO_eV', 'GAP_eV']])
