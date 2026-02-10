import json
from typing import Dict, List, Optional, Union

import numpy as np

from abfe import template


def generate_approach_conf(out_path: str,
                           input_ligands_sdf_path: Union[str, None],
                           input_protein_pdb_path: Union[str, None],
                           input_cofactor_sdf_path: Union[str, None],
                           out_approach_path: str,
                           ligand_names: List[str],
                           num_replica: int,
                           python_bin:str,
                           build_system:bool,
                           small_mol_ff:str
                           ):
    ## Ugly implementation every defined variable is added to conf! :)
    conf_settings = {k: v for k, v in locals().items() if not k.startswith("__")}

    out_IO = open(out_path, "w")
    json.dump(conf_settings, out_IO, indent=4)


def generate_ligand_conf(out_path: str, code_path: str, run_path: str, input_data_path: str,
                         num_replica: int = 1, num_sim_threads: int = 7,
                         n_vdw_windows_complex: int = 21, n_rest_windows_complex: int = 11, n_coul_windows_complex: int = 11,
                         n_vdw_windows_ligand: int = 11, n_coul_windows_ligand: int = 11,
                         gmx_run_kernel_path: str = template.gmx_submit_kernels_path + "/def_cpu_job.sh",
                         gmx_cont_kernel_path: str = template.gmx_submit_kernels_path + "/def_cpu_job_cont.sh",
                         gmx_add_flag:str="",
                         ligand_config: Optional[Dict] = None):
    """Generate Snakemake configuration JSON for a single ligand/replica.

    Parameters
    ----------
    ligand_config : dict, optional
        Per-ligand overrides from YAML config. Supported keys:
          lambda_windows.complex_vdw   -> n_vdw_windows_complex
          lambda_windows.complex_coul  -> n_coul_windows_complex
          lambda_windows.complex_restraints -> n_rest_windows_complex
          lambda_windows.ligand_vdw    -> n_vdw_windows_ligand
          lambda_windows.ligand_coul   -> n_coul_windows_ligand
          production_nsteps            -> nsteps in prod MDP
          production_ns                -> converted to nsteps (ns * 250000 at 4fs dt)
    """
    # Apply per-ligand overrides from YAML config
    if ligand_config:
        lw = ligand_config.get("lambda_windows", {})
        if "complex_vdw" in lw:
            n_vdw_windows_complex = int(lw["complex_vdw"])
        if "complex_coul" in lw:
            n_coul_windows_complex = int(lw["complex_coul"])
        if "complex_restraints" in lw:
            n_rest_windows_complex = int(lw["complex_restraints"])
        if "ligand_vdw" in lw:
            n_vdw_windows_ligand = int(lw["ligand_vdw"])
        if "ligand_coul" in lw:
            n_coul_windows_ligand = int(lw["ligand_coul"])

    ## get all the window ids
    lam_vdw_complex_range = list(np.round(np.linspace(0, 1, n_vdw_windows_complex), 2))
    lam_coul_complex_range = list(np.round(np.linspace(0, 1, n_rest_windows_complex), 2))
    lam_rest_complex_range = list(np.round(np.linspace(0, 1, n_coul_windows_complex), 2))

    lam_vdw_ligand_range = list(np.round(np.linspace(0, 1, n_vdw_windows_ligand), 2))
    lam_coul_ligand_range = list(np.round(np.linspace(0, 1, n_coul_windows_ligand), 2))

    vdw_complex_windows = [f'vdw.{i}' for i in range(n_vdw_windows_complex)]
    rest_complex_windows = [f'restraints.{i}' for i in range(n_rest_windows_complex)]
    coul_complex_windows = [f'coul.{i}' for i in range(n_coul_windows_complex)]

    vdw_ligand_windows = [f'vdw.{i}' for i in range(n_vdw_windows_ligand)]
    coul_ligand_windows = [f'coul.{i}' for i in range(n_coul_windows_ligand)]

    complex_windows = vdw_complex_windows + rest_complex_windows + coul_complex_windows
    ligand_windows = vdw_ligand_windows + coul_ligand_windows

    # Production nsteps override (default 5000000 = 10 ns at 4 fs dt)
    production_nsteps = 5000000
    if ligand_config:
        if "production_nsteps" in ligand_config:
            production_nsteps = int(ligand_config["production_nsteps"])
        elif "production_ns" in ligand_config:
            production_nsteps = int(float(ligand_config["production_ns"]) * 250000)

    # Parallel:
    num_sim_threads = num_sim_threads

    run_num = num_replica
    run_path = run_path
    num_retries = 3

    gmx_run_kernel_path = gmx_run_kernel_path
    gmx_cont_kernel_path = gmx_cont_kernel_path
    gmx_add_flag = gmx_add_flag

    conf_settings = {k: v for k, v in locals().items()
                     if not k.startswith("__") and k != "ligand_config" and k != "lw"}

    out_IO = open(out_path, "w")
    json.dump(conf_settings, out_IO, indent=4)
