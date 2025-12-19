"""
Interactive protein extraction from MD trajectories.
"""

import MDAnalysis as mda
import numpy as np


def extract_gpcr_interactive():
    """Interactively extract a protein chain from an MD trajectory."""
    
    # Get input files
    topology = input("Topology file (.gro/.pdb/.tpr): ").strip()
    trajectory = input("Trajectory file (.xtc/.trr) or press Enter for none: ").strip()
    
    # Load
    if trajectory:
        u = mda.Universe(topology, trajectory)
        print(f"\nLoaded: {u.atoms.n_atoms} atoms, {u.trajectory.n_frames} frames")
    else:
        u = mda.Universe(topology)
        print(f"\nLoaded: {u.atoms.n_atoms} atoms")
    
    protein = u.select_atoms('protein')
    print(f"Protein: {protein.n_residues} residues, {protein.n_atoms} atoms")
    
    # Detect chains - look for gaps OR resets in residue numbering
    resids = protein.residues.resids
    diffs = np.diff(resids)
    
    # Chain break = big positive gap OR any negative jump (reset)
    breaks = np.where((diffs > 30) | (diffs < 0))[0]
    
    if len(breaks) == 0:
        chains = [(0, len(resids) - 1)]  # indices into residues
    else:
        boundaries = [0] + list(breaks + 1) + [len(resids)]
        chains = [(boundaries[i], boundaries[i + 1] - 1) for i in range(len(boundaries) - 1)]
    
    # Display chains
    print(f"\nFound {len(chains)} protein chain(s):")
    for i, (start_idx, end_idx) in enumerate(chains):
        start_resid = protein.residues[start_idx].resid
        end_resid = protein.residues[end_idx].resid
        n_res = end_idx - start_idx + 1
        print(f"  [{i + 1}] resid {start_resid}-{end_resid} ({n_res} residues)")
    
    # Select
    choice = input("\nSelect chain number (or enter custom selection): ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(chains):
        start_idx, end_idx = chains[int(choice) - 1]
        # Use residue indices to get the actual atoms
        selected_residues = protein.residues[start_idx:end_idx + 1]
        gpcr = selected_residues.atoms
    else:
        gpcr = u.select_atoms(choice)
    
    print(f"\nSelected: {gpcr.n_residues} residues, {gpcr.n_atoms} atoms")
    
    # Output names
    default_struct = 'gpcr_only.gro'
    default_traj = 'gpcr_only.xtc'
    
    out_struct = input(f"Output structure [{default_struct}]: ").strip() or default_struct
    gpcr.write(out_struct)
    print(f"Wrote: {out_struct}")
    
    if trajectory:
        out_traj = input(f"Output trajectory [{default_traj}]: ").strip() or default_traj
        with mda.Writer(out_traj, gpcr.n_atoms) as W:
            for ts in u.trajectory:
                W.write(gpcr)
        print(f"Wrote: {out_traj}")
    
    print("\nDone!")
    return u, gpcr


if __name__ == '__main__':
    extract_gpcr_interactive()