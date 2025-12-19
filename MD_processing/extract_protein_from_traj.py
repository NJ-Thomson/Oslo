"""
GPCR extraction from MD trajectories.
Supports both CLI arguments (for SLURM) and interactive mode.
"""

import MDAnalysis as mda
import numpy as np
import subprocess
import shutil
import tempfile
import os
import argparse


def detect_chains(protein):
    """Detect protein chains based on residue numbering."""
    resids = protein.residues.resids
    diffs = np.diff(resids)
    breaks = np.where((diffs > 30) | (diffs < 0))[0]
    
    if len(breaks) == 0:
        chains = [(0, len(resids) - 1)]
    else:
        boundaries = [0] + list(breaks + 1) + [len(resids)]
        chains = [(boundaries[i], boundaries[i + 1] - 1) for i in range(len(boundaries) - 1)]
    
    return chains


def extract_gpcr(topology, trajectory=None, chain=None, selection=None,
                 out_struct='gpcr_only.gro', out_traj='gpcr_only.xtc',
                 interactive=False):
    """
    Extract a protein chain from an MD trajectory.
    
    Parameters
    ----------
    topology : str
        Topology file (.gro/.pdb/.tpr)
    trajectory : str, optional
        Trajectory file (.xtc/.trr)
    chain : int, optional
        Chain number (1-indexed)
    selection : str, optional
        Custom MDAnalysis selection string
    out_struct : str
        Output structure filename
    out_traj : str
        Output trajectory filename
    interactive : bool
        Prompt for missing arguments
    """
    
    # Load system
    if trajectory:
        u = mda.Universe(topology, trajectory)
        print(f"Loaded: {u.atoms.n_atoms} atoms, {u.trajectory.n_frames} frames")
    else:
        u = mda.Universe(topology)
        print(f"Loaded: {u.atoms.n_atoms} atoms")
    
    protein = u.select_atoms('protein')
    print(f"Protein: {protein.n_residues} residues, {protein.n_atoms} atoms")
    
    # Detect chains
    chains = detect_chains(protein)
    
    print(f"\nFound {len(chains)} protein chain(s):")
    for i, (start_idx, end_idx) in enumerate(chains):
        start_resid = protein.residues[start_idx].resid
        end_resid = protein.residues[end_idx].resid
        n_res = end_idx - start_idx + 1
        print(f"  [{i + 1}] resid {start_resid}-{end_resid} ({n_res} residues)")
    
    # Determine selection
    if selection:
        gpcr = u.select_atoms(selection)
    elif chain:
        if 1 <= chain <= len(chains):
            start_idx, end_idx = chains[chain - 1]
            gpcr = protein.residues[start_idx:end_idx + 1].atoms
        else:
            raise ValueError(f"Chain {chain} not found. Available: 1-{len(chains)}")
    elif interactive:
        choice = input("\nSelect chain number (or enter custom selection): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(chains):
            start_idx, end_idx = chains[int(choice) - 1]
            gpcr = protein.residues[start_idx:end_idx + 1].atoms
        else:
            gpcr = u.select_atoms(choice)
    else:
        raise ValueError("Must specify --chain or --selection")
    
    print(f"\nSelected: {gpcr.n_residues} residues, {gpcr.n_atoms} atoms")
    
    # Write structure
    gpcr.write(out_struct)
    print(f"Wrote: {out_struct}")
    
    # Write trajectory
    if trajectory:
        gmx = shutil.which('gmx')
        
        if gmx:
            print("Using GROMACS for fast trajectory extraction...")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ndx', delete=False) as f:
                f.write("[ GPCR ]\n")
                indices = gpcr.atoms.indices + 1
                for i in range(0, len(indices), 15):
                    f.write(" ".join(str(idx) for idx in indices[i:i+15]) + "\n")
                ndx_file = f.name
            
            try:
                subprocess.run([
                    gmx, 'trjconv',
                    '-f', trajectory,
                    '-s', topology,
                    '-n', ndx_file,
                    '-o', out_traj
                ], input=b'0\n', check=True)
                print(f"Wrote: {out_traj}")
            finally:
                os.unlink(ndx_file)
        
        else:
            print("GROMACS not found, using MDAnalysis...")
            with mda.Writer(out_traj, gpcr.n_atoms) as W:
                for ts in u.trajectory:
                    W.write(gpcr)
            print(f"Wrote: {out_traj}")
    
    print("\nDone!")
    return u, gpcr


def main():
    parser = argparse.ArgumentParser(
        description='Extract GPCR/protein chain from MD trajectory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  %(prog)s -i
  
  # CLI mode for SLURM
  %(prog)s -f system.gro -x md.xtc -c 1 -os gpcr.gro -ot gpcr.xtc
  
  # Custom selection
  %(prog)s -f system.gro -x md.xtc -s "protein and resid 13:318"
        """
    )
    
    parser.add_argument('-f', '--topology', help='Topology file (.gro/.pdb/.tpr)')
    parser.add_argument('-x', '--trajectory', help='Trajectory file (.xtc/.trr)')
    parser.add_argument('-c', '--chain', type=int, help='Chain number (1-indexed)')
    parser.add_argument('-s', '--selection', help='Custom MDAnalysis selection')
    parser.add_argument('-os', '--out-struct', default='gpcr_only.gro', help='Output structure')
    parser.add_argument('-ot', '--out-traj', default='gpcr_only.xtc', help='Output trajectory')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Full interactive mode
    if args.interactive and not args.topology:
        args.topology = input("Topology file (.gro/.pdb/.tpr): ").strip()
        args.trajectory = input("Trajectory file (.xtc/.trr) or Enter for none: ").strip() or None
    
    if not args.topology:
        parser.error("Topology required. Use -f or -i for interactive mode.")
    
    extract_gpcr(
        topology=args.topology,
        trajectory=args.trajectory,
        chain=args.chain,
        selection=args.selection,
        out_struct=args.out_struct,
        out_traj=args.out_traj,
        interactive=args.interactive
    )


if __name__ == '__main__':
    main()