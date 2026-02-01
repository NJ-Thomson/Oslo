#!/usr/bin/env python3
"""
setup_rbfe_per_receptor.py

Set up RBFE calculations with independent pose alignment per receptor.

This script ensures each receptor gets its own optimal pose pair by:
1. Running compare_warhead_poses.py for the specific receptor/inhibitor combo
2. Creating complexes with the aligned poses
3. Organizing into the proper directory structure

The key insight: poses should be aligned WITHIN each receptor's binding site,
not shared across different receptors. This ensures proper spatial overlap
for the alchemical transformation.

Output structure:
    {output_dir}/
        {receptor}/
            {inhibitor}/
                complexes/
                    complex_acry_best.pdb
                    complex_chlo_to_acry.pdb   # CHLO aligned to acry's best pose
                    complex_acry_to_chlo.pdb   # ACRY aligned to chlo's best pose
                    complex_chlo_best.pdb
                    ligand_*.sdf
                rbfe_prep_metadata.json

Usage:
    # Single inhibitor/receptor
    python setup_rbfe_per_receptor.py \\
        --receptor 5ACB \\
        --inhibitor Inhib_32 \\
        --docking_dir ../Outputs/claw_cov/03_docking \\
        --receptor_dir ../Outputs/claw_cov/02_receptors \\
        --output_dir ../Outputs/claw_cov/04_rbfe_prep

    # All inhibitors for one receptor
    python setup_rbfe_per_receptor.py \\
        --receptor 5ACB \\
        --all_inhibitors \\
        --docking_dir ../Outputs/claw_cov/03_docking \\
        --receptor_dir ../Outputs/claw_cov/02_receptors \\
        --output_dir ../Outputs/claw_cov/04_rbfe_prep
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
except ImportError:
    print("ERROR: RDKit required. Install via conda/pip.")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

RECEPTORS = {
    '5ACB': {'cys_resid': 1039, 'cys_chain': 'A', 'name': 'CDK12'},
    '7NXJ': {'cys_resid': 1017, 'cys_chain': 'A', 'name': 'CDK13'},
}

INHIBITORS = ['Inhib_32', 'Inhib_36', 'Inhib_78', 'Inhib_86']


# ============================================================================
# Pose Selection Functions (from compare_warhead_poses.py)
# ============================================================================

def load_poses(sdf_path: Path) -> List[Chem.Mol]:
    """Load all poses from SDF file."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    poses = [mol for mol in suppl if mol is not None]
    return poses


def get_vina_affinity(mol: Chem.Mol) -> float:
    """Get Vina affinity from molecule properties."""
    for prop in ['minimizedAffinity', 'affinity']:
        if mol.HasProp(prop):
            try:
                return float(mol.GetProp(prop))
            except:
                pass
    return 0.0


def find_mcs_mapping(mol1: Chem.Mol, mol2: Chem.Mol) -> List[Tuple[int, int]]:
    """Find Maximum Common Substructure atom mapping."""
    mcs = rdFMCS.FindMCS(
        [mol1, mol2],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=False,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        timeout=10
    )

    if mcs.canceled or mcs.numAtoms == 0:
        return []

    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    if mcs_mol is None:
        return []

    matches1 = mol1.GetSubstructMatches(mcs_mol)
    matches2 = mol2.GetSubstructMatches(mcs_mol)

    if not matches1 or not matches2:
        return []

    return list(zip(matches1[0], matches2[0]))


def calculate_rmsd(mol1: Chem.Mol, mol2: Chem.Mol, 
                   atom_mapping: List[Tuple[int, int]]) -> float:
    """Calculate RMSD between molecules using atom mapping."""
    if not atom_mapping:
        return float('inf')

    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()

    sq_dists = []
    for idx1, idx2 in atom_mapping:
        p1 = conf1.GetAtomPosition(idx1)
        p2 = conf2.GetAtomPosition(idx2)
        sq_dists.append((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    return np.sqrt(np.mean(sq_dists))


def select_aligned_poses(acry_poses: List[Chem.Mol], 
                         chlo_poses: List[Chem.Mol],
                         verbose: bool = True) -> Dict:
    """
    Select best-aligned poses for RBFE.
    
    Returns:
        Dictionary with selected poses and metadata
    """
    if verbose:
        print(f"    Loaded {len(acry_poses)} acry poses, {len(chlo_poses)} chlo poses")
    
    # Find MCS mapping (compute once)
    mcs_mapping = find_mcs_mapping(acry_poses[0], chlo_poses[0])
    
    if verbose:
        if mcs_mapping:
            print(f"    MCS: {len(mcs_mapping)} atoms")
        else:
            print(f"    WARNING: MCS failed")

    # Get best poses by Vina affinity
    acry_affinities = [get_vina_affinity(m) for m in acry_poses]
    chlo_affinities = [get_vina_affinity(m) for m in chlo_poses]

    best_acry_idx = int(np.argmin(acry_affinities))
    best_chlo_idx = int(np.argmin(chlo_affinities))

    if verbose:
        print(f"    Best ACRY: pose {best_acry_idx} ({acry_affinities[best_acry_idx]:.2f} kcal/mol)")
        print(f"    Best CHLO: pose {best_chlo_idx} ({chlo_affinities[best_chlo_idx]:.2f} kcal/mol)")

    # Find chlo pose closest to best acry
    rmsds_to_acry = []
    for chlo_mol in chlo_poses:
        rmsd = calculate_rmsd(acry_poses[best_acry_idx], chlo_mol, mcs_mapping)
        rmsds_to_acry.append(rmsd)
    chlo_to_acry_idx = int(np.argmin(rmsds_to_acry))

    # Find acry pose closest to best chlo
    rmsds_to_chlo = []
    for acry_mol in acry_poses:
        rmsd = calculate_rmsd(acry_mol, chlo_poses[best_chlo_idx], mcs_mapping)
        rmsds_to_chlo.append(rmsd)
    acry_to_chlo_idx = int(np.argmin(rmsds_to_chlo))

    if verbose:
        print(f"    CHLO→ACRY: pose {chlo_to_acry_idx} (RMSD={rmsds_to_acry[chlo_to_acry_idx]:.2f} Å)")
        print(f"    ACRY→CHLO: pose {acry_to_chlo_idx} (RMSD={rmsds_to_chlo[acry_to_chlo_idx]:.2f} Å)")

    return {
        'acry_best': {
            'idx': best_acry_idx,
            'mol': acry_poses[best_acry_idx],
            'affinity': acry_affinities[best_acry_idx]
        },
        'chlo_best': {
            'idx': best_chlo_idx,
            'mol': chlo_poses[best_chlo_idx],
            'affinity': chlo_affinities[best_chlo_idx]
        },
        'chlo_to_acry': {
            'idx': chlo_to_acry_idx,
            'mol': chlo_poses[chlo_to_acry_idx],
            'affinity': chlo_affinities[chlo_to_acry_idx],
            'rmsd': rmsds_to_acry[chlo_to_acry_idx]
        },
        'acry_to_chlo': {
            'idx': acry_to_chlo_idx,
            'mol': acry_poses[acry_to_chlo_idx],
            'affinity': acry_affinities[acry_to_chlo_idx],
            'rmsd': rmsds_to_chlo[acry_to_chlo_idx]
        },
        'mcs_n_atoms': len(mcs_mapping) if mcs_mapping else 0
    }


# ============================================================================
# Complex Assembly
# ============================================================================

def write_ligand_pdb(mol: Chem.Mol, output_path: Path, resname: str = "LIG"):
    """Write ligand to PDB file."""
    for atom in mol.GetAtoms():
        mi = Chem.AtomPDBResidueInfo()
        mi.SetResidueName(resname)
        mi.SetResidueNumber(1)
        mi.SetChainId("A")
        mi.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(mi)

    Chem.MolToPDBFile(mol, str(output_path))


def create_complex_pdb(receptor_pdb: Path, ligand_mol: Chem.Mol,
                       output_path: Path, lig_resname: str = "LIG"):
    """Create complex PDB by combining receptor and ligand."""
    # Write ligand to temp file
    lig_tmp = output_path.parent / f"_tmp_lig_{output_path.stem}.pdb"
    write_ligand_pdb(ligand_mol, lig_tmp, lig_resname)

    # Read receptor
    with open(receptor_pdb) as f:
        receptor_lines = [l for l in f if l.startswith(('ATOM', 'HETATM', 'TER'))]

    # Read ligand
    with open(lig_tmp) as f:
        ligand_lines = [l for l in f if l.startswith(('ATOM', 'HETATM'))]

    # Combine
    with open(output_path, 'w') as f:
        for line in receptor_lines:
            if not line.startswith('TER'):
                f.write(line)
        f.write("TER\n")
        for line in ligand_lines:
            if line.startswith('ATOM'):
                line = 'HETATM' + line[6:]
            f.write(line)
        f.write("END\n")

    # Cleanup
    lig_tmp.unlink()


# ============================================================================
# Main Processing
# ============================================================================

def process_inhibitor(receptor: str, inhibitor: str, 
                      docking_dir: Path, receptor_dir: Path,
                      output_dir: Path, verbose: bool = True) -> bool:
    """
    Process a single inhibitor for a receptor.
    
    Creates the 04_rbfe_prep structure with aligned poses.
    """
    if verbose:
        print(f"\n  Processing {receptor}/{inhibitor}...")
    
    rec_config = RECEPTORS.get(receptor)
    if not rec_config:
        print(f"    ERROR: Unknown receptor {receptor}")
        return False
    
    # Input paths
    acry_sdf = docking_dir / receptor / inhibitor / "acry" / "docked_poses.sdf"
    chlo_sdf = docking_dir / receptor / inhibitor / "chlo" / "docked_poses.sdf"
    receptor_pdb = receptor_dir / f"{receptor}_predock.pdb"
    
    # Validate inputs
    for p, name in [(acry_sdf, 'acry poses'), (chlo_sdf, 'chlo poses'), 
                    (receptor_pdb, 'receptor')]:
        if not p.exists():
            print(f"    ERROR: {name} not found: {p}")
            return False
    
    # Output directory
    out_dir = output_dir / receptor / inhibitor
    complexes_dir = out_dir / "complexes"
    complexes_dir.mkdir(parents=True, exist_ok=True)
    
    # Load poses
    acry_poses = load_poses(acry_sdf)
    chlo_poses = load_poses(chlo_sdf)
    
    if not acry_poses or not chlo_poses:
        print(f"    ERROR: No poses loaded")
        return False
    
    # Select aligned poses
    selected = select_aligned_poses(acry_poses, chlo_poses, verbose=verbose)
    
    # Create complex PDBs and save ligand SDFs
    systems = [
        ('acry_best', selected['acry_best']['mol']),
        ('chlo_to_acry', selected['chlo_to_acry']['mol']),
        ('acry_to_chlo', selected['acry_to_chlo']['mol']),
        ('chlo_best', selected['chlo_best']['mol']),
    ]
    
    complex_paths = {}
    for name, mol in systems:
        # Create complex PDB
        out_pdb = complexes_dir / f"complex_{name}.pdb"
        create_complex_pdb(receptor_pdb, mol, out_pdb, "LIG")
        complex_paths[name] = str(out_pdb)
        
        # Save ligand SDF
        out_sdf = complexes_dir / f"ligand_{name}.sdf"
        writer = Chem.SDWriter(str(out_sdf))
        writer.write(mol)
        writer.close()
    
    # Save metadata
    metadata = {
        'receptor': str(receptor_pdb),
        'acry_sdf': str(acry_sdf),
        'chlo_sdf': str(chlo_sdf),
        'cys_resid': rec_config['cys_resid'],
        'mcs_atoms': selected['mcs_n_atoms'],
        'selections': {
            'acry_best': {
                'pose_idx': selected['acry_best']['idx'],
                'affinity': selected['acry_best']['affinity'],
                'complex_pdb': complex_paths['acry_best']
            },
            'chlo_best': {
                'pose_idx': selected['chlo_best']['idx'],
                'affinity': selected['chlo_best']['affinity'],
                'complex_pdb': complex_paths['chlo_best']
            },
            'chlo_to_acry': {
                'pose_idx': selected['chlo_to_acry']['idx'],
                'affinity': selected['chlo_to_acry']['affinity'],
                'rmsd_to_acry_best': selected['chlo_to_acry']['rmsd'],
                'complex_pdb': complex_paths['chlo_to_acry']
            },
            'acry_to_chlo': {
                'pose_idx': selected['acry_to_chlo']['idx'],
                'affinity': selected['acry_to_chlo']['affinity'],
                'rmsd_to_chlo_best': selected['acry_to_chlo']['rmsd'],
                'complex_pdb': complex_paths['acry_to_chlo']
            }
        },
        'rbfe_pairs': {
            'forward_from_acry': {
                'stateA': 'acry_best',
                'stateB': 'chlo_to_acry',
                'description': 'ACRY→CHLO starting from best acry pose'
            },
            'forward_from_chlo': {
                'stateA': 'acry_to_chlo',
                'stateB': 'chlo_best',
                'description': 'ACRY→CHLO starting from best chlo pose'
            }
        }
    }
    
    with open(out_dir / "rbfe_prep_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"    ✓ Created {len(systems)} complexes in {complexes_dir}")
        print(f"    ✓ ACRY-aligned: acry[{selected['acry_best']['idx']}] ↔ chlo[{selected['chlo_to_acry']['idx']}] (RMSD={selected['chlo_to_acry']['rmsd']:.2f} Å)")
        print(f"    ✓ CHLO-aligned: acry[{selected['acry_to_chlo']['idx']}] ↔ chlo[{selected['chlo_best']['idx']}] (RMSD={selected['acry_to_chlo']['rmsd']:.2f} Å)")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Set up RBFE with per-receptor pose alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--receptor', required=True, choices=list(RECEPTORS.keys()),
                        help='Receptor PDB code (5ACB or 7NXJ)')
    parser.add_argument('--inhibitor', help='Specific inhibitor (e.g., Inhib_32)')
    parser.add_argument('--all_inhibitors', action='store_true',
                        help='Process all inhibitors for the receptor')
    parser.add_argument('--docking_dir', required=True,
                        help='Directory with docking results')
    parser.add_argument('--receptor_dir', required=True,
                        help='Directory with prepared receptor PDBs')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for RBFE prep')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    args = parser.parse_args()
    
    docking_dir = Path(args.docking_dir)
    receptor_dir = Path(args.receptor_dir)
    output_dir = Path(args.output_dir)
    
    # Validate directories
    if not docking_dir.exists():
        print(f"ERROR: Docking directory not found: {docking_dir}")
        return 1
    if not receptor_dir.exists():
        print(f"ERROR: Receptor directory not found: {receptor_dir}")
        return 1
    
    # Determine inhibitors to process
    if args.all_inhibitors:
        inhibitors = INHIBITORS
    elif args.inhibitor:
        inhibitors = [args.inhibitor]
    else:
        print("ERROR: Specify --inhibitor or --all_inhibitors")
        return 1
    
    verbose = not args.quiet
    
    print("=" * 60)
    print("RBFE POSE SETUP (PER-RECEPTOR ALIGNMENT)")
    print("=" * 60)
    print(f"\nReceptor: {args.receptor} ({RECEPTORS[args.receptor]['name']})")
    print(f"Inhibitors: {', '.join(inhibitors)}")
    print(f"Output: {output_dir}")
    
    # Process each inhibitor
    results = {}
    for inhib in inhibitors:
        success = process_inhibitor(
            args.receptor, inhib,
            docking_dir, receptor_dir, output_dir,
            verbose=verbose
        )
        results[inhib] = success
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    n_success = sum(results.values())
    n_total = len(results)
    
    for inhib, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {inhib}")
    
    print(f"\nProcessed {n_success}/{n_total} inhibitors successfully")
    
    if n_success < n_total:
        print("\nSome inhibitors failed. Check docking results exist.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
