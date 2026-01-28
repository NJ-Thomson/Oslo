#!/usr/bin/env python3
"""
Simple AutoDock Vina Docking Script

Docks a single ligand to a single receptor using a reference ligand for binding site.

Usage:
    python dock_ligand.py --ligand LIGAND.sdf --receptor RECEPTOR_predock.pdb \
                          --reference REF.pdb --ligname LIGNAME

Examples:
    # Dock a single ligand
    python dock_ligand.py --ligand conformers/Inhib_45.sdf \
                          --receptor 4CXA_predock.pdb \
                          --reference 4CXA_reference.pdb \
                          --ligname LIG

    # With custom output name (path allowed)
    python dock_ligand.py --ligand conformers/Inhib_45.sdf \
                          --receptor 4CXA_predock.pdb \
                          --reference 4CXA_reference.pdb \
                          --ligname LIG \
                          --output noncov_docking_results/4CXA_Inhib_45
"""

import os
import sys
import argparse
import subprocess
import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def run_command(cmd, description, verbose=True):
    """Execute a shell command and handle errors."""
    if verbose:
        print(f"  {description}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {description} failed!")
        print(f"  STDERR: {result.stderr}")
        return False
    if verbose:
        print(f"  ✓ {description} complete")
    return True


def ensure_parent_dir(pathlike):
    """Ensure the parent directory of a path exists."""
    p = Path(pathlike)
    parent = p.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def extract_binding_site(reference_pdb, ligand_name=None, padding=10.0, max_size=30.0):
    """
    Extract binding site coordinates from reference PDB with ligand.

    Args:
        reference_pdb: Path to reference PDB with co-crystallized ligand
        ligand_name: Residue name of ligand (e.g., 'LIG', '9GF')
        padding: Buffer around ligand (Angstroms)
        max_size: Maximum box dimension (Angstroms)

    Returns:
        dict with center_x/y/z and size_x/y/z
    """
    exclude_list = {
        'HOH', 'WAT', 'TIP', 'TIP3', 'NA', 'CL', 'K', 'MG',
        'CA', 'ZN', 'FE', 'MN', 'SO4', 'PO4', 'GOL', 'EDO',
        'ACE', 'NME', 'NH2'
    }

    print(f"\nExtracting binding site from: {reference_pdb}")
    if ligand_name:
        print(f"  Looking for ligand: {ligand_name}")

    with open(reference_pdb, 'r') as f:
        lines = f.readlines()

    ligand_coords = []
    ligand_residues = {}

    for line in lines:
        if line.startswith("HETATM"):
            resname = line[17:20].strip()

            if ligand_name and resname != ligand_name:
                continue
            if resname in exclude_list:
                continue

            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ligand_coords.append([x, y, z])

                if resname not in ligand_residues:
                    ligand_residues[resname] = 0
                ligand_residues[resname] += 1
            except Exception:
                pass

    if not ligand_coords:
        print(f"  ERROR: No ligand found in reference!")
        if ligand_name:
            print(f"  Could not find '{ligand_name}'")
        print(f"  Available HETATM residues:")
        available = set()
        for line in lines:
            if line.startswith("HETATM"):
                resname = line[17:20].strip()
                if resname not in exclude_list:
                    available.add(resname)
        print(f"    {', '.join(sorted(available)) if available else 'None'}")
        sys.exit(1)

    print(f"  ✓ Found: {', '.join(f'{k} ({v} atoms)' for k, v in ligand_residues.items())}")

    ligand_coords = np.array(ligand_coords)
    center = ligand_coords.mean(axis=0)

    # Calculate box size with padding
    ligand_min = ligand_coords.min(axis=0)
    ligand_max = ligand_coords.max(axis=0)
    ligand_span = ligand_max - ligand_min
    box_size = ligand_span + padding
    box_size = np.minimum(box_size, max_size)  # Cap at max_size

    print(f"  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"  Box size: ({box_size[0]:.1f}, {box_size[1]:.1f}, {box_size[2]:.1f}) Å")

    return {
        'center_x': float(center[0]),
        'center_y': float(center[1]),
        'center_z': float(center[2]),
        'size_x': float(box_size[0]),
        'size_y': float(box_size[1]),
        'size_z': float(box_size[2])
    }


def prepare_ligand(sdf_file, output_pdbqt):
    """Prepare ligand using Meeko."""
    ensure_parent_dir(output_pdbqt)
    cmd = [
        "mk_prepare_ligand.py",
        "-i", str(sdf_file),
        "-o", str(output_pdbqt)
    ]
    return run_command(cmd, "Preparing ligand with Meeko")


def prepare_receptor(pdb_file, output_pdbqt):
    """Prepare receptor using Open Babel."""
    pdb_file = str(pdb_file)
    output_pdbqt = str(output_pdbqt)

    ensure_parent_dir(output_pdbqt)

    # Add hydrogens to a temporary PDB next to the output
    temp_pdb = Path(output_pdbqt).with_suffix('').as_posix() + "_temp_H.pdb"
    cmd = ['obabel', pdb_file, '-O', temp_pdb, '-h']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR adding hydrogens: {result.stderr}")
        return False

    # Convert to PDBQT
    cmd = ['obabel', temp_pdb, '-O', output_pdbqt, '-xr']
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Clean up temp file
    try:
        Path(temp_pdb).unlink(missing_ok=True)
    except Exception:
        pass

    if result.returncode != 0:
        print(f"  ERROR creating PDBQT: {result.stderr}")
        return False

    print(f"  ✓ Prepared receptor: {output_pdbqt}")
    return True


def run_vina(receptor_pdbqt, ligand_pdbqt, output_pdbqt, binding_site,
             exhaustiveness=32, num_modes=20, energy_range=4.0):
    """Run AutoDock Vina docking."""
    ensure_parent_dir(output_pdbqt)
    cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--out", str(output_pdbqt),
        "--center_x", str(binding_site['center_x']),
        "--center_y", str(binding_site['center_y']),
        "--center_z", str(binding_site['center_z']),
        "--size_x", str(binding_site['size_x']),
        "--size_y", str(binding_site['size_y']),
        "--size_z", str(binding_site['size_z']),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--energy_range", str(energy_range)
    ]

    print(f"\nRunning Vina (exhaustiveness={exhaustiveness})...")
    print(f"  This may take a few minutes...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Vina failed!")
        print(f"  {result.stderr}")
        return False

    print(f"  ✓ Docking complete")
    return True


def parse_vina_results(output_pdbqt):
    """Parse docking results from Vina output PDBQT."""
    results = []
    try:
        with open(output_pdbqt, 'r') as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            affinity = float(parts[3])
                            results.append({
                                'mode': len(results) + 1,
                                'affinity': affinity
                            })
                        except Exception:
                            pass
    except Exception as e:
        print(f"  Warning: Could not parse results: {e}")
        return []
    return results


def export_docked_poses_meeko(vina_out_pdbqt, out_sdf, best_only=True):
    """
    Export Vina poses from PDBQT to SDF via Meeko, preserving atom order.
    best_only: if True, write the first (best) pose only; otherwise writes all poses.
    """
    ensure_parent_dir(out_sdf)
    cmd = [
        "mk_export.py",
        str(vina_out_pdbqt),
        "-s", str(out_sdf)
    ]
    if not best_only:
        cmd += ["--all_dlg_poses"]
    return run_command(cmd, "Exporting docked poses to SDF with Meeko")


def sdf_to_pdb(sdf_file, pdb_file, resname='LIG'):
    """Convert SDF to PDB using Open Babel."""
    cmd = ['obabel', str(sdf_file), '-O', str(pdb_file), '-h']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False

    # Fix residue name in output PDB
    lines = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                # Replace residue name (columns 18-20)
                line = line[:17] + f'{resname:>3}' + line[20:]
            lines.append(line)
    with open(pdb_file, 'w') as f:
        f.writelines(lines)
    return True


def create_complex_pdb(receptor_pdb, ligand_sdf, output_pdb, ligand_resname='LIG'):
    """
    Create complex PDB by combining receptor and docked ligand.
    """
    ensure_parent_dir(output_pdb)

    # Convert ligand SDF to PDB
    ligand_pdb = Path(output_pdb).parent / 'ligand_temp.pdb'
    if not sdf_to_pdb(ligand_sdf, ligand_pdb, ligand_resname):
        print("  Warning: Could not convert ligand to PDB")
        return False

    # Read receptor (exclude END/ENDMDL)
    receptor_lines = []
    with open(receptor_pdb) as f:
        for line in f:
            if not line.startswith(('END', 'ENDMDL')):
                receptor_lines.append(line)

    # Read ligand atoms
    ligand_lines = []
    with open(ligand_pdb) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                # Change ATOM to HETATM for ligand
                line = 'HETATM' + line[6:]
                ligand_lines.append(line)

    # Write combined complex
    with open(output_pdb, 'w') as f:
        f.writelines(receptor_lines)
        f.write('TER\n')
        f.writelines(ligand_lines)
        f.write('END\n')

    # Clean up temp file
    if ligand_pdb.exists():
        ligand_pdb.unlink()

    return True


# =============================================================================
# H-bond screening functions
# =============================================================================

HBOND_DIST_MAX = 3.5  # Angstroms
DONORS = {'N', 'O'}
ACCEPTORS = {'N', 'O', 'F'}


def parse_residue_spec(spec):
    """Parse residue spec like 'M816' into (resname, resnum)."""
    aa_map = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    spec = spec.strip().upper()
    match = re.match(r'([A-Z]+)(\d+)', spec)
    if match:
        res_name = match.group(1)
        res_num = int(match.group(2))
        if len(res_name) == 1 and res_name in aa_map:
            res_name = aa_map[res_name]
        return res_name, res_num
    return None, None


def parse_pdb_for_hbonds(pdb_file, residue_filter=None):
    """Parse PDB atoms for H-bond analysis."""
    atoms = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                try:
                    res_num = int(line[22:26])
                except ValueError:
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]

                if residue_filter and res_num not in residue_filter:
                    continue

                atoms.append({
                    'name': atom_name,
                    'resname': res_name,
                    'resnum': res_num,
                    'coords': np.array([x, y, z]),
                    'element': element,
                    'is_backbone': atom_name in ['N', 'CA', 'C', 'O', 'H', 'HN']
                })
    return atoms


def get_ligand_atoms_rdkit(mol):
    """Extract atom info from RDKit molecule."""
    conf = mol.GetConformer()
    atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        symbol = atom.GetSymbol()
        atoms.append({
            'idx': i,
            'symbol': symbol,
            'coords': np.array([pos.x, pos.y, pos.z]),
            'is_donor': symbol in DONORS and atom.GetTotalNumHs() > 0,
            'is_acceptor': symbol in ACCEPTORS
        })
    return atoms


def calculate_hbonds(ligand_atoms, receptor_atoms):
    """Calculate H-bonds between ligand and receptor."""
    hbonds = []
    for lig_atom in ligand_atoms:
        for rec_atom in receptor_atoms:
            dist = np.linalg.norm(lig_atom['coords'] - rec_atom['coords'])
            if dist > HBOND_DIST_MAX:
                continue

            rec_element = rec_atom['element'].upper()
            is_rec_donor = rec_element in DONORS
            is_rec_acceptor = rec_element in ACCEPTORS

            if lig_atom['is_donor'] and is_rec_acceptor:
                hbonds.append({
                    'lig_atom': lig_atom['symbol'],
                    'rec_atom': rec_atom['name'],
                    'rec_resname': rec_atom['resname'],
                    'rec_resnum': rec_atom['resnum'],
                    'distance': dist
                })
            if lig_atom['is_acceptor'] and is_rec_donor:
                hbonds.append({
                    'lig_atom': lig_atom['symbol'],
                    'rec_atom': rec_atom['name'],
                    'rec_resname': rec_atom['resname'],
                    'rec_resnum': rec_atom['resnum'],
                    'distance': dist
                })
    return hbonds


def protonate_sdf(input_sdf, output_sdf, ph=7.4):
    """Protonate SDF at specified pH using Open Babel."""
    cmd = ['obabel', str(input_sdf), '-O', str(output_sdf), '-p', str(ph)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def screen_poses_for_hbonds(poses_sdf, receptor_pdb, required_residues, min_required=1, ph=7.4):
    """
    Screen poses for H-bonds to required residues.

    Returns:
        List of (pose_idx, mol, score, contacted_residues) for poses meeting criteria,
        sorted by number of contacts (descending) then by original rank.
    """
    if not HAS_RDKIT:
        print("  Warning: RDKit not available, skipping H-bond screening")
        return None

    # Parse required residues
    required_resnums = set()
    for spec in required_residues:
        _, resnum = parse_residue_spec(spec)
        if resnum:
            required_resnums.add(resnum)

    if not required_resnums:
        return None

    # Protonate ligand
    poses_file = Path(poses_sdf)
    protonated_file = poses_file.parent / f"{poses_file.stem}_pH{ph}.sdf"

    if protonate_sdf(poses_file, protonated_file, ph):
        supplier = Chem.SDMolSupplier(str(protonated_file), removeHs=False)
    else:
        supplier = Chem.SDMolSupplier(str(poses_sdf), removeHs=False)

    # Parse receptor
    receptor_atoms = parse_pdb_for_hbonds(receptor_pdb, residue_filter=required_resnums)

    # Analyze each pose
    results = []
    for i, mol in enumerate(supplier):
        if mol is None:
            continue

        lig_atoms = get_ligand_atoms_rdkit(mol)
        hbonds = calculate_hbonds(lig_atoms, receptor_atoms)

        # Find contacted required residues
        contacted = set()
        for hb in hbonds:
            if hb['rec_resnum'] in required_resnums:
                contacted.add(f"{hb['rec_resname']}{hb['rec_resnum']}")

        results.append({
            'pose_idx': i,
            'mol': mol,
            'n_contacts': len(contacted),
            'contacted': list(contacted),
            'passes': len(contacted) >= min_required
        })

    return results


def select_best_pose_with_hbonds(screening_results, min_required=1):
    """
    Select best pose that meets H-bond criteria.
    Prioritizes by: passes criteria > number of contacts > original rank.
    """
    if not screening_results:
        return None

    # Filter passing poses
    passing = [r for r in screening_results if r['passes']]

    if passing:
        # Sort by number of contacts (descending), then by original pose index
        passing.sort(key=lambda x: (-x['n_contacts'], x['pose_idx']))
        return passing[0]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Dock a ligand to a receptor using AutoDock Vina",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--ligand', '-l', required=True,
                        help='Ligand SDF file')
    parser.add_argument('--receptor', '-r', required=True,
                        help='Receptor PDB file (e.g., 4CXA_predock.pdb)')
    parser.add_argument('--reference', '-ref', required=True,
                        help='Reference PDB with co-crystallized ligand')

    # Optional arguments
    parser.add_argument('--ligname', '-n', default=None,
                        help='Ligand residue name in reference (e.g., LIG, 9GF)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output prefix (path allowed). If not provided, uses outdir/receptor_ligand')
    parser.add_argument('--outdir', '-d', default='docking_results',
                        help='Base output directory if --output is not a path (default: docking_results)')

    # Vina parameters
    parser.add_argument('--exhaustiveness', '-e', type=int, default=32,
                        help='Search exhaustiveness (default: 32)')
    parser.add_argument('--num-modes', type=int, default=20,
                        help='Number of poses to generate (default: 20)')
    parser.add_argument('--energy-range', type=float, default=4.0,
                        help='Energy range for poses in kcal/mol (default: 4.0)')
    parser.add_argument('--padding', type=float, default=10.0,
                        help='Box padding around ligand in Å (default: 10.0)')

    # H-bond screening parameters
    parser.add_argument('--prioritise_hbonds',
                        help='Comma-separated residues to prioritise H-bonds (e.g., M816,E814,Y815,D819)')
    parser.add_argument('--min_hbond_contacts', type=int, default=1,
                        help='Minimum number of prioritised residues with H-bonds (default: 1)')
    parser.add_argument('--protonate_ph', type=float, default=7.4,
                        help='pH for ligand protonation before H-bond analysis (default: 7.4)')

    args = parser.parse_args()

    # Validate inputs
    ligand_path = Path(args.ligand)
    receptor_path = Path(args.receptor)
    reference_path = Path(args.reference)

    if not ligand_path.exists():
        print(f"ERROR: Ligand file not found: {args.ligand}")
        sys.exit(1)
    if not receptor_path.exists():
        print(f"ERROR: Receptor file not found: {args.receptor}")
        sys.exit(1)
    if not reference_path.exists():
        print(f"ERROR: Reference file not found: {args.reference}")
        sys.exit(1)

    # Determine output directory and prefix
    if args.output:
        out_path = Path(args.output)
        if out_path.parent == Path('.'):
            # output is just a name; place it in args.outdir
            outdir = Path(args.outdir)
            output_prefix = out_path.name
        else:
            # output includes a directory; use it as outdir and basename as prefix
            outdir = out_path.parent
            output_prefix = out_path.name
    else:
        outdir = Path(args.outdir)
        receptor_name = receptor_path.stem.replace('_predock', '')
        ligand_name = ligand_path.stem
        output_prefix = f"{receptor_name}_{ligand_name}"

    # Create the output directory
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("AUTODOCK VINA DOCKING")
    print("="*60)
    print(f"\nInputs:")
    print(f"  Ligand:    {ligand_path}")
    print(f"  Receptor:  {receptor_path}")
    print(f"  Reference: {reference_path}")
    print(f"\nOutput:")
    print(f"  Directory: {outdir}")
    print(f"  Prefix:    {output_prefix}")

    # Step 1: Extract binding site from reference
    print("\n" + "-"*60)
    print("Step 1: Extract binding site")
    print("-"*60)
    binding_site = extract_binding_site(reference_path, args.ligname, args.padding)

    # Step 2: Prepare ligand
    print("\n" + "-"*60)
    print("Step 2: Prepare ligand")
    print("-"*60)
    ligand_pdbqt = outdir / f"{output_prefix}_ligand.pdbqt"
    if not prepare_ligand(ligand_path, ligand_pdbqt):
        print("ERROR: Ligand preparation failed")
        sys.exit(1)

    # Step 3: Prepare receptor
    print("\n" + "-"*60)
    print("Step 3: Prepare receptor")
    print("-"*60)
    receptor_pdbqt = outdir / f"{output_prefix}_receptor.pdbqt"
    if not prepare_receptor(receptor_path, receptor_pdbqt):
        print("ERROR: Receptor preparation failed")
        sys.exit(1)

    # Step 4: Run Vina
    print("\n" + "-"*60)
    print("Step 4: Run docking")
    print("-"*60)
    output_pdbqt = outdir / f"{output_prefix}_docked.pdbqt"
    if not run_vina(receptor_pdbqt, ligand_pdbqt, output_pdbqt, binding_site,
                    args.exhaustiveness, args.num_modes, args.energy_range):
        print("ERROR: Docking failed")
        sys.exit(1)

    # Step 5: Parse and display results
    print("\n" + "-"*60)
    print("Step 5: Results")
    print("-"*60)
    results = parse_vina_results(output_pdbqt)

    if results:
        print(f"\n  Best affinity: {results[0]['affinity']:.2f} kcal/mol")
        print(f"  Total poses: {len(results)}")
        print(f"\n  Top 5 poses:")
        for r in results[:5]:
            print(f"    Mode {r['mode']}: {r['affinity']:.2f} kcal/mol")

        # Save results to JSON
        results_file = outdir / f"{output_prefix}_results.json"
        ensure_parent_dir(results_file)
        with open(results_file, 'w') as f:
            json.dump({
                'ligand': str(ligand_path),
                'receptor': str(receptor_path),
                'reference': str(reference_path),
                'binding_site': binding_site,
                'exhaustiveness': args.exhaustiveness,
                'poses': results,
                'best_affinity': results[0]['affinity']
            }, f, indent=2)
        print(f"\n  Results saved: {results_file}")
    else:
        print("\n  Warning: No poses found")

    # Step 6: Export SDF for CHARMM-GUI using Meeko (preserves atom order)
    print("\n" + "-"*60)
    print("Step 6: Export docked pose(s) to SDF")
    print("-"*60)
    sdf_best = outdir / f"{output_prefix}_best.sdf"
    sdf_all = outdir / f"{output_prefix}_poses.sdf"

    # Export all poses first (needed for H-bond screening)
    export_docked_poses_meeko(output_pdbqt, sdf_all, best_only=False)

    # Step 7: H-bond screening (if requested)
    hbond_selected = False
    if args.prioritise_hbonds:
        print("\n" + "-"*60)
        print("Step 7: H-bond screening")
        print("-"*60)

        required_residues = [r.strip() for r in args.prioritise_hbonds.split(',')]
        print(f"\n  Screening for H-bonds to: {', '.join(required_residues)}")
        print(f"  Minimum contacts required: {args.min_hbond_contacts}")

        screening_results = screen_poses_for_hbonds(
            sdf_all, receptor_path, required_residues,
            min_required=args.min_hbond_contacts,
            ph=args.protonate_ph
        )

        if screening_results:
            # Print screening summary
            print(f"\n  Pose screening results:")
            for r in screening_results:
                status = "PASS" if r['passes'] else "FAIL"
                contacts = ', '.join(r['contacted']) if r['contacted'] else 'None'
                print(f"    Pose {r['pose_idx']+1}: {status} | Contacts: {contacts}")

            # Select best pose with H-bonds
            best_hbond = select_best_pose_with_hbonds(screening_results, args.min_hbond_contacts)

            if best_hbond:
                print(f"\n  Selected pose {best_hbond['pose_idx']+1} with {best_hbond['n_contacts']} hinge contact(s)")
                print(f"    Contacted residues: {', '.join(best_hbond['contacted'])}")

                # Write selected pose to best.sdf
                writer = Chem.SDWriter(str(sdf_best))
                writer.write(best_hbond['mol'])
                writer.close()
                hbond_selected = True

                # Also write screened poses (all passing)
                sdf_screened = outdir / f"{output_prefix}_hbond_filtered.sdf"
                writer = Chem.SDWriter(str(sdf_screened))
                for r in screening_results:
                    if r['passes']:
                        writer.write(r['mol'])
                writer.close()
                print(f"  Filtered poses: {sdf_screened}")
            else:
                print(f"\n  WARNING: No poses meet H-bond criteria!")
                print(f"  Falling back to Vina best pose (pose 1)")
        else:
            print(f"\n  H-bond screening skipped (RDKit not available or no valid residues)")

    # Export best pose if not already done by H-bond screening
    if not hbond_selected:
        if not export_docked_poses_meeko(output_pdbqt, sdf_best, best_only=True):
            print("WARNING: Could not export best pose SDF via Meeko")

    # Step 8: Create complex PDB (receptor + ligand)
    print("\n" + "-"*60)
    print("Step 8: Create complex PDB")
    print("-"*60)
    complex_pdb = outdir / f"{output_prefix}_complex.pdb"
    if create_complex_pdb(receptor_path, sdf_best, complex_pdb, ligand_resname='LIG'):
        print(f"  Complex PDB: {complex_pdb}")
    else:
        print("  WARNING: Could not create complex PDB")

    print(f"\nOutput files:")
    print(f"  Best pose SDF: {sdf_best}")
    if hbond_selected:
        print(f"    (Selected by H-bond criteria, not Vina score)")
    print(f"  All poses SDF: {sdf_all}")
    print(f"  Complex PDB:   {complex_pdb}")

    # Summary
    print("\n" + "="*60)
    print("DOCKING COMPLETE")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  {complex_pdb}  <- Use this for MD setup")
    print(f"  {sdf_best}")
    print(f"  {sdf_all}")
    print(f"  {output_pdbqt}")
    if results:
        print(f"  {outdir / f'{output_prefix}_results.json'}")
    if hbond_selected:
        print(f"\nNote: Best pose selected by H-bond contacts to {args.prioritise_hbonds}")
    print(f"\nVisualize:")
    print(f"  pymol {complex_pdb}")
    print()


if __name__ == "__main__":
    main()
