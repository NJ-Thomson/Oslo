#!/usr/bin/env python3
"""
Versatile AutoDock Vina Docking Script

Usage:
    python3 vina_docking.py --ligand LIGAND.sdf --receptor RECEPTOR.pdb \\
                            --reference REF.pdb --ligname LIGNAME

Arguments:
    --ligand      : Ligand SDF file
    --receptor    : Receptor PDB file(s) - can specify multiple
    --reference   : Reference PDB with ligand (for binding site center)
    --ligname     : Ligand residue name in reference (e.g., 9GF)
    --output      : Output directory (default: vina_results)
    --exhaustiveness : Search thoroughness (default: 32)
    --num-modes   : Number of poses (default: 20)
    --energy-range: Energy range for poses (default: 4 kcal/mol)

Examples:
    python3 vina_docking.py --ligand 9GF_ideal.sdf \\
                            --receptor ArrB1_CB1.pdb Gio_CB1.pdb \\
                            --reference 7fee.pdb --ligname 9GF
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import json
import numpy as np

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_command(cmd, description):
    """Execute a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDERR: {result.stderr}")
        print(f"STDOUT: {result.stdout}")
        return False
    
    print(f"SUCCESS: {description} completed")
    if result.stdout:
        print(f"Output: {result.stdout[:500]}")
    return True


def check_file_exists(filepath, description):
    """Check if required file exists"""
    if not Path(filepath).exists():
        print(f"ERROR: {description} not found: {filepath}")
        return False
    print(f"✓ Found {description}: {filepath}")
    return True


def extract_binding_site_from_reference(reference_pdb, ligand_name=None, exclude_list=None):
    """Extract binding site coordinates from reference PDB with ligand"""
    
    if exclude_list is None:
        exclude_list = {'HOH', 'WAT', 'TIP', 'TIP3', 'NA', 'CL', 'K', 'MG', 
                       'CA', 'ZN', 'FE', 'MN', 'SO4', 'PO4', 'GOL', 'EDO'}
    
    print(f"\n{'='*60}")
    print(f"Extracting binding site from: {reference_pdb}")
    if ligand_name:
        print(f"Looking for ligand: {ligand_name}")
    print(f"{'='*60}")
    
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
            except:
                pass
    
    if not ligand_coords:
        print(f"ERROR: No ligand found!")
        if ligand_name:
            print(f"  Could not find '{ligand_name}' in reference")
        sys.exit(1)
    
    print(f"✓ Found ligand(s): {', '.join(f'{k} ({v} atoms)' for k, v in ligand_residues.items())}")
    
    ligand_coords = np.array(ligand_coords)
    center = ligand_coords.mean(axis=0)
    
    # Calculate box size
    ligand_min = ligand_coords.min(axis=0)
    ligand_max = ligand_coords.max(axis=0)
    ligand_span = ligand_max - ligand_min
    box_size = ligand_span + 10  # Add 10Å buffer
    box_size = np.minimum(box_size, 30)  # Cap at 30Å
    
    print(f"\n✓ Binding site:")
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


def prepare_ligand_meeko(sdf_file, output_pdbqt):
    """Prepare ligand using Meeko"""
    cmd = [
        "mk_prepare_ligand.py",
        "-i", sdf_file,
        "-o", output_pdbqt,
        "--keep_nonpolar_hydrogens"
    ]
    return run_command(cmd, f"Preparing ligand from {sdf_file}")


def prepare_receptor_simple(pdb_file, output_pdbqt):
    """Prepare receptor using Open Babel"""
    
    # Add hydrogens
    temp_pdb_h = pdb_file.replace('.pdb', '_withH.pdb')
    cmd = ['obabel', pdb_file, '-O', temp_pdb_h, '-h']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Could not add hydrogens: {result.stderr}")
        return False
    
    # Convert to PDBQT
    cmd = ['obabel', temp_pdb_h, '-O', output_pdbqt, '-xr']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up
    if Path(temp_pdb_h).exists():
        Path(temp_pdb_h).unlink()
    
    if result.returncode != 0:
        print(f"ERROR: Could not create PDBQT: {result.stderr}")
        return False
    
    print(f"✓ Prepared receptor: {output_pdbqt}")
    return True


def run_vina_docking(receptor_pdbqt, ligand_pdbqt, output_pdbqt, binding_site, 
                     exhaustiveness, num_modes, energy_range):
    """Run AutoDock Vina docking"""
    
    cmd = [
        "vina",
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--out", output_pdbqt,
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
    
    success = run_command(cmd, f"Docking with Vina")
    
    if success:
        # Parse output PDBQT instead of log file
        parse_vina_output(output_pdbqt)
    
    return success


def parse_vina_output(output_pdbqt):
    """Parse Vina output PDBQT file for energies"""
    
    try:
        with open(output_pdbqt, 'r') as f:
            lines = f.readlines()
        
        print(f"\n{'='*60}")
        print(f"Vina Results")
        print(f"{'='*60}")
        
        results = []
        
        for line in lines:
            # Vina writes energies in REMARK lines
            if line.startswith("REMARK VINA RESULT:"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        affinity = float(parts[3])
                        mode = len(results) + 1
                        results.append({
                            'mode': mode,
                            'affinity': affinity
                        })
                    except:
                        pass
        
        if results:
            print(f"\nBest binding affinity: {results[0]['affinity']:.2f} kcal/mol")
            print(f"Number of poses: {len(results)}")
            print(f"\nTop 5 poses:")
            for res in results[:5]:
                print(f"  Mode {res['mode']}: {res['affinity']:.2f} kcal/mol")
            
            # Save results
            json_file = output_pdbqt.replace('_docked.pdbqt', '_results.json')
            with open(json_file, 'w') as f:
                json.dump({
                    'receptor': Path(output_pdbqt).stem.replace('_docked', ''),
                    'poses': results,
                    'best_affinity': results[0]['affinity']
                }, f, indent=2)
            
            print(f"\n✓ Results saved to {json_file}")
        else:
            print("\nWarning: Could not parse energies from output")
        
    except Exception as e:
        print(f"Warning: Could not parse results: {e}")


def parse_vina_results(log_file, output_pdbqt):
    """Deprecated - kept for compatibility"""
    parse_vina_output(output_pdbqt)


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main(args):
    """Main workflow for Vina docking"""
    
    print("\n" + "="*80)
    print("AUTODOCK VINA DOCKING WORKFLOW")
    print("="*80)
    
    output_dir = Path(args.output)
    
    print(f"\nConfiguration:")
    print(f"  Ligand:        {args.ligand}")
    print(f"  Receptor(s):   {', '.join(args.receptor)}")
    print(f"  Reference:     {args.reference}")
    if args.ligname:
        print(f"  Ligand name:   {args.ligname}")
    print(f"  Output dir:    {output_dir.absolute()}")
    print(f"  Exhaustiveness: {args.exhaustiveness}")
    print(f"  Num modes:     {args.num_modes}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Check input files
    print("\n1. Checking input files...")
    if not check_file_exists(args.ligand, "Ligand SDF"):
        sys.exit(1)
    
    for receptor in args.receptor:
        if not check_file_exists(receptor, "Receptor"):
            sys.exit(1)
    
    if not check_file_exists(args.reference, "Reference PDB"):
        sys.exit(1)
    
    # Check Vina is available
    print("\n2. Checking AutoDock Vina installation...")
    result = subprocess.run(["vina", "--help"], capture_output=True)
    if result.returncode != 0:
        print("ERROR: vina not found!")
        print("Install with: pip install vina --break-system-packages")
        sys.exit(1)
    print("✓ AutoDock Vina found")
    
    # Extract binding site from reference
    print("\n3. Extracting binding site from reference...")
    binding_site = extract_binding_site_from_reference(args.reference, args.ligname)
    
    # Prepare ligand
    print("\n4. Preparing ligand with Meeko...")
    ligand_pdbqt = str(output_dir / f"{Path(args.ligand).stem}_prepared.pdbqt")
    if not prepare_ligand_meeko(args.ligand, ligand_pdbqt):
        sys.exit(1)
    
    # Process each receptor
    for receptor_pdb in args.receptor:
        receptor_name = Path(receptor_pdb).stem
        
        print(f"\n{'#'*80}")
        print(f"PROCESSING {receptor_name}")
        print(f"{'#'*80}")
        
        # Prepare receptor
        print(f"\n5. Preparing {receptor_name} receptor...")
        receptor_pdbqt = str(output_dir / f"{receptor_name}_prepared.pdbqt")
        if not prepare_receptor_simple(receptor_pdb, receptor_pdbqt):
            print(f"Warning: Could not prepare {receptor_name}, skipping...")
            continue
        
        # Run Vina
        print(f"\n6. Running Vina for {receptor_name}...")
        print(f"   This will take a few minutes per pose (exhaustiveness={args.exhaustiveness})...")
        
        output_pdbqt = str(output_dir / f"{receptor_name}_docked.pdbqt")
        
        if not run_vina_docking(receptor_pdbqt, ligand_pdbqt, output_pdbqt, 
                               binding_site, args.exhaustiveness, 
                               args.num_modes, args.energy_range):
            print(f"Warning: Vina failed for {receptor_name}")
            continue
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print(f"\nView results:")
    for receptor in args.receptor:
        receptor_name = Path(receptor).stem
        print(f"  cat {output_dir}/{receptor_name}_results.json")
    print(f"\nVisualize:")
    for receptor in args.receptor:
        receptor_name = Path(receptor).stem
        print(f"  pymol {receptor_name}.pdb {output_dir}/{receptor_name}_docked.pdbqt")
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoDock Vina docking workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 %(prog)s --ligand 9GF_ideal.sdf --receptor ArrB1_CB1.pdb --reference 7fee.pdb --ligname 9GF

  # Multiple receptors
  python3 %(prog)s --ligand 9GF_ideal.sdf --receptor ArrB1_CB1.pdb Gio_CB1.pdb \\
                   --reference 7fee.pdb --ligname 9GF

  # Custom parameters
  python3 %(prog)s --ligand 9GF_ideal.sdf --receptor ArrB1_CB1.pdb --reference 7fee.pdb \\
                   --ligname 9GF --exhaustiveness 64 --num-modes 50
        """
    )
    
    # Required arguments
    parser.add_argument('--ligand', required=True, help='Ligand SDF file')
    parser.add_argument('--receptor', required=True, nargs='+', help='Receptor PDB file(s)')
    parser.add_argument('--reference', required=True, help='Reference PDB with ligand')
    
    # Important optional
    parser.add_argument('--ligname', default=None, help='Ligand residue name in reference (e.g., 9GF)')
    parser.add_argument('--output', default='vina_results', help='Output directory (default: vina_results)')
    
    # Vina parameters
    parser.add_argument('--exhaustiveness', type=int, default=32,
                        help='Search exhaustiveness (default: 32, higher=slower but better)')
    parser.add_argument('--num-modes', type=int, default=20,
                        help='Number of binding modes (default: 20)')
    parser.add_argument('--energy-range', type=float, default=4.0,
                        help='Energy range for poses in kcal/mol (default: 4)')
    
    args = parser.parse_args()
    
    main(args)
