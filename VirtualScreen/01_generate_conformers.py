#!/usr/bin/env python3
"""
Generate 3D conformers from SMILES using RDKit ETKDG + MMFF94 optimisation.
Falls back to OpenBabel for problematic SMILES that RDKit cannot kekulize.

Input: Tab or space-separated text file with columns:
    Column 1: Ligand name
    Column 2: SMILES

Output: Individual .mol files for each ligand

Dependencies:
    - rdkit: conda install -c conda-forge rdkit
    - openbabel: conda install -c conda-forge openbabel
"""

import argparse
import subprocess
import shutil
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_conformer_openbabel(name, smiles, output_dir, output_format='sdf'):
    """Fallback: Generate 3D conformer using OpenBabel."""
    output_path = output_dir / f"{name}.{output_format}"
    
    # Check if obabel is available
    if shutil.which("obabel") is None:
        print(f"  ERROR: OpenBabel not found. Install with: conda install -c conda-forge openbabel")
        return False
    
    try:
        result = subprocess.run(
            ["obabel", f"-:{smiles}", "-O", str(output_path), "--gen3d", "--ff", "MMFF94", "--title", name],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and output_path.exists():
            print(f"  Generated (OpenBabel fallback): {output_path}")
            return True
        else:
            print(f"  ERROR: OpenBabel failed for {name}")
            if result.stderr:
                print(f"    {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ERROR: OpenBabel timed out for {name}")
        return False
    except Exception as e:
        print(f"  ERROR: OpenBabel exception for {name}: {e}")
        return False


def generate_conformer_rdkit(name, smiles, output_dir, output_format='sdf'):
    """Generate 3D conformer using RDKit."""
    # Try standard parsing first
    mol = Chem.MolFromSmiles(smiles)
    
    # If that fails, try without sanitization then sanitize manually
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(mol)
        except Exception:
            mol = None
    
    if mol is None:
        return False
    
    # Set molecule name
    mol.SetProp("_Name", name)
    
    mol = Chem.AddHs(mol)
    
    # Generate 3D conformer using ETKDGv3
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result == -1:
        return False
    
    # Optimise with MMFF94 force field
    opt_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
    if opt_result != 0:
        # Try UFF as fallback optimiser
        AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        print(f"  WARNING: MMFF94 did not converge, used UFF for {name}")
    
    # Write to file
    output_path = output_dir / f"{name}.{output_format}"
    
    if output_format == 'sdf':
        writer = Chem.SDWriter(str(output_path))
        writer.write(mol)
        writer.close()
    else:
        Chem.MolToMolFile(mol, str(output_path))
    
    print(f"  Generated: {output_path}")
    return True


def generate_conformer(name, smiles, output_dir, output_format='sdf'):
    """Generate 3D conformer, trying RDKit first then OpenBabel as fallback."""
    # Try RDKit first
    if generate_conformer_rdkit(name, smiles, output_dir, output_format):
        return True
    
    # If RDKit fails, try OpenBabel
    print(f"  RDKit failed for {name}, trying OpenBabel...")
    return generate_conformer_openbabel(name, smiles, output_dir, output_format)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D conformers from SMILES using RDKit"
    )
    parser.add_argument(
        "input_file",
        help="Text file with ligand names (col 1) and SMILES (col 2)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="conformers",
        help="Output directory for structure files (default: conformers)"
    )
    parser.add_argument(
        "-f", "--format",
        default="sdf",
        choices=["sdf", "mol"],
        help="Output format: sdf or mol (default: sdf)"
    )
    parser.add_argument(
        "-d", "--delimiter",
        default=None,
        help="Column delimiter (default: whitespace)"
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Read input file
    print(f"Reading ligands from: {args.input_file}")
    print(f"Output format: {args.format}")
    success_count = 0
    fail_count = 0
    
    with open(args.input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(args.delimiter)
            if len(parts) < 2:
                print(f"  WARNING: Skipping line {line_num} - not enough columns")
                continue
            
            name = parts[0].strip()
            smiles = parts[1].strip()
            
            print(f"Processing {name}...")
            if generate_conformer(name, smiles, output_dir, args.format):
                success_count += 1
            else:
                fail_count += 1
    
    print(f"\nComplete: {success_count} succeeded, {fail_count} failed")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()