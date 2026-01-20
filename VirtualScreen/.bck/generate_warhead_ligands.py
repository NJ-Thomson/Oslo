#!/usr/bin/env python3
"""
Generate Covalent Warhead Variants from Brominated Scaffolds

Replaces bromine with reactive warheads (acrylamide, chloroacetamide, etc.)
for covalent inhibitor design.

Chemistry:
    Scaffold-Br  →  Scaffold-NH-C(=O)-CH=CH2      (acrylamide)
    Scaffold-Br  →  Scaffold-NH-C(=O)-CH2-Cl      (chloroacetamide)
    Scaffold-Br  →  Scaffold-NH-C(=O)-CH2-Br      (bromoacetamide)

Usage:
    python generate_warhead_ligands.py --input Inhib_32.sdf --warhead acrylamide
    python generate_warhead_ligands.py --input Inhib_32.sdf --warhead all
    python generate_warhead_ligands.py --input ligands/*.sdf --warhead acrylamide chloroacetamide

Output:
    For each input: Inhib_32_acrylamide.sdf, Inhib_32_chloroacetamide.sdf, etc.
"""

import argparse
import sys
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
except ImportError:
    print("ERROR: RDKit is required")
    print("Install with: conda install -c conda-forge rdkit")
    sys.exit(1)


# Warhead definitions
# These SMILES have the NH2 that will attach to the scaffold carbon
WARHEADS = {
    'acrylamide': {
        'smiles': '[NH2]C(=O)C=C',        # H2N-C(=O)-CH=CH2
        'full_name': 'Acrylamide',
        'mechanism': 'Michael addition to β-carbon',
    },
    'chloroacetamide': {
        'smiles': '[NH2]C(=O)CCl',         # H2N-C(=O)-CH2-Cl  
        'full_name': 'Chloroacetamide',
        'mechanism': 'SN2 displacement of Cl',
    },
    'bromoacetamide': {
        'smiles': '[NH2]C(=O)CBr',         # H2N-C(=O)-CH2-Br
        'full_name': 'Bromoacetamide', 
        'mechanism': 'SN2 displacement of Br',
    },
    'propiolamide': {
        'smiles': '[NH2]C(=O)C#C',         # H2N-C(=O)-C≡CH
        'full_name': 'Propiolamide',
        'mechanism': 'Michael addition',
    },
    'vinylsulfonamide': {
        'smiles': '[NH2]S(=O)(=O)C=C',     # H2N-SO2-CH=CH2
        'full_name': 'Vinyl sulfonamide',
        'mechanism': 'Michael addition',
    },
}


def find_bromine_attachment(mol):
    """
    Find bromine atoms attached to aromatic carbons (typical synthetic handle).
    
    Returns list of (Br_idx, attached_C_idx) tuples.
    """
    br_attachments = []
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'Br':
            br_idx = atom.GetIdx()
            neighbors = atom.GetNeighbors()
            
            if len(neighbors) == 1:
                neighbor = neighbors[0]
                if neighbor.GetSymbol() == 'C':
                    br_attachments.append((br_idx, neighbor.GetIdx()))
    
    return br_attachments


def replace_br_with_warhead(mol, br_idx, c_idx, warhead_smiles):
    """
    Replace Br with warhead at the specified position.
    
    The nitrogen of the warhead bonds to the carbon where Br was attached.
    Br-C becomes N-C (warhead attaches via its NH2 becoming NH)
    """
    # Create editable molecule
    emol = Chem.RWMol(Chem.Mol(mol))  # Make a copy
    
    # Get the warhead
    warhead = Chem.MolFromSmiles(warhead_smiles)
    if warhead is None:
        print(f"  ERROR: Invalid warhead SMILES: {warhead_smiles}")
        return None
    
    # Find the nitrogen in the warhead (the NH2 attachment point)
    n_idx_warhead = None
    for atom in warhead.GetAtoms():
        if atom.GetSymbol() == 'N':
            # Check if this N has 2 H (is NH2)
            total_h = atom.GetTotalNumHs()
            if total_h == 2:
                n_idx_warhead = atom.GetIdx()
                break
    
    if n_idx_warhead is None:
        # Try finding any nitrogen
        for atom in warhead.GetAtoms():
            if atom.GetSymbol() == 'N':
                n_idx_warhead = atom.GetIdx()
                print(f"  Using N at index {n_idx_warhead}")
                break
    
    if n_idx_warhead is None:
        print(f"  ERROR: Could not find N in warhead")
        print(f"  Warhead SMILES: {warhead_smiles}")
        print(f"  Warhead atoms: {[a.GetSymbol() for a in warhead.GetAtoms()]}")
        return None
    
    # Remove Br from scaffold
    emol.RemoveAtom(br_idx)
    
    # Adjust c_idx if Br was before C in atom ordering
    if br_idx < c_idx:
        c_idx -= 1
    
    # Get the scaffold molecule after Br removal
    scaffold = emol.GetMol()
    
    # Combine scaffold + warhead
    combo = Chem.RWMol(Chem.CombineMols(scaffold, warhead))
    
    # Calculate offset for warhead atoms in combined molecule
    offset = scaffold.GetNumAtoms()
    n_idx_combo = n_idx_warhead + offset
    
    # Add bond between scaffold C and warhead N
    combo.AddBond(c_idx, n_idx_combo, Chem.BondType.SINGLE)
    
    # Update the nitrogen - it loses one H when bonding to scaffold
    n_atom = combo.GetAtomWithIdx(n_idx_combo)
    current_h = n_atom.GetNumExplicitHs()
    if current_h > 0:
        n_atom.SetNumExplicitHs(current_h - 1)
    
    # Sanitize and clean up
    try:
        result = combo.GetMol()
        Chem.SanitizeMol(result)
        result = Chem.RemoveHs(result)
        return result
    except Exception as e:
        print(f"  ERROR sanitizing molecule: {e}")
        # Try returning unsanitized
        try:
            return combo.GetMol()
        except:
            return None


def generate_warhead_variant(input_mol, warhead_type, mol_name="ligand"):
    """
    Generate a warhead variant from a brominated scaffold.
    """
    config = WARHEADS.get(warhead_type)
    if not config:
        print(f"  ERROR: Unknown warhead type: {warhead_type}")
        print(f"  Available: {', '.join(WARHEADS.keys())}")
        return None
    
    # Find Br attachment points
    br_attachments = find_bromine_attachment(input_mol)
    
    if not br_attachments:
        print(f"  ERROR: No bromine found in {mol_name}")
        return None
    
    if len(br_attachments) > 1:
        print(f"  WARNING: Multiple bromines found ({len(br_attachments)}), using first one")
    
    br_idx, c_idx = br_attachments[0]
    
    print(f"  Found Br at atom {br_idx}, attached to C at atom {c_idx}")
    print(f"  Replacing with {config['full_name']}")
    
    # Replace Br with warhead
    new_mol = replace_br_with_warhead(input_mol, br_idx, c_idx, config['smiles'])
    
    if new_mol is None:
        return None
    
    # Set properties
    new_mol.SetProp("_Name", f"{mol_name}_{warhead_type}")
    new_mol.SetProp("Warhead", warhead_type)
    new_mol.SetProp("Mechanism", config['mechanism'])
    
    return new_mol


def process_molecule(input_file, warhead_types, output_dir):
    """
    Process a single molecule file and generate warhead variants.
    """
    input_path = Path(input_file)
    mol_name = input_path.stem
    
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"{'='*60}")
    
    # Load molecule
    if input_path.suffix.lower() == '.sdf':
        suppl = Chem.SDMolSupplier(str(input_path), removeHs=False)
        mol = next(suppl, None)
    elif input_path.suffix.lower() == '.mol':
        mol = Chem.MolFromMolFile(str(input_path), removeHs=False)
    else:
        print(f"  ERROR: Unsupported format: {input_path.suffix}")
        return []
    
    if mol is None:
        print(f"  ERROR: Could not load molecule")
        return []
    
    print(f"  Loaded: {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")
    print(f"  Formula: {CalcMolFormula(mol)}")
    
    # Check for Br
    br_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'Br')
    print(f"  Bromine atoms: {br_count}")
    
    if br_count == 0:
        print(f"  ERROR: No bromine found - cannot generate warhead variants")
        return []
    
    # Generate variants
    output_files = []
    
    for warhead in warhead_types:
        print(f"\n  Generating {warhead} variant...")
        
        new_mol = generate_warhead_variant(mol, warhead, mol_name)
        
        if new_mol is None:
            continue
        
        # Generate 3D coordinates
        try:
            AllChem.EmbedMolecule(new_mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(new_mol)
        except Exception as e:
            print(f"  WARNING: Could not generate 3D coords: {e}")
        
        # Write output
        output_file = output_dir / f"{mol_name}_{warhead}.sdf"
        
        writer = Chem.SDWriter(str(output_file))
        writer.write(new_mol)
        writer.close()
        
        print(f"  ✓ Created: {output_file}")
        print(f"    Formula: {CalcMolFormula(new_mol)}")
        print(f"    Atoms: {new_mol.GetNumAtoms()}")
        
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate covalent warhead variants from brominated scaffolds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available warheads:
{chr(10).join(f"  {k:20s} - {v['full_name']} ({v['mechanism']})" for k, v in WARHEADS.items())}

Examples:
  # Single warhead
  python %(prog)s --input Inhib_32.sdf --warhead acrylamide

  # Multiple warheads
  python %(prog)s --input Inhib_32.sdf --warhead acrylamide chloroacetamide

  # All warheads
  python %(prog)s --input Inhib_32.sdf --warhead all

  # Multiple input files
  python %(prog)s --input ligands/*.sdf --warhead acrylamide
        """
    )
    
    parser.add_argument('--input', '-i', nargs='+', required=True,
                        help='Input SDF/MOL file(s) with brominated scaffold')
    parser.add_argument('--warhead', '-w', nargs='+', required=True,
                        help='Warhead type(s) or "all"')
    parser.add_argument('--outdir', '-o', default='warhead_ligands',
                        help='Output directory (default: warhead_ligands)')
    
    args = parser.parse_args()
    
    # Process warhead selection
    if 'all' in args.warhead:
        warhead_types = list(WARHEADS.keys())
    else:
        warhead_types = args.warhead
        for w in warhead_types:
            if w not in WARHEADS:
                print(f"ERROR: Unknown warhead '{w}'")
                print(f"Available: {', '.join(WARHEADS.keys())}")
                sys.exit(1)
    
    print("\n" + "="*60)
    print("WARHEAD VARIANT GENERATOR")
    print("="*60)
    print(f"\nWarheads to generate: {', '.join(warhead_types)}")
    print(f"Input files: {len(args.input)}")
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Process each input
    all_outputs = []
    for input_file in args.input:
        outputs = process_molecule(input_file, warhead_types, outdir)
        all_outputs.extend(outputs)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nGenerated {len(all_outputs)} warhead variants:")
    for f in all_outputs:
        print(f"  {f}")
    
    print(f"\nNext steps:")
    print(f"  1. Use covalent_place.py to position these in the binding site")
    print(f"  2. Or use for covalent docking with appropriate software")
    print()


if __name__ == "__main__":
    main()