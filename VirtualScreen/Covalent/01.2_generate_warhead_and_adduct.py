#!/usr/bin/env python3
"""
Generate Covalent Warhead Variants AND Covalent Adducts from Brominated Scaffolds

This script generates:
1. Pre-reaction ligands (warhead attached, ready for non-covalent positioning)
2. Covalent adducts (warhead reacted with cysteine fragment, for scoring/validation)

For acrylamide Michael addition:
    Scaffold-Br  →  Scaffold-NH-C(=O)-CH=CH2           (pre-reaction)
    Scaffold-Br  →  Scaffold-NH-C(=O)-CH2-CH2-S-[Cys]  (adduct)

Usage:
    python generate_warhead_and_adduct.py --input Inhib_32.sdf --warhead acrylamide
    python generate_warhead_and_adduct.py --smiles "Brc1cncc2[nH]cc(-c3csc(-c4cc[nH]cn4)n3)c2c1" --name Inhib_32 --warhead acrylamide

Output:
    - Inhib_32_acrylamide.sdf          (pre-reaction, warhead intact)
    - Inhib_32_acrylamide_adduct.sdf   (post-reaction, for covalent docking)
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
# Pre-reaction: The intact warhead attached to scaffold
# Adduct fragment: What the warhead becomes after reacting with Cys-SH
WARHEADS = {
    'acrylamide': {
        'pre_reaction_smiles': '[NH2]C(=O)C=C',           # H2N-C(=O)-CH=CH2
        'adduct_smiles': '[NH2]C(=O)CCS',                 # H2N-C(=O)-CH2-CH2-S (Michael addition product)
        'reactive_atom_symbol': 'S',                       # The S will connect to protein
        'full_name': 'Acrylamide',
        'mechanism': 'Michael addition to β-carbon',
        'description': 'β-carbon attacked by Cys thiolate',
    },
    'chloroacetamide': {
        'pre_reaction_smiles': '[NH2]C(=O)CCl',           # H2N-C(=O)-CH2-Cl
        'adduct_smiles': '[NH2]C(=O)CS',                  # H2N-C(=O)-CH2-S (SN2 product)
        'reactive_atom_symbol': 'S',
        'full_name': 'Chloroacetamide',
        'mechanism': 'SN2 displacement of Cl',
        'description': 'Cl displaced by Cys thiolate',
    },
    'bromoacetamide': {
        'pre_reaction_smiles': '[NH2]C(=O)CBr',           # H2N-C(=O)-CH2-Br
        'adduct_smiles': '[NH2]C(=O)CS',                  # H2N-C(=O)-CH2-S (SN2 product)
        'reactive_atom_symbol': 'S',
        'full_name': 'Bromoacetamide',
        'mechanism': 'SN2 displacement of Br',
        'description': 'Br displaced by Cys thiolate',
    },
    'propiolamide': {
        'pre_reaction_smiles': '[NH2]C(=O)C#C',           # H2N-C(=O)-C≡CH
        'adduct_smiles': '[NH2]C(=O)C=CS',                # H2N-C(=O)-CH=CH-S (vinyl thioether)
        'reactive_atom_symbol': 'S',
        'full_name': 'Propiolamide',
        'mechanism': 'Michael addition to alkyne',
        'description': 'Terminal carbon attacked by Cys thiolate',
    },
    'vinylsulfonamide': {
        'pre_reaction_smiles': '[NH2]S(=O)(=O)C=C',       # H2N-SO2-CH=CH2
        'adduct_smiles': '[NH2]S(=O)(=O)CCS',             # H2N-SO2-CH2-CH2-S
        'reactive_atom_symbol': 'S',
        'full_name': 'Vinyl sulfonamide',
        'mechanism': 'Michael addition',
        'description': 'β-carbon attacked by Cys thiolate',
    },
}


def find_bromine_attachment(mol):
    """
    Find bromine atoms attached to carbons (typical synthetic handle).
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


def replace_br_with_fragment(mol, br_idx, c_idx, fragment_smiles):
    """
    Replace Br with a fragment at the specified position.
    The nitrogen (or first attachment point) bonds to the carbon where Br was.
    """
    emol = Chem.RWMol(Chem.Mol(mol))
    
    fragment = Chem.MolFromSmiles(fragment_smiles)
    if fragment is None:
        print(f"  ERROR: Invalid fragment SMILES: {fragment_smiles}")
        return None
    
    # Find attachment point in fragment (NH2 nitrogen)
    attach_idx = None
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() >= 2:
            attach_idx = atom.GetIdx()
            break
    
    if attach_idx is None:
        # Fallback: find any nitrogen
        for atom in fragment.GetAtoms():
            if atom.GetSymbol() == 'N':
                attach_idx = atom.GetIdx()
                break
    
    if attach_idx is None:
        print(f"  ERROR: Could not find attachment point (N) in fragment")
        return None
    
    # Remove Br
    emol.RemoveAtom(br_idx)
    
    # Adjust c_idx if needed
    if br_idx < c_idx:
        c_idx -= 1
    
    scaffold = emol.GetMol()
    
    # Combine
    combo = Chem.RWMol(Chem.CombineMols(scaffold, fragment))
    
    offset = scaffold.GetNumAtoms()
    attach_idx_combo = attach_idx + offset
    
    # Add bond
    combo.AddBond(c_idx, attach_idx_combo, Chem.BondType.SINGLE)
    
    # Adjust hydrogen count on N
    n_atom = combo.GetAtomWithIdx(attach_idx_combo)
    current_h = n_atom.GetNumExplicitHs()
    if current_h > 0:
        n_atom.SetNumExplicitHs(current_h - 1)
    
    try:
        result = combo.GetMol()
        Chem.SanitizeMol(result)
        result = Chem.RemoveHs(result)
        return result
    except Exception as e:
        print(f"  ERROR sanitizing: {e}")
        return None


def find_terminal_sulfur(mol):
    """Find terminal sulfur (attachment point for covalent bond to protein)."""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'S':
            # Check if it's terminal (only one heavy atom neighbor)
            heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            if len(heavy_neighbors) == 1:
                return atom.GetIdx()
    return None


def generate_variants(input_mol, warhead_type, mol_name="ligand"):
    """
    Generate both pre-reaction warhead and covalent adduct variants.
    
    Returns:
        (pre_reaction_mol, adduct_mol) or (None, None) on failure
    """
    config = WARHEADS.get(warhead_type)
    if not config:
        print(f"  ERROR: Unknown warhead: {warhead_type}")
        return None, None
    
    br_attachments = find_bromine_attachment(input_mol)
    
    if not br_attachments:
        print(f"  ERROR: No bromine found in {mol_name}")
        return None, None
    
    if len(br_attachments) > 1:
        print(f"  WARNING: Multiple bromines ({len(br_attachments)}), using first")
    
    br_idx, c_idx = br_attachments[0]
    print(f"  Found Br at atom {br_idx}, attached to C at atom {c_idx}")
    
    # Generate pre-reaction (intact warhead)
    print(f"  Generating pre-reaction {config['full_name']}...")
    pre_reaction = replace_br_with_fragment(
        input_mol, br_idx, c_idx, 
        config['pre_reaction_smiles']
    )
    
    if pre_reaction:
        pre_reaction.SetProp("_Name", f"{mol_name}_{warhead_type}")
        pre_reaction.SetProp("Type", "pre-reaction")
        pre_reaction.SetProp("Warhead", warhead_type)
        print(f"    ✓ Pre-reaction: {Chem.MolToSmiles(pre_reaction)}")
    
    # Generate adduct (reacted with Cys)
    print(f"  Generating covalent adduct...")
    adduct = replace_br_with_fragment(
        input_mol, br_idx, c_idx,
        config['adduct_smiles']
    )
    
    if adduct:
        adduct.SetProp("_Name", f"{mol_name}_{warhead_type}_adduct")
        adduct.SetProp("Type", "covalent_adduct")
        adduct.SetProp("Warhead", warhead_type)
        adduct.SetProp("Mechanism", config['mechanism'])
        
        # Mark the reactive sulfur for covalent docking
        s_idx = find_terminal_sulfur(adduct)
        if s_idx is not None:
            adduct.SetProp("CovalentAtomIdx", str(s_idx))
            print(f"    ✓ Adduct: {Chem.MolToSmiles(adduct)}")
            print(f"    ✓ Covalent attachment S at atom index {s_idx}")
        else:
            print(f"    WARNING: Could not find terminal S in adduct")
    
    return pre_reaction, adduct


def generate_3d_and_save(mol, output_path, optimize=True):
    """Generate 3D coordinates and save to file."""
    if mol is None:
        return False
    
    mol = Chem.AddHs(mol)
    
    # Generate 3D
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result == -1:
        # Try with random coords
        AllChem.EmbedMolecule(mol, randomSeed=42)
    
    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            except:
                print(f"    WARNING: Could not optimize geometry")
    
    # Save
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()
    
    print(f"    Saved: {output_path}")
    return True


def process_smiles(smiles, name, warhead_types, output_dir):
    """Process a SMILES string directly."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"ERROR: Could not parse SMILES: {smiles}")
        return []
    
    return process_mol(mol, name, warhead_types, output_dir)


def process_file(input_file, warhead_types, output_dir):
    """Process a structure file."""
    input_path = Path(input_file)
    mol_name = input_path.stem
    
    if input_path.suffix.lower() == '.sdf':
        suppl = Chem.SDMolSupplier(str(input_path), removeHs=False)
        mol = next(suppl, None)
    elif input_path.suffix.lower() == '.mol':
        mol = Chem.MolFromMolFile(str(input_path), removeHs=False)
    else:
        print(f"ERROR: Unsupported format: {input_path.suffix}")
        return []
    
    if mol is None:
        print(f"ERROR: Could not load {input_file}")
        return []
    
    return process_mol(mol, mol_name, warhead_types, output_dir)


def process_mol(mol, mol_name, warhead_types, output_dir):
    """Process a molecule and generate all variants."""
    print(f"\n{'='*60}")
    print(f"Processing: {mol_name}")
    print(f"{'='*60}")
    print(f"  Input SMILES: {Chem.MolToSmiles(mol)}")
    print(f"  Formula: {CalcMolFormula(mol)}")
    
    output_files = []
    
    for warhead in warhead_types:
        print(f"\n  --- {warhead.upper()} ---")
        
        pre_reaction, adduct = generate_variants(mol, warhead, mol_name)
        
        if pre_reaction:
            pre_path = output_dir / f"{mol_name}_{warhead}.sdf"
            if generate_3d_and_save(pre_reaction, pre_path):
                output_files.append(('pre-reaction', pre_path))
        
        if adduct:
            adduct_path = output_dir / f"{mol_name}_{warhead}_adduct.sdf"
            if generate_3d_and_save(adduct, adduct_path):
                output_files.append(('adduct', adduct_path))
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate warhead variants AND covalent adducts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available warheads:
{chr(10).join(f"  {k:18s} - {v['full_name']}: {v['description']}" for k, v in WARHEADS.items())}

Examples:
  # From SMILES
  python %(prog)s --smiles "Brc1ccccc1" --name benzyl --warhead acrylamide

  # From file
  python %(prog)s --input Inhib_32.sdf --warhead acrylamide

  # All warheads
  python %(prog)s --input Inhib_32.sdf --warhead all

Output files:
  - *_acrylamide.sdf         Pre-reaction (intact warhead, for positioning)
  - *_acrylamide_adduct.sdf  Covalent adduct (for docking/scoring)
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', nargs='+',
                             help='Input SDF/MOL file(s)')
    input_group.add_argument('--smiles', '-s',
                             help='Input SMILES string')
    
    parser.add_argument('--name', '-n', default='ligand',
                        help='Molecule name (required with --smiles)')
    parser.add_argument('--warhead', '-w', nargs='+', required=True,
                        help='Warhead type(s) or "all"')
    parser.add_argument('--outdir', '-o', default='warhead_ligands',
                        help='Output directory')
    
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
    print("WARHEAD + ADDUCT GENERATOR")
    print("="*60)
    print(f"Warheads: {', '.join(warhead_types)}")
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    all_outputs = []
    
    if args.smiles:
        outputs = process_smiles(args.smiles, args.name, warhead_types, outdir)
        all_outputs.extend(outputs)
    else:
        for input_file in args.input:
            outputs = process_file(input_file, warhead_types, outdir)
            all_outputs.extend(outputs)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nOutput directory: {outdir.resolve()}")
    print(f"Generated {len(all_outputs)} files:")
    for file_type, path in all_outputs:
        print(f"  [{file_type:12s}] {path}")
    
    print(f"\n" + "-"*60)
    print("USAGE NOTES:")
    print("-"*60)
    print("""
For covalent docking with GNINA or similar:

1. PRE-REACTION files (*_acrylamide.sdf):
   - Use for initial pose generation / non-covalent docking
   - Position the warhead near the target cysteine
   
2. ADDUCT files (*_acrylamide_adduct.sdf):
   - Use for covalent docking / scoring
   - The terminal -SH represents where Cys-Sγ attaches
   - For GNINA: dock this with flexible Cys sidechain
   - For GROMACS: build full adduct with protein Cys

For AutoDock4 covalent:
   - AD4 covalent expects the ligand to already have the
     reactive sulfur. Use the ADDUCT files.
   - The 'CovalentAtomIdx' property marks which S to use.
""")


if __name__ == "__main__":
    main()
