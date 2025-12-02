#!/usr/bin/env python3
"""
Automated MODELLER input preparation from PDB ID.

This script:
1. Downloads a PDB structure from RCSB
2. Extracts UniProt accession codes from the PDB metadata
3. Downloads canonical FASTA sequences from UniProt
4. Identifies chains and their residue ranges
5. Detects internal gaps (missing loops)
6. Generates MODELLER-ready alignment file and PDB

Usage:
    python prepare_modeller_from_pdb.py
    
    Then enter the PDB ID when prompted (e.g., 8WU1)

Requirements:
    pip install requests

Output:
    {PDB_ID}_chains.fasta      - FASTA sequences for each chain
    {PDB_ID}_for_modeller.pdb  - Cleaned PDB with selected chains
    {PDB_ID}_loop.ali          - MODELLER alignment file
    model_loops.py             - MODELLER script (ready to run)
"""

import requests
import sys
import os
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

# Three letter to one letter amino acid code
AA3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # Modified residues
    'SEP': 'S', 'TPO': 'T', 'PTR': 'Y',  # Phosphorylated
    'MSE': 'M',  # Selenomethionine
    'HSD': 'H', 'HSE': 'H', 'HSP': 'H',  # Histidine variants
    'CSO': 'C', 'CSD': 'C',  # Modified cysteines
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def download_pdb(pdb_id):
    """Download PDB file from RCSB. Try PDB format first, then mmCIF."""
    pdb_id = pdb_id.upper()
    
    # Try PDB format first
    url_pdb = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Downloading PDB from {url_pdb}...")
    
    response = requests.get(url_pdb)
    if response.status_code == 200:
        filename = f"{pdb_id}.pdb"
        with open(filename, 'w') as f:
            f.write(response.text)
        print(f"  Saved: {filename}")
        return filename, 'pdb'
    
    # If PDB format not available, try mmCIF
    print(f"  PDB format not available, trying mmCIF...")
    url_cif = f"https://files.rcsb.org/download/{pdb_id}.cif"
    print(f"Downloading mmCIF from {url_cif}...")
    
    response = requests.get(url_cif)
    if response.status_code == 200:
        filename = f"{pdb_id}.cif"
        with open(filename, 'w') as f:
            f.write(response.text)
        print(f"  Saved: {filename}")
        return filename, 'cif'
    
    raise Exception(f"Failed to download structure {pdb_id} in either PDB or mmCIF format")


def get_uniprot_mappings(pdb_id):
    """Get UniProt accession codes for each chain from RCSB API."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
    print(f"Fetching PDB metadata...")
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch PDB metadata: {response.status_code}")
    
    # Get polymer entities
    url_poly = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}"
    
    # Get all polymer entities
    entities = {}
    for entity_id in range(1, 20):  # Check up to 20 entities
        url_entity = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/{entity_id}"
        resp = requests.get(url_entity)
        if resp.status_code != 200:
            break
        
        data = resp.json()
        entity_info = {
            'entity_id': entity_id,
            'description': data.get('rcsb_polymer_entity', {}).get('pdbx_description', 'Unknown'),
            'chains': [],
            'uniprot': None,
            'uniprot_name': None,
        }
        
        # Get chain mappings
        instances = data.get('rcsb_polymer_entity_container_identifiers', {}).get('auth_asym_ids', [])
        entity_info['chains'] = instances
        
        # Get UniProt reference
        uniprot_refs = data.get('rcsb_polymer_entity_container_identifiers', {}).get('uniprot_ids', [])
        if uniprot_refs:
            entity_info['uniprot'] = uniprot_refs[0]
        
        # Also check reference_sequence_identifiers for UniProt
        ref_seq = data.get('rcsb_polymer_entity_align', [])
        for ref in ref_seq:
            if ref.get('reference_database_name') == 'UniProt':
                entity_info['uniprot'] = ref.get('reference_database_accession')
                break
        
        entities[entity_id] = entity_info
    
    return entities


def download_uniprot_fasta(uniprot_id):
    """Download FASTA sequence from UniProt."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    print(f"  Downloading UniProt {uniprot_id}...")
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"    Warning: Could not download UniProt {uniprot_id}")
        return None, None
    
    lines = response.text.strip().split('\n')
    header = lines[0]
    sequence = ''.join(lines[1:])
    
    # Parse header for protein name
    # Format: >sp|P21554|CNR1_HUMAN Cannabinoid receptor 1 OS=Homo sapiens ...
    parts = header.split('|')
    if len(parts) >= 3:
        name_part = parts[2].split()[0] if parts[2] else uniprot_id
    else:
        name_part = uniprot_id
    
    return sequence, name_part


def parse_pdb_residues(pdb_file):
    """Parse PDB file and return residues present in each chain."""
    chain_residues = defaultdict(dict)
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                
                res_name = line[17:20].strip()
                chain_id = line[21]
                
                try:
                    res_num = int(line[22:26].strip())
                except ValueError:
                    continue
                
                chain_residues[chain_id][res_num] = res_name
    
    return chain_residues


def parse_cif_residues(cif_file):
    """Parse mmCIF file and return residues present in each chain."""
    chain_residues = defaultdict(dict)
    
    # mmCIF ATOM record columns (space-separated):
    # 0:  group_PDB (ATOM/HETATM)
    # 1:  id (atom serial)
    # 2:  type_symbol (element)
    # 3:  label_atom_id (atom name)
    # 4:  label_alt_id (alt loc)
    # 5:  label_comp_id (residue name)
    # 6:  label_asym_id (internal chain)
    # 7:  label_entity_id
    # 8:  label_seq_id (internal residue number)
    # 9:  pdbx_PDB_ins_code
    # 10: Cartn_x
    # 11: Cartn_y
    # 12: Cartn_z
    # 13: occupancy
    # 14: B_iso_or_equiv
    # 15: pdbx_formal_charge
    # 16: auth_seq_id (author residue number) <- USE THIS
    # 17: auth_comp_id (author residue name)
    # 18: auth_asym_id (author chain ID) <- USE THIS
    # 19: auth_atom_id
    # 20: pdbx_PDB_model_num
    
    with open(cif_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                if len(parts) < 19:
                    continue
                
                atom_name = parts[3]  # label_atom_id
                if atom_name != "CA":
                    continue
                
                res_name = parts[5]   # label_comp_id
                
                # Use AUTHOR chain ID and residue number (what you see in viewers)
                try:
                    chain_id = parts[18]  # auth_asym_id
                    res_num = int(parts[16])  # auth_seq_id
                except (ValueError, IndexError):
                    # Fallback to label columns if auth columns fail
                    try:
                        chain_id = parts[6]   # label_asym_id
                        res_num = int(parts[8])  # label_seq_id
                    except (ValueError, IndexError):
                        continue
                
                chain_residues[chain_id][res_num] = res_name
    
    return chain_residues


def parse_structure_residues(structure_file, file_format):
    """Parse structure file (PDB or mmCIF) and return residues."""
    if file_format == 'pdb':
        return parse_pdb_residues(structure_file)
    elif file_format == 'cif':
        return parse_cif_residues(structure_file)
    else:
        raise ValueError(f"Unknown file format: {file_format}")


def find_internal_gaps(residue_dict):
    """Find internal gaps in a residue dictionary."""
    if not residue_dict:
        return [], None, None
    
    residues = sorted(residue_dict.keys())
    first_res = min(residues)
    last_res = max(residues)
    
    gaps = []
    in_gap = False
    gap_start = None
    
    for i in range(first_res, last_res + 1):
        if i not in residue_dict:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                gaps.append((gap_start, i - 1))
                in_gap = False
    
    return gaps, first_res, last_res


def extract_chains_to_pdb(input_file, output_pdb, chains_config, file_format):
    """Extract specific chains and residue ranges from structure file, output as PDB."""
    # Build a lookup of chain_id -> (start_res, end_res)
    chain_ranges = {c['chain_id']: (c['start_res'], c['end_res']) for c in chains_config}
    chains_to_keep = set(chain_ranges.keys())
    
    with open(input_file, 'r') as f_in, open(output_pdb, 'w') as f_out:
        if file_format == 'pdb':
            for line in f_in:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    chain_id = line[21]
                    if chain_id in chains_to_keep:
                        try:
                            res_num = int(line[22:26].strip())
                        except ValueError:
                            continue
                        
                        start_res, end_res = chain_ranges[chain_id]
                        if start_res <= res_num <= end_res:
                            f_out.write(line)
                elif line.startswith("END"):
                    f_out.write(line)
        
        elif file_format == 'cif':
            # Convert mmCIF to PDB format
            # mmCIF columns: see parse_cif_residues for full list
            atom_num = 0
            for line in f_in:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    parts = line.split()
                    if len(parts) < 19:
                        continue
                    
                    # Use author chain ID (position 18)
                    chain_id = parts[18]  # auth_asym_id
                    if chain_id not in chains_to_keep:
                        continue
                    
                    try:
                        res_num = int(parts[16])  # auth_seq_id
                    except (ValueError, IndexError):
                        continue
                    
                    start_res, end_res = chain_ranges[chain_id]
                    if not (start_res <= res_num <= end_res):
                        continue
                    
                    # Extract fields for PDB format
                    atom_num += 1
                    record = parts[0]
                    atom_name = parts[3]  # label_atom_id
                    res_name = parts[5]   # label_comp_id
                    x = float(parts[10])
                    y = float(parts[11])
                    z = float(parts[12])
                    occ = float(parts[13])
                    bfac = float(parts[14])
                    element = parts[2]
                    
                    # Format atom name
                    if len(atom_name) == 4:
                        atom_field = atom_name
                    else:
                        atom_field = f" {atom_name:<3}"
                    
                    pdb_line = f"{record:<6}{atom_num:>5} {atom_field} {res_name:<3} {chain_id}{res_num:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{bfac:>6.2f}          {element:>2}\n"
                    f_out.write(pdb_line)
            
            f_out.write("END\n")
    
    print(f"  Saved: {output_pdb}")


def write_alignment(template_seqs, target_seqs, chains_config, 
                    pdb_file, ali_file, template_name, target_name):
    """Write PIR-format alignment file."""
    
    first_chain = chains_config[0]
    last_chain = chains_config[-1]
    
    template_full = "/".join(template_seqs)
    target_full = "/".join(target_seqs)
    
    with open(ali_file, 'w') as f:
        # Template entry
        f.write(f">P1;{template_name}\n")
        f.write(f"structureX:{pdb_file}:{first_chain['start_res']}:{first_chain['chain_id']}:{last_chain['end_res']}:{last_chain['chain_id']}::::\n")
        
        for i in range(0, len(template_full), 75):
            chunk = template_full[i:i+75]
            if i + 75 >= len(template_full):
                chunk += "*"
            f.write(chunk + "\n")
        
        f.write("\n")
        
        # Target entry
        f.write(f">P1;{target_name}\n")
        f.write(f"sequence:{target_name}::::::::\n")
        
        for i in range(0, len(target_full), 75):
            chunk = target_full[i:i+75]
            if i + 75 >= len(target_full):
                chunk += "*"
            f.write(chunk + "\n")
    
    print(f"  Saved: {ali_file}")


def write_modeller_script(pdb_id, ali_file, template_name, target_name):
    """Write the MODELLER loop modeling script."""
    script = f'''#!/usr/bin/env python3
"""
MODELLER script for loop modeling.

Usage:
    python model_loops.py <PDB_ID>
    
Example:
    python model_loops.py 8WU1

Output:
    <PDB_ID>_fill.B99990001.pdb (and more models based on ending_model setting)
"""

import sys
from modeller import *
from modeller.automodel import *

# Get PDB ID from command line or prompt
if len(sys.argv) > 1:
    pdb_id = sys.argv[1].strip().upper().replace(".PDB", "").replace(".CIF", "")
else:
    pdb_id = input("Enter PDB ID: ").strip().upper().replace(".PDB", "").replace(".CIF", "")

if not pdb_id:
    print("Error: No PDB ID provided")
    sys.exit(1)

# File names based on PDB ID
ali_file = f"{{pdb_id}}_loop.ali"
modeller_pdb = f"{{pdb_id}}_for_modeller.pdb"
template_name = f"{{pdb_id}}_template"
target_name = f"{{pdb_id}}_fill"

print(f"Running MODELLER for {{pdb_id}}...")
print(f"  Alignment file: {{ali_file}}")
print(f"  Template PDB: {{modeller_pdb}}")

# Set up environment
env = Environ()
env.io.atom_files_directory = ['.']

# Use AutoModel to build models with loops filled
a = AutoModel(env,
              alnfile=ali_file,
              knowns=template_name,
              sequence=target_name,
              assess_methods=(assess.DOPE, assess.GA341))

# Generate multiple models to sample different loop conformations
a.starting_model = 1
a.ending_model = 5  # Generate 5 models - increase for better sampling

# Optimization settings
a.library_schedule = autosched.slow
a.max_var_iterations = 300

# MD refinement settings
a.md_level = refine.slow

# Build models
a.make()

# Print summary
print("\\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
for model in a.outputs:
    if model['failure'] is None:
        print(f"{{model['name']:30s}} DOPE: {{model['DOPE score']:10.3f}}  GA341: {{model['GA341 score'][0]:6.3f}}")
print("="*60)
print("\\nLower DOPE score = better model")
print("Select the model with the lowest DOPE score for your MD simulation")
'''
    
    with open('model_loops.py', 'w') as f:
        f.write(script)
    
    print(f"  Saved: model_loops.py")


def main():
    print("=" * 70)
    print("MODELLER Input Preparation from PDB")
    print("=" * 70)
    
    # Get PDB ID from command line or prompt
    if len(sys.argv) > 1:
        pdb_id = sys.argv[1].strip().upper()
    else:
        pdb_id = input("\nEnter PDB ID: ").strip().upper()
    
    # Clean up input - remove common extensions and whitespace
    pdb_id = pdb_id.replace(".PDB", "").replace(".CIF", "").replace(".ENT", "").strip()
    
    if not pdb_id:
        print("Error: No PDB ID provided")
        sys.exit(1)
    
    # Validate PDB ID format (4 characters, alphanumeric)
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        print(f"Warning: '{pdb_id}' doesn't look like a standard PDB ID (4 alphanumeric characters)")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            sys.exit(1)
    
    print(f"\nProcessing {pdb_id}...")
    
    # Step 1: Download PDB
    print("\n[1/6] Downloading PDB structure...")
    structure_file, file_format = download_pdb(pdb_id)
    
    # Step 2: Get UniProt mappings
    print("\n[2/6] Fetching UniProt mappings...")
    entities = get_uniprot_mappings(pdb_id)
    
    # Display entities and let user select
    print("\n" + "-" * 70)
    print("Available chains:")
    print("-" * 70)
    
    for eid, info in entities.items():
        chains_str = ", ".join(info['chains'])
        uniprot_str = info['uniprot'] if info['uniprot'] else "No UniProt"
        print(f"  Entity {eid}: {info['description'][:40]:<40s}")
        print(f"           Chains: {chains_str}")
        print(f"           UniProt: {uniprot_str}")
        print()
    
    # Get chains to include
    print("-" * 70)
    chains_input = input("Enter chains to include (comma-separated, e.g., A,D): ").strip().upper()
    selected_chains = [c.strip() for c in chains_input.split(",")]
    
    # Step 3: Download UniProt sequences
    print("\n[3/6] Downloading UniProt FASTA sequences...")
    
    chain_info = {}
    fasta_content = ""
    
    for eid, info in entities.items():
        for chain in info['chains']:
            if chain in selected_chains:
                if info['uniprot']:
                    seq, name = download_uniprot_fasta(info['uniprot'])
                    if seq:
                        chain_info[chain] = {
                            'uniprot': info['uniprot'],
                            'name': name,
                            'description': info['description'],
                            'fasta': seq,
                        }
                        fasta_content += f">{chain}|{info['uniprot']}|{name} {info['description']}\n{seq}\n\n"
                else:
                    print(f"  Warning: No UniProt for chain {chain}, skipping...")
    
    # Save FASTA
    fasta_file = f"{pdb_id}_chains.fasta"
    with open(fasta_file, 'w') as f:
        f.write(fasta_content)
    print(f"  Saved: {fasta_file}")
    
    # Step 4: Parse PDB and find gaps
    print("\n[4/6] Analyzing structure for gaps...")
    chain_residues = parse_structure_residues(structure_file, file_format)
    
    chains_config = []
    
    for chain in selected_chains:
        if chain not in chain_info:
            print(f"  Warning: No FASTA for chain {chain}, skipping...")
            continue
        
        if chain not in chain_residues:
            print(f"  Warning: Chain {chain} not found in PDB, skipping...")
            continue
        
        gaps, first_res, last_res = find_internal_gaps(chain_residues[chain])
        
        print(f"\n  Chain {chain} ({chain_info[chain]['name']}):")
        print(f"    Residue range in structure: {first_res}-{last_res}")
        print(f"    FASTA length: {len(chain_info[chain]['fasta'])}")
        
        if gaps:
            print(f"    Internal gaps:")
            for gs, ge in gaps:
                print(f"      {gs}-{ge} ({ge - gs + 1} residues)")
        else:
            print(f"    No internal gaps")
        
        # Ask user to confirm/adjust residue range
        print(f"\n    Current range: {first_res}-{last_res}")
        range_input = input(f"    Enter new range (or press Enter to keep): ").strip()
        
        if range_input:
            parts = range_input.split("-")
            start_res = int(parts[0])
            end_res = int(parts[1])
        else:
            start_res = first_res
            end_res = last_res
        
        chains_config.append({
            'chain_id': chain,
            'name': chain_info[chain]['name'],
            'fasta': chain_info[chain]['fasta'],
            'start_res': start_res,
            'end_res': end_res,
        })
    
    if not chains_config:
        print("Error: No valid chains to process")
        sys.exit(1)
    
    # Step 5: Create output files
    print("\n[5/6] Creating MODELLER input files...")
    
    # Extract selected chains to new PDB (with correct residue ranges)
    modeller_pdb = f"{pdb_id}_for_modeller.pdb"
    extract_chains_to_pdb(structure_file, modeller_pdb, chains_config, file_format)
    
    # Build sequences for alignment
    template_seqs = []
    target_seqs = []
    
    for config in chains_config:
        chain_id = config['chain_id']
        fasta = config['fasta']
        start_res = config['start_res']
        end_res = config['end_res']
        
        # Get residues actually in the structure
        struct_residues = chain_residues.get(chain_id, {})
        
        # Build BOTH sequences based on what's in the structure
        # Template = what's actually in PDB (with gaps for missing residues)
        # Target = canonical UniProt sequence (to revert mutations)
        template_seq = ""
        target_seq = ""
        
        for res_num in range(start_res, end_res + 1):
            # Get canonical residue from FASTA (0-indexed)
            fasta_idx = res_num - 1
            if fasta_idx < len(fasta):
                canonical_res = fasta[fasta_idx]
            else:
                canonical_res = 'X'
            
            if res_num in struct_residues:
                # Residue exists in structure - use actual PDB residue for template
                res3 = struct_residues[res_num]
                pdb_res = AA3TO1.get(res3, 'X')
                template_seq += pdb_res
                target_seq += canonical_res  # Use canonical for target (reverts mutations)
            else:
                # Residue missing in structure - gap in both
                template_seq += "-"
                target_seq += canonical_res  # Fill with canonical sequence
        
        # Report any mutations that will be reverted
        mutations = []
        for i, (t, s) in enumerate(zip(template_seq, target_seq)):
            if t != '-' and s != '-' and t != s:
                mutations.append((start_res + i, t, s))
        
        if mutations:
            print(f"\n  Chain {chain_id} mutations to revert ({len(mutations)} total):")
            for res_num, pdb_res, canon_res in mutations[:10]:
                print(f"    {res_num}: {pdb_res} -> {canon_res}")
            if len(mutations) > 10:
                print(f"    ... and {len(mutations) - 10} more")
        
        # Verify lengths
        if len(template_seq) != len(target_seq):
            print(f"  Warning: Length mismatch for chain {chain_id}")
            print(f"    Template: {len(template_seq)}, Target: {len(target_seq)}")
            min_len = min(len(template_seq), len(target_seq))
            template_seq = template_seq[:min_len]
            target_seq = target_seq[:min_len]
        
        template_seqs.append(template_seq)
        target_seqs.append(target_seq)
    
    # Write alignment file
    ali_file = f"{pdb_id}_loop.ali"
    template_name = f"{pdb_id}_template"
    target_name = f"{pdb_id}_fill"
    
    write_alignment(template_seqs, target_seqs, chains_config,
                    modeller_pdb, ali_file, template_name, target_name)
    
    # Step 6: Write MODELLER script
    print("\n[6/6] Creating MODELLER script...")
    write_modeller_script(pdb_id, ali_file, template_name, target_name)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  {fasta_file:<30s} - UniProt FASTA sequences")
    print(f"  {modeller_pdb:<30s} - Cleaned PDB for MODELLER")
    print(f"  {ali_file:<30s} - MODELLER alignment file")
    print(f"  {'model_loops.py':<30s} - MODELLER script")
    
    print(f"\nChains processed:")
    for config in chains_config:
        print(f"  Chain {config['chain_id']}: {config['name']} (residues {config['start_res']}-{config['end_res']})")
    
    print(f"\nTo run MODELLER:")
    print(f"  python model_loops.py")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
