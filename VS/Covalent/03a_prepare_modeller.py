#!/usr/bin/env python3
"""
Prepare MODELLER inputs from pre-prepared PDB files.

This script:
1. Uses a local prepared PDB file (not downloaded)
2. Extracts UniProt accession codes from the PDB metadata (via RCSB API)
3. Downloads canonical FASTA sequences from UniProt
4. Identifies chains and their residue ranges
5. Detects internal gaps (missing loops)
6. Generates MODELLER-ready alignment file

Usage:
    python prepare_modeller_local.py 4CXA_prepared.pdb
    python prepare_modeller_local.py 5ACB_prepared.pdb
    
    # Or batch process multiple files:
    python prepare_modeller_local.py 4CXA_prepared.pdb 5ACB_prepared.pdb 5EFQ_prepared.pdb 7NXJ_prepared.pdb

Requirements:
    pip install requests

Output:
    {PDB_ID}_chains.fasta      - FASTA sequences for each chain
    {PDB_ID}_for_modeller.pdb  - Cleaned PDB with selected chains
    {PDB_ID}_loop.ali          - MODELLER alignment file
"""

import requests
import sys
import os
import re
import argparse
from collections import defaultdict
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

AA3TO1 = {
    # Standard amino acids
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    
    # Phosphorylated residues
    'SEP': 'S',  # Phosphoserine
    'TPO': 'T',  # Phosphothreonine
    'PTR': 'Y',  # Phosphotyrosine
    'PHD': 'D',  # Phosphoaspartate
    'HIP': 'H',  # Phosphohistidine
    
    # Selenomethionine
    'MSE': 'M',
    'SEC': 'C',  # Selenocysteine
    
    # Histidine protonation states
    'HSD': 'H',  # Delta-protonated
    'HSE': 'H',  # Epsilon-protonated
    'HSP': 'H',  # Doubly protonated (charged)
    'HID': 'H',  # AMBER delta
    'HIE': 'H',  # AMBER epsilon
    'HIP': 'H',  # AMBER protonated
    
    # Modified cysteines
    'CSO': 'C',  # S-hydroxycysteine
    'CSD': 'C',  # 3-sulfinoalanine
    'CSS': 'C',  # S-mercaptocysteine
    'CSX': 'C',  # S-oxy cysteine
    'CME': 'C',  # S,S-(2-hydroxyethyl)thiocysteine
    'CYM': 'C',  # Deprotonated cysteine (thiolate)
    'CYX': 'C',  # Cystine (disulfide bonded)
    
    # Modified lysines
    'MLY': 'K',  # N-dimethyl-lysine
    'MLZ': 'K',  # N-methyl-lysine
    'M3L': 'K',  # N-trimethyllysine
    'ALY': 'K',  # N-acetyllysine
    'KCX': 'K',  # Lysine NZ-carboxylic acid
    'LLP': 'K',  # Lysine-pyridoxal phosphate
    
    # Modified arginines
    'ARN': 'R',  # N-dimethylarginine
    'AGM': 'R',  # 5-methyl-arginine
    
    # Modified methionines
    'MHO': 'M',  # Methionine sulfoxide
    'SMC': 'C',  # S-methylcysteine
    
    # Modified prolines
    'HYP': 'P',  # Hydroxyproline
    
    # Modified glutamates/aspartates
    'CGU': 'E',  # Gamma-carboxyglutamic acid
    'PCA': 'E',  # Pyroglutamic acid (5-oxoproline)
    'GLZ': 'G',  # Amino-acetic acid
    
    # Other modifications
    'NEP': 'H',  # N1-phosphonohistidine
    'FME': 'M',  # N-formylmethionine
    'ORN': 'A',  # Ornithine (map to Ala as placeholder)
    'DAL': 'A',  # D-alanine
    'DVA': 'V',  # D-valine
    'AIB': 'A',  # Alpha-aminoisobutyric acid
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def extract_pdb_id(filename):
    """Extract PDB ID from filename like '4CXA_prepared.pdb'."""
    basename = Path(filename).stem  # Remove .pdb
    # Try to extract 4-character PDB ID
    match = re.match(r'^([A-Za-z0-9]{4})', basename)
    if match:
        return match.group(1).upper()
    return basename.upper()[:4]


def get_uniprot_mappings(pdb_id):
    """Get UniProt accession codes for each chain from RCSB API."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
    print(f"Fetching PDB metadata for {pdb_id}...")
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"  Warning: Could not fetch metadata for {pdb_id}")
        return {}
    
    entities = {}
    for entity_id in range(1, 20):
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
        
        instances = data.get('rcsb_polymer_entity_container_identifiers', {}).get('auth_asym_ids', [])
        entity_info['chains'] = instances
        
        uniprot_refs = data.get('rcsb_polymer_entity_container_identifiers', {}).get('uniprot_ids', [])
        if uniprot_refs:
            entity_info['uniprot'] = uniprot_refs[0]
        
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
    
    parts = header.split('|')
    if len(parts) >= 3:
        name_part = parts[2].split()[0] if parts[2] else uniprot_id
    else:
        name_part = uniprot_id
    
    return sequence, name_part


def parse_pdb_residues(pdb_file):
    """Parse PDB file and return residues present in each chain.
    
    Includes both ATOM and HETATM records for modified amino acids
    (e.g., TPO, SEP, PTR, MSE, etc.)
    """
    chain_residues = defaultdict(dict)
    
    # Modified residues that appear as HETATM but should be treated as amino acids
    modified_residues = set(AA3TO1.keys())
    
    hetatm_found = []  # Track all HETATM for debugging
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
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
                
            elif line.startswith("HETATM"):
                res_name = line[17:20].strip()
                atom_name = line[12:16].strip()
                
                # Debug: show all HETATM residues we encounter
                if res_name not in ['HOH', 'WAT'] and atom_name == "CA":
                    chain_id = line[21]
                    try:
                        res_num = int(line[22:26].strip())
                        hetatm_found.append(f"{res_name} chain {chain_id} res {res_num}")
                    except:
                        pass
                
                # Check if this is a modified amino acid
                if res_name in modified_residues:
                    if atom_name != "CA":
                        continue
                    
                    chain_id = line[21]
                    
                    try:
                        res_num = int(line[22:26].strip())
                    except ValueError:
                        continue
                    
                    print(f"  Including modified residue: {res_name} -> {AA3TO1.get(res_name, 'X')} at chain {chain_id} residue {res_num}")
                    chain_residues[chain_id][res_num] = res_name
    
    # Show what HETATM CA atoms were found
    if hetatm_found:
        print(f"\n  HETATM CA atoms found: {hetatm_found}")
    
    # Summary of what was found
    for chain_id in sorted(chain_residues.keys()):
        res_nums = sorted(chain_residues[chain_id].keys())
        if res_nums:
            print(f"  Chain {chain_id}: {len(res_nums)} residues ({res_nums[0]}-{res_nums[-1]})")
    
    return chain_residues


def find_internal_gaps(residues_dict):
    """Find gaps in residue numbering (missing loops)."""
    if not residues_dict:
        return [], None, None
    
    res_nums = sorted(residues_dict.keys())
    first_res = res_nums[0]
    last_res = res_nums[-1]
    
    gaps = []
    for i in range(len(res_nums) - 1):
        if res_nums[i + 1] - res_nums[i] > 1:
            gap_start = res_nums[i] + 1
            gap_end = res_nums[i + 1] - 1
            gaps.append((gap_start, gap_end))
    
    return gaps, first_res, last_res


def extract_chains_to_pdb(input_pdb, output_pdb, chains_config):
    """Extract specified chains to a new PDB file.
    
    Includes both ATOM and HETATM records for modified amino acids.
    HETATM records for modified amino acids are converted to ATOM records
    for MODELLER compatibility.
    """
    selected_chains = {c['chain_id'] for c in chains_config}
    chain_ranges = {c['chain_id']: (c['start_res'], c['end_res']) for c in chains_config}
    
    # Modified residues that should be included
    modified_residues = set(AA3TO1.keys())
    
    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith("ATOM"):
                chain_id = line[21]
                if chain_id in selected_chains:
                    try:
                        res_num = int(line[22:26].strip())
                        start, end = chain_ranges[chain_id]
                        if start <= res_num <= end:
                            f_out.write(line)
                    except ValueError:
                        f_out.write(line)
                        
            elif line.startswith("HETATM"):
                # Include modified amino acids (TPO, SEP, etc.)
                res_name = line[17:20].strip()
                chain_id = line[21]
                
                if res_name in modified_residues and chain_id in selected_chains:
                    try:
                        res_num = int(line[22:26].strip())
                        start, end = chain_ranges[chain_id]
                        if start <= res_num <= end:
                            # Convert HETATM to ATOM for MODELLER compatibility
                            # Also convert modified residue name to standard name
                            standard_res = AA3TO1.get(res_name, 'X')
                            # Map single letter back to three letter code
                            standard_res3 = {
                                'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
                                'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
                                'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
                                'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
                            }.get(standard_res, res_name)
                            
                            # Convert HETATM to ATOM and change residue name
                            new_line = "ATOM  " + line[6:17] + standard_res3 + line[20:]
                            f_out.write(new_line)
                            
                            # Print info about conversion (only for CA to reduce noise)
                            atom_name = line[12:16].strip()
                            if atom_name == "CA":
                                print(f"  Converted: HETATM {res_name} {res_num} -> ATOM {standard_res3}")
                    except ValueError:
                        f_out.write(line)
                        
            elif line.startswith("TER"):
                chain_id = line[21] if len(line) > 21 else ''
                if chain_id in selected_chains or chain_id == '':
                    f_out.write(line)
            elif line.startswith("END"):
                f_out.write(line)
    
    print(f"  Created: {output_pdb}")


def write_alignment(template_seqs, target_seqs, chains_config, modeller_pdb, ali_file, template_name, target_name):
    """Write MODELLER alignment file."""
    
    # Build chain/residue specification for MODELLER
    chain_spec_parts = []
    for config in chains_config:
        chain_spec_parts.append(f"FIRST:{config['chain_id']}")
    chain_spec = ", ".join(chain_spec_parts)
    
    end_spec_parts = []
    for config in chains_config:
        end_spec_parts.append(f"{config['end_res']}:{config['chain_id']}")
    end_spec = ", ".join(end_spec_parts)
    
    # Join sequences with '/' for multi-chain
    template_full = "/".join(template_seqs)
    target_full = "/".join(target_seqs)
    
    # Format sequences (80 chars per line)
    def format_seq(seq, width=75):
        lines = []
        for i in range(0, len(seq), width):
            lines.append(seq[i:i+width])
        return "\n".join(lines)
    
    pdb_basename = Path(modeller_pdb).name
    
    with open(ali_file, 'w') as f:
        # Template (structure)
        f.write(f">P1;{template_name}\n")
        f.write(f"structureX:{pdb_basename}:{chains_config[0]['start_res']}:{chains_config[0]['chain_id']}:{chains_config[-1]['end_res']}:{chains_config[-1]['chain_id']}::::\n")
        f.write(format_seq(template_full))
        f.write("*\n\n")
        
        # Target (to build)
        f.write(f">P1;{target_name}\n")
        f.write(f"sequence:{target_name}:{chains_config[0]['start_res']}:{chains_config[0]['chain_id']}:{chains_config[-1]['end_res']}:{chains_config[-1]['chain_id']}::::\n")
        f.write(format_seq(target_full))
        f.write("*\n")
    
    print(f"  Created: {ali_file}")


def process_pdb(pdb_file, interactive=True):
    """Process a single prepared PDB file."""
    
    pdb_path = Path(pdb_file)
    if not pdb_path.exists():
        print(f"Error: File not found: {pdb_file}")
        return False
    
    pdb_id = extract_pdb_id(pdb_file)
    print(f"\n{'='*70}")
    print(f"Processing: {pdb_file}")
    print(f"PDB ID: {pdb_id}")
    print(f"{'='*70}")
    
    # Step 1: Get UniProt mappings from RCSB
    print("\n[1/5] Fetching UniProt mappings...")
    entities = get_uniprot_mappings(pdb_id)
    
    if not entities:
        print("  Warning: Could not fetch entity information from RCSB")
        print("  You may need to provide UniProt IDs manually")
    
    # Step 2: Parse local PDB to find chains
    print("\n[2/5] Analyzing local PDB structure...")
    chain_residues = parse_pdb_residues(pdb_file)
    
    print(f"  Found chains: {', '.join(sorted(chain_residues.keys()))}")
    
    # Display entities and chains
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
    
    # Show chains in local PDB
    print("Chains in local PDB:")
    for chain_id in sorted(chain_residues.keys()):
        gaps, first_res, last_res = find_internal_gaps(chain_residues[chain_id])
        n_residues = len(chain_residues[chain_id])
        gap_info = f", {len(gaps)} gaps" if gaps else ""
        print(f"  Chain {chain_id}: residues {first_res}-{last_res} ({n_residues} resolved{gap_info})")
    
    # Get chains to include
    print("-" * 70)
    if interactive:
        chains_input = input("Enter chains to include (comma-separated, e.g., A,B): ").strip().upper()
    else:
        # Non-interactive: use all protein chains
        chains_input = ",".join(sorted(chain_residues.keys()))
        print(f"Using all chains: {chains_input}")
    
    selected_chains = [c.strip() for c in chains_input.split(",") if c.strip()]
    
    # Step 3: Download UniProt sequences
    print("\n[3/5] Downloading UniProt FASTA sequences...")
    
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
                    print(f"  Warning: No UniProt for chain {chain}")
    
    # Save FASTA
    fasta_file = f"{pdb_id}_chains.fasta"
    with open(fasta_file, 'w') as f:
        f.write(fasta_content)
    print(f"  Saved: {fasta_file}")
    
    # Step 4: Analyze gaps and prepare chains
    print("\n[4/5] Analyzing gaps...")
    
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
            print(f"    Internal gaps (missing loops):")
            for gs, ge in gaps:
                print(f"      {gs}-{ge} ({ge - gs + 1} residues)")
        else:
            print(f"    No internal gaps detected")
        
        # Adjust residue range if needed
        if interactive:
            print(f"\n    Current range: {first_res}-{last_res}")
            range_input = input(f"    Enter new range (or press Enter to keep): ").strip()
            
            if range_input:
                parts = range_input.split("-")
                start_res = int(parts[0])
                end_res = int(parts[1])
            else:
                start_res = first_res
                end_res = last_res
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
        return False
    
    # Step 5: Create output files
    print("\n[5/5] Creating MODELLER input files...")
    
    # Copy prepared PDB as modeller input
    modeller_pdb = f"{pdb_id}_for_modeller.pdb"
    extract_chains_to_pdb(pdb_file, modeller_pdb, chains_config)
    
    # Build sequences for alignment
    template_seqs = []
    target_seqs = []
    
    for config in chains_config:
        chain_id = config['chain_id']
        fasta = config['fasta']
        start_res = config['start_res']
        end_res = config['end_res']
        
        struct_residues = chain_residues.get(chain_id, {})
        
        # Template = what's ACTUALLY in the PDB (including mutations, modifications)
        # Target = canonical UniProt sequence (what we want to build)
        template_seq = ""
        target_seq = ""
        
        for res_num in range(start_res, end_res + 1):
            # Get canonical residue from FASTA (0-indexed)
            fasta_idx = res_num - 1
            if 0 <= fasta_idx < len(fasta):
                canonical_res = fasta[fasta_idx]
            else:
                canonical_res = 'X'
            
            if res_num in struct_residues:
                # Residue exists in structure - use ACTUAL PDB residue for template
                res3 = struct_residues[res_num]
                pdb_res = AA3TO1.get(res3, 'X')
                
                # Template gets what's actually in the PDB
                template_seq += pdb_res
                # Target gets the canonical sequence (MODELLER will mutate if different)
                target_seq += canonical_res
            else:
                # Residue missing in structure - gap in template, filled in target
                template_seq += "-"
                target_seq += canonical_res
        
        # Report mutations that MODELLER will make
        mutations = []
        for i, (t, s) in enumerate(zip(template_seq, target_seq)):
            if t != '-' and s != '-' and t != s:
                mutations.append((start_res + i, t, s))
        
        if mutations:
            print(f"\n  Chain {chain_id} - MODELLER will mutate these residues:")
            for res_num, pdb_res, canon_res in mutations[:10]:
                print(f"    Residue {res_num}: {pdb_res} (in PDB) -> {canon_res} (canonical)")
            if len(mutations) > 10:
                print(f"    ... and {len(mutations) - 10} more")
        
        # Verify lengths match
        if len(template_seq) != len(target_seq):
            print(f"  ERROR: Length mismatch for chain {chain_id}")
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
    
    # Summary
    print("\n" + "=" * 70)
    print(f"COMPLETE: {pdb_id}")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  {fasta_file:<30s} - UniProt FASTA sequences")
    print(f"  {modeller_pdb:<30s} - PDB for MODELLER")
    print(f"  {ali_file:<30s} - MODELLER alignment file")
    
    print(f"\nChains processed:")
    for config in chains_config:
        print(f"  Chain {config['chain_id']}: {config['name']} (residues {config['start_res']}-{config['end_res']})")
    
    print(f"\nTo run MODELLER:")
    print(f"  python model_loops.py {pdb_id}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MODELLER inputs from pre-prepared PDB files"
    )
    parser.add_argument(
        "pdb_files",
        nargs="+",
        help="Pre-prepared PDB files (e.g., 4CXA_prepared.pdb)"
    )
    parser.add_argument(
        "--non-interactive", "-n",
        action="store_true",
        help="Run without prompts (use all chains, default ranges)"
    )
    args = parser.parse_args()
    
    interactive = not args.non_interactive
    
    for pdb_file in args.pdb_files:
        try:
            process_pdb(pdb_file, interactive=interactive)
        except Exception as e:
            print(f"\nError processing {pdb_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 70)
    print("ALL FILES PROCESSED")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the generated alignment files (*_loop.ali)")
    print("  2. Run MODELLER for each structure:")
    print("     python model_loops.py <PDB_ID>")
    print("  3. Select best models (lowest DOPE score)")


if __name__ == "__main__":
    main()
