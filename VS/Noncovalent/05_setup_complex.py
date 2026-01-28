#!/usr/bin/env python3
"""
GROMACS System Setup for Protein-Ligand Complex

Builds a solvated, charge-neutralized system ready for MD simulation using
AMBER99SB-ILDN force field for the protein and GAFF2 for the ligand.

Workflow:
1. Process receptor with pdb2gmx (AMBER99SB-ILDN)
2. Combine receptor and ligand coordinates
3. Create simulation box
4. Solvate with TIP3P water
5. Add ions for charge neutralization
6. Generate position restraints
7. Perform energy minimization

Dependencies:
    - GROMACS (gmx)
    - Output from 04_parameterize_ligand.py

Usage:
    python 05_setup_complex.py \\
        --receptor RECEPTOR.pdb \\
        --ligand_itp LIGAND.itp \\
        --ligand_gro LIGAND.gro \\
        --output_dir OUTPUT [options]

Examples:
    # Basic setup
    python 05_setup_complex.py \\
        --receptor Outputs/docking/4CXA_predock.pdb \\
        --ligand_itp Outputs/NonCovalent/params/Inhib_42/LIG.itp \\
        --ligand_gro Outputs/NonCovalent/params/Inhib_42/LIG.gro \\
        --output_dir Outputs/NonCovalent/complex/4CXA_Inhib_42

    # With specific box size
    python 05_setup_complex.py \\
        --receptor receptor.pdb \\
        --ligand_itp LIG.itp \\
        --ligand_gro LIG.gro \\
        --docked_pose docked_best.sdf \\
        --output_dir complex \\
        --box_distance 1.2
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
import re


def find_gmx():
    """Find GROMACS executable."""
    for cmd in ['gmx', 'gmx_mpi']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    return None


def run_command(cmd, description, cwd=None, input_text=None, verbose=True):
    """Execute a shell command and handle errors."""
    if verbose:
        print(f"  {description}...")
        if isinstance(cmd, list):
            print(f"    $ {' '.join(cmd)}")

    # Stream output directly to terminal for real-time feedback
    if input_text:
        # Need to use PIPE for stdin when providing input
        result = subprocess.run(
            cmd,
            text=True,
            cwd=cwd,
            input=input_text,
            shell=isinstance(cmd, str)
        )
    else:
        # No input needed, just run and let output go to terminal
        result = subprocess.run(
            cmd,
            text=True,
            cwd=cwd,
            shell=isinstance(cmd, str)
        )

    if result.returncode != 0:
        print(f"  ERROR: {description} failed!")
        return False, result

    if verbose:
        print(f"  Done")
    return True, result


def extract_ligand_position_from_sdf(sdf_path):
    """
    Extract ligand coordinates from SDF file.
    Returns list of (atom, x, y, z) tuples.
    """
    coords = []
    with open(sdf_path) as f:
        lines = f.readlines()

    # SDF format: line 4 has counts, then atom block
    if len(lines) < 5:
        return None

    # Parse counts line
    counts = lines[3].split()
    if len(counts) < 2:
        return None
    n_atoms = int(counts[0])

    # Parse atom coordinates
    for i in range(4, 4 + n_atoms):
        if i >= len(lines):
            break
        parts = lines[i].split()
        if len(parts) >= 4:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            atom = parts[3]
            coords.append((atom, x, y, z))

    return coords


def update_ligand_gro_coordinates(ligand_gro, docked_sdf, output_gro):
    """
    Update ligand GRO coordinates to match docked pose from SDF.

    This aligns the parameterized ligand structure to the docked position.
    """
    # Read docked coordinates
    docked_coords = extract_ligand_position_from_sdf(docked_sdf)
    if docked_coords is None:
        print("  WARNING: Could not read docked coordinates")
        shutil.copy(ligand_gro, output_gro)
        return True

    # Read original GRO
    with open(ligand_gro) as f:
        lines = f.readlines()

    title = lines[0]
    n_atoms = int(lines[1].strip())
    box = lines[-1]

    # Update coordinates (GRO format: positions in nm, SDF in Angstroms)
    new_lines = [title, lines[1]]

    for i, line in enumerate(lines[2:-1]):
        if i >= len(docked_coords):
            new_lines.append(line)
            continue

        # Parse GRO line
        resnum = line[0:5]
        resname = line[5:10]
        atomname = line[10:15]
        atomnum = line[15:20]

        # Get new coordinates (convert A to nm)
        _, x, y, z = docked_coords[i]
        x_nm, y_nm, z_nm = x / 10.0, y / 10.0, z / 10.0

        # Format new line
        new_line = f"{resnum}{resname}{atomname}{atomnum}{x_nm:8.3f}{y_nm:8.3f}{z_nm:8.3f}\n"
        new_lines.append(new_line)

    new_lines.append(box)

    with open(output_gro, 'w') as f:
        f.writelines(new_lines)

    return True


def run_pdb2gmx(gmx, pdb_input, gro_output, top_output, work_dir, ff='amber99sb-ildn', water='tip3p'):
    """
    Run pdb2gmx to generate topology for protein.
    """
    # Convert to absolute paths since we run with cwd=work_dir
    pdb_input = Path(pdb_input).resolve()
    gro_output = Path(gro_output).resolve()
    top_output = Path(top_output).resolve()
    posre_output = (work_dir / 'posre.itp').resolve()

    cmd = [
        gmx, 'pdb2gmx',
        '-f', str(pdb_input),
        '-o', str(gro_output),
        '-p', str(top_output),
        '-i', str(posre_output),  # Position restraints file
        '-ff', ff,
        '-water', water,
        '-ignh'  # Ignore hydrogens in input
    ]

    return run_command(cmd, f"Running pdb2gmx with {ff} force field", cwd=str(work_dir))


def combine_receptor_ligand(receptor_gro, ligand_gro, output_gro, ligand_resname='LIG'):
    """
    Combine receptor and ligand GRO files.
    """
    # Read receptor
    with open(receptor_gro) as f:
        rec_lines = f.readlines()

    rec_title = rec_lines[0].strip()
    rec_n_atoms = int(rec_lines[1].strip())
    rec_atoms = rec_lines[2:2+rec_n_atoms]
    rec_box = rec_lines[-1]

    # Read ligand
    with open(ligand_gro) as f:
        lig_lines = f.readlines()

    lig_n_atoms = int(lig_lines[1].strip())
    lig_atoms = lig_lines[2:2+lig_n_atoms]

    # Renumber ligand atoms
    new_lig_atoms = []
    for i, line in enumerate(lig_atoms):
        # Update residue number and atom number
        new_resnum = rec_n_atoms // 10 + 2  # After protein residues
        new_atomnum = rec_n_atoms + i + 1

        # Parse and reformat
        old_resname = line[5:10].strip()
        atomname = line[10:15]
        coords = line[20:]  # Keep coordinates and velocities

        new_line = f"{new_resnum:5d}{ligand_resname:>5s}{atomname}{new_atomnum:5d}{coords}"
        new_lig_atoms.append(new_line)

    # Write combined GRO
    total_atoms = rec_n_atoms + lig_n_atoms
    with open(output_gro, 'w') as f:
        f.write(f"Complex: {rec_title}\n")
        f.write(f"{total_atoms}\n")
        f.writelines(rec_atoms)
        f.writelines(new_lig_atoms)
        f.write(rec_box)

    return True


def process_ligand_itp(ligand_itp, output_dir, resname='LIG'):
    """
    Process ligand ITP file from ACPYPE.

    ACPYPE includes [ atomtypes ] in the molecule ITP, but GROMACS requires
    atomtypes to be defined before any [ moleculetype ] sections. This function:
    1. Extracts [ atomtypes ] to a separate file
    2. Creates a cleaned ITP without atomtypes

    Returns:
        tuple: (atomtypes_itp_path, cleaned_itp_path) or (None, original_itp) if no atomtypes
    """
    with open(ligand_itp) as f:
        content = f.read()

    # Check if atomtypes section exists
    if '[ atomtypes ]' not in content:
        return None, ligand_itp

    lines = content.split('\n')
    atomtypes_lines = []
    cleaned_lines = []
    in_atomtypes = False

    for line in lines:
        stripped = line.strip()

        # Detect start of atomtypes section
        if stripped == '[ atomtypes ]':
            in_atomtypes = True
            atomtypes_lines.append(line)
            continue

        # Detect end of atomtypes section (next section starts)
        if in_atomtypes and stripped.startswith('[') and 'atomtypes' not in stripped:
            in_atomtypes = False

        if in_atomtypes:
            atomtypes_lines.append(line)
        else:
            cleaned_lines.append(line)

    # Remove leading empty lines from cleaned content
    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)

    # Write atomtypes file
    atomtypes_itp = output_dir / f'{resname}_atomtypes.itp'
    with open(atomtypes_itp, 'w') as f:
        f.write('; Ligand atom types (extracted from ACPYPE output)\n')
        f.write('\n'.join(atomtypes_lines))
        f.write('\n')

    # Write cleaned ligand ITP
    cleaned_itp = output_dir / f'{resname}.itp'
    with open(cleaned_itp, 'w') as f:
        f.write('; Ligand topology (atomtypes in separate file)\n')
        f.write('\n'.join(cleaned_lines))

    print(f"  Extracted atomtypes to: {atomtypes_itp}")
    print(f"  Cleaned ligand ITP: {cleaned_itp}")

    return atomtypes_itp, cleaned_itp


def create_complex_topology(receptor_top, ligand_itp, output_top, ligand_resname='LIG',
                            atomtypes_itp=None):
    """
    Create combined topology file including ligand.

    Args:
        receptor_top: Path to receptor topology from pdb2gmx
        ligand_itp: Path to ligand ITP (should not contain atomtypes)
        output_top: Output topology path
        ligand_resname: Ligand residue name
        atomtypes_itp: Path to atomtypes ITP (optional, if extracted separately)
    """
    with open(receptor_top) as f:
        top_content = f.read()

    # If we have atomtypes, include them after forcefield.itp
    if atomtypes_itp:
        atomtypes_include = f'\n; Include ligand atom types\n#include "{atomtypes_itp}"\n'

        # Find the forcefield include and insert after it
        # Look for #include "...forcefield.itp" or similar
        ff_match = re.search(r'(#include\s+["\'].*?forcefield\.itp["\'].*?\n)', top_content)
        if ff_match:
            insert_pos = ff_match.end()
            top_content = top_content[:insert_pos] + atomtypes_include + top_content[insert_pos:]
        else:
            # Alternative: insert after the first #include
            first_include = re.search(r'(#include\s+["\'].*?["\'].*?\n)', top_content)
            if first_include:
                insert_pos = first_include.end()
                top_content = top_content[:insert_pos] + atomtypes_include + top_content[insert_pos:]

    # Add ligand ITP include before [ system ]
    itp_include = f'\n; Include ligand topology\n#include "{ligand_itp}"\n'

    # Find [ system ] section
    system_match = re.search(r'\n\[ system \]', top_content)
    if system_match:
        insert_pos = system_match.start()
        top_content = top_content[:insert_pos] + itp_include + top_content[insert_pos:]
    else:
        # Append before molecules
        mol_match = re.search(r'\n\[ molecules \]', top_content)
        if mol_match:
            insert_pos = mol_match.start()
            top_content = top_content[:insert_pos] + itp_include + top_content[insert_pos:]

    # Add ligand to [ molecules ] section
    # Find the molecules section and add ligand
    if '[ molecules ]' in top_content:
        # Add ligand at the end of molecules
        top_content = top_content.rstrip() + f"\n{ligand_resname}                 1\n"
    else:
        top_content += f"\n[ molecules ]\nProtein_chain_A     1\n{ligand_resname}                 1\n"

    with open(output_top, 'w') as f:
        f.write(top_content)

    return True


def create_box(gmx, input_gro, output_gro, box_type='dodecahedron', distance=1.0):
    """
    Create simulation box around the complex.
    """
    cmd = [
        gmx, 'editconf',
        '-f', str(input_gro),
        '-o', str(output_gro),
        '-bt', box_type,
        '-d', str(distance),
        '-c'  # Center the molecule
    ]

    return run_command(cmd, f"Creating {box_type} box with {distance} nm buffer")


def solvate(gmx, input_gro, output_gro, top_file, water_model='spc216'):
    """
    Solvate the system with water.
    """
    cmd = [
        gmx, 'solvate',
        '-cp', str(input_gro),
        '-cs', water_model,
        '-o', str(output_gro),
        '-p', str(top_file)
    ]

    return run_command(cmd, "Solvating system")


def add_ions(gmx, input_gro, output_gro, top_file, work_dir, concentration=0.15):
    """
    Add ions to neutralize and set ionic strength.
    """
    # Convert to absolute paths since we run with cwd=work_dir
    input_gro = Path(input_gro).resolve()
    output_gro = Path(output_gro).resolve()
    top_file = Path(top_file).resolve()
    work_dir = Path(work_dir).resolve()

    # Create ions.mdp
    ions_mdp = work_dir / 'ions.mdp'
    with open(ions_mdp, 'w') as f:
        f.write("; Minimal MDP for genion\n")
        f.write("integrator = steep\n")
        f.write("nsteps = 0\n")

    # Run grompp
    tpr = work_dir / 'ions.tpr'
    mdout = work_dir / 'mdout_ions.mdp'
    cmd = [
        gmx, 'grompp',
        '-f', str(ions_mdp),
        '-c', str(input_gro),
        '-p', str(top_file),
        '-o', str(tpr),
        '-po', str(mdout),  # Write processed mdp to work_dir
        '-maxwarn', '1'
    ]

    success, _ = run_command(cmd, "Preparing for ion addition", cwd=str(work_dir))
    if not success:
        return False, None

    # Run genion (select SOL group = 13)
    cmd = [
        gmx, 'genion',
        '-s', str(tpr),
        '-o', str(output_gro),
        '-p', str(top_file),
        '-pname', 'NA',
        '-nname', 'CL',
        '-neutral',
        '-conc', str(concentration)
    ]

    return run_command(cmd, "Adding ions", input_text='SOL\n', cwd=str(work_dir))


def write_em_mdp(output_path):
    """Write energy minimization MDP file."""
    content = """; Energy minimization
integrator          = steep
emtol               = 1000.0
emstep              = 0.01
nsteps              = 50000

; Neighbor searching
nstlist             = 10
cutoff-scheme       = Verlet
pbc                 = xyz

; Electrostatics
coulombtype         = PME
rcoulomb            = 1.0

; VdW
vdwtype             = Cut-off
rvdw                = 1.0

; Other
constraints         = none
"""
    with open(output_path, 'w') as f:
        f.write(content)
    return output_path


def run_energy_minimization(gmx, input_gro, top_file, output_gro, work_dir):
    """
    Run energy minimization.
    """
    # Convert to absolute paths since we run with cwd=work_dir
    input_gro = Path(input_gro).resolve()
    top_file = Path(top_file).resolve()
    output_gro = Path(output_gro).resolve()
    work_dir = Path(work_dir).resolve()

    # Write EM MDP
    em_mdp = work_dir / 'em.mdp'
    write_em_mdp(em_mdp)

    # grompp
    tpr = work_dir / 'em.tpr'
    mdout = work_dir / 'mdout_em.mdp'
    cmd = [
        gmx, 'grompp',
        '-f', str(em_mdp),
        '-c', str(input_gro),
        '-p', str(top_file),
        '-o', str(tpr),
        '-po', str(mdout),  # Write processed mdp to work_dir
        '-maxwarn', '2'
    ]

    success, _ = run_command(cmd, "Preparing energy minimization", cwd=str(work_dir))
    if not success:
        return False, None

    # mdrun
    cmd = [
        gmx, 'mdrun',
        '-v',
        '-deffnm', str(work_dir / 'em')
    ]

    success, result = run_command(cmd, "Running energy minimization", cwd=str(work_dir))
    if not success:
        return False, result

    # Copy final structure
    shutil.copy(work_dir / 'em.gro', output_gro)

    return True, result


def generate_posres(gmx, input_gro, output_itp, force_constant=1000):
    """
    Generate position restraints for heavy atoms.
    """
    cmd = [
        gmx, 'genrestr',
        '-f', str(input_gro),
        '-o', str(output_itp),
        '-fc', str(force_constant), str(force_constant), str(force_constant)
    ]

    # Select backbone (group 4) or Protein-H (group 3)
    return run_command(cmd, "Generating position restraints", input_text='4\n')


def get_atom_names_from_itp(itp_file):
    """
    Extract atom names from ITP file's [ atoms ] section.
    Returns list of atom names in order.
    """
    atom_names = []
    in_atoms_section = False

    with open(itp_file) as f:
        for line in f:
            stripped = line.strip()

            # Detect atoms section
            if stripped == '[ atoms ]':
                in_atoms_section = True
                continue

            # End of atoms section
            if in_atoms_section and stripped.startswith('['):
                break

            # Skip comments and empty lines
            if in_atoms_section and stripped and not stripped.startswith(';'):
                parts = stripped.split()
                if len(parts) >= 5:
                    # Format: nr type resnr residue atom cgnr charge mass
                    atom_name = parts[4]  # atom name is 5th column
                    atom_names.append(atom_name)

    return atom_names


def split_complex_pdb(complex_pdb, output_dir, ligand_itp, ligand_resname='LIG'):
    """
    Split a complex PDB into receptor and ligand components.
    Uses atom names from ITP to ensure consistency with topology.

    Args:
        complex_pdb: Path to complex PDB (receptor + ligand)
        output_dir: Directory for output files
        ligand_itp: Path to ligand ITP file (for atom names)
        ligand_resname: Residue name of ligand (default: LIG)

    Returns:
        tuple: (receptor_pdb_path, ligand_gro_path) or (None, None) on failure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get atom names from ITP
    itp_atom_names = get_atom_names_from_itp(ligand_itp)
    if not itp_atom_names:
        print(f"  ERROR: Could not extract atom names from ITP: {ligand_itp}")
        return None, None
    print(f"  ITP has {len(itp_atom_names)} atoms")

    receptor_lines = []
    ligand_coords = []

    with open(complex_pdb) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                resname = line[17:20].strip()
                if resname == ligand_resname:
                    # Extract coordinates (in Angstroms)
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ligand_coords.append((x, y, z))
                else:
                    receptor_lines.append(line)
            elif line.startswith(('TER', 'END')):
                continue

    if not receptor_lines:
        print(f"  ERROR: No receptor atoms found in complex")
        return None, None

    if not ligand_coords:
        print(f"  ERROR: No ligand atoms with resname '{ligand_resname}' found in complex")
        return None, None

    print(f"  Complex has {len(ligand_coords)} ligand atoms")

    # Check atom count matches
    if len(ligand_coords) != len(itp_atom_names):
        print(f"  ERROR: Atom count mismatch! ITP has {len(itp_atom_names)}, complex has {len(ligand_coords)}")
        print(f"  This may be due to different hydrogen counts. Try re-parameterizing the ligand.")
        return None, None

    # Write receptor PDB
    receptor_pdb = output_dir / 'receptor_from_complex.pdb'
    with open(receptor_pdb, 'w') as f:
        f.writelines(receptor_lines)
        f.write('TER\n')
        f.write('END\n')

    # Write ligand GRO with correct atom names from ITP
    ligand_gro = output_dir / 'ligand_from_complex.gro'
    with open(ligand_gro, 'w') as f:
        f.write(f"{ligand_resname} extracted from docked complex\n")
        f.write(f"{len(ligand_coords):5d}\n")
        for i, (atom_name, (x, y, z)) in enumerate(zip(itp_atom_names, ligand_coords), 1):
            # Convert Angstroms to nm
            x_nm = x / 10.0
            y_nm = y / 10.0
            z_nm = z / 10.0
            f.write(f"{1:5d}{ligand_resname:5s}{atom_name:>5s}{i:5d}{x_nm:8.3f}{y_nm:8.3f}{z_nm:8.3f}\n")
        f.write("   0.00000   0.00000   0.00000\n")

    print(f"  Extracted receptor: {receptor_pdb}")
    print(f"  Extracted ligand:   {ligand_gro} (with ITP atom names)")

    return receptor_pdb, ligand_gro


def main():
    parser = argparse.ArgumentParser(
        description="Setup GROMACS protein-ligand complex",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options (two modes)
    parser.add_argument('--complex', '-c', default=None,
                        help='Docked complex PDB (receptor + ligand from docking). Use this OR --receptor + --ligand_gro')
    parser.add_argument('--receptor', '-r', default=None,
                        help='Receptor PDB file (use with --ligand_gro)')
    parser.add_argument('--ligand_gro', default=None,
                        help='Ligand GRO coordinate file (use with --receptor)')
    parser.add_argument('--ligand_itp', required=True,
                        help='Ligand ITP topology file (from 04_parameterize_ligand.py)')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Output directory')

    # Optional arguments
    parser.add_argument('--docked_pose', '-d', default=None,
                        help='Docked ligand SDF to update coordinates (only with --receptor mode)')
    parser.add_argument('--ligand_resname', default='LIG',
                        help='Ligand residue name (default: LIG)')
    parser.add_argument('--ff', default='amber99sb-ildn',
                        help='Force field for protein (default: amber99sb-ildn)')
    parser.add_argument('--water', default='tip3p',
                        help='Water model (default: tip3p)')
    parser.add_argument('--box_type', default='cubic',
                        choices=['cubic', 'dodecahedron', 'octahedron'],
                        help='Box type (default: cubic)')
    parser.add_argument('--box_distance', type=float, default=1.0,
                        help='Distance from solute to box edge in nm (default: 1.0)')
    parser.add_argument('--ion_conc', type=float, default=0.15,
                        help='Ion concentration in M (default: 0.15)')
    parser.add_argument('--skip_em', action='store_true',
                        help='Skip energy minimization')
    parser.add_argument('--gmx', default=None,
                        help='GROMACS executable (auto-detected if not specified)')

    args = parser.parse_args()

    # Create output directory early (needed for split_complex)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resname = args.ligand_resname[:3].upper()

    # Validate inputs - two modes
    ligand_itp = Path(args.ligand_itp)
    if not ligand_itp.exists():
        print(f"ERROR: Ligand ITP file not found: {ligand_itp}")
        sys.exit(1)

    if args.complex:
        # Mode 1: Use complex PDB
        complex_path = Path(args.complex)
        if not complex_path.exists():
            print(f"ERROR: Complex PDB not found: {complex_path}")
            sys.exit(1)

        print("\n" + "-"*60)
        print("Extracting receptor and ligand from complex PDB")
        print("-"*60)
        receptor_path, ligand_gro = split_complex_pdb(complex_path, output_dir, ligand_itp, resname)

        if receptor_path is None:
            print("ERROR: Failed to extract receptor/ligand from complex")
            sys.exit(1)

    elif args.receptor and args.ligand_gro:
        # Mode 2: Separate receptor and ligand files
        receptor_path = Path(args.receptor)
        ligand_gro = Path(args.ligand_gro)

        if not receptor_path.exists():
            print(f"ERROR: Receptor file not found: {receptor_path}")
            sys.exit(1)
        if not ligand_gro.exists():
            print(f"ERROR: Ligand GRO file not found: {ligand_gro}")
            sys.exit(1)
    else:
        print("ERROR: Must specify either --complex OR both --receptor and --ligand_gro")
        sys.exit(1)

    # Find GROMACS
    gmx = args.gmx or find_gmx()
    if gmx is None:
        print("ERROR: GROMACS not found. Please install GROMACS or specify --gmx")
        sys.exit(1)

    print("\n" + "="*60)
    print("GROMACS COMPLEX SETUP")
    print("="*60)
    print(f"\nInputs:")
    if args.complex:
        print(f"  Complex:     {args.complex}")
    else:
        print(f"  Receptor:    {receptor_path}")
        print(f"  Ligand GRO:  {ligand_gro}")
    print(f"  Ligand ITP:  {ligand_itp}")
    if args.docked_pose and not args.complex:
        print(f"  Docked pose: {args.docked_pose}")
    print(f"\nParameters:")
    print(f"  Force field: {args.ff}")
    print(f"  Water model: {args.water}")
    print(f"  Box type:    {args.box_type}")
    print(f"  Box buffer:  {args.box_distance} nm")
    print(f"  Ion conc:    {args.ion_conc} M")
    print(f"\nOutput:")
    print(f"  Directory:   {output_dir}")

    # Step 1: Process receptor with pdb2gmx
    print("\n" + "-"*60)
    print("Step 1: Process receptor with pdb2gmx")
    print("-"*60)

    receptor_gro = output_dir / 'receptor.gro'
    receptor_top = output_dir / 'topol.top'

    success, _ = run_pdb2gmx(gmx, receptor_path, receptor_gro, receptor_top,
                             output_dir, ff=args.ff, water=args.water)
    if not success:
        print("ERROR: pdb2gmx failed")
        sys.exit(1)

    # Step 2: Prepare ligand coordinates
    print("\n" + "-"*60)
    print("Step 2: Prepare ligand coordinates")
    print("-"*60)

    ligand_positioned = output_dir / f'{resname}_positioned.gro'

    if args.docked_pose and Path(args.docked_pose).exists():
        print("  Using docked pose coordinates...")
        update_ligand_gro_coordinates(ligand_gro, args.docked_pose, ligand_positioned)
    else:
        print("  Using original ligand coordinates...")
        shutil.copy(ligand_gro, ligand_positioned)

    # Process ligand ITP - extract atomtypes if present (ACPYPE issue)
    atomtypes_itp, ligand_itp_local = process_ligand_itp(ligand_itp, output_dir, resname)
    if atomtypes_itp is None:
        # No atomtypes to extract, just copy the file
        ligand_itp_local = output_dir / f'{resname}.itp'
        shutil.copy(ligand_itp, ligand_itp_local)
        print(f"  Copied ligand topology to {ligand_itp_local}")

    # Step 3: Combine receptor and ligand
    print("\n" + "-"*60)
    print("Step 3: Combine receptor and ligand")
    print("-"*60)

    complex_gro = output_dir / 'complex.gro'
    combine_receptor_ligand(receptor_gro, ligand_positioned, complex_gro, resname)
    print(f"  Created: {complex_gro}")

    # Update topology to include ligand
    complex_top = output_dir / 'topol.top'
    atomtypes_name = atomtypes_itp.name if atomtypes_itp else None
    create_complex_topology(receptor_top, ligand_itp_local.name, complex_top, resname,
                            atomtypes_itp=atomtypes_name)
    print(f"  Updated: {complex_top}")

    # Step 4: Create box
    print("\n" + "-"*60)
    print("Step 4: Create simulation box")
    print("-"*60)

    boxed_gro = output_dir / 'complex_box.gro'
    success, _ = create_box(gmx, complex_gro, boxed_gro,
                            box_type=args.box_type, distance=args.box_distance)
    if not success:
        print("ERROR: Box creation failed")
        sys.exit(1)

    # Step 5: Solvate
    print("\n" + "-"*60)
    print("Step 5: Solvate system")
    print("-"*60)

    solvated_gro = output_dir / 'complex_solv.gro'
    success, _ = solvate(gmx, boxed_gro, solvated_gro, complex_top)
    if not success:
        print("ERROR: Solvation failed")
        sys.exit(1)

    # Step 6: Add ions
    print("\n" + "-"*60)
    print("Step 6: Add ions")
    print("-"*60)

    ionized_gro = output_dir / 'complex_ions.gro'
    success, _ = add_ions(gmx, solvated_gro, ionized_gro, complex_top,
                          output_dir, concentration=args.ion_conc)
    if not success:
        print("ERROR: Ion addition failed")
        sys.exit(1)

    # Step 7: Energy minimization
    if not args.skip_em:
        print("\n" + "-"*60)
        print("Step 7: Energy minimization")
        print("-"*60)

        em_gro = output_dir / 'complex_em.gro'
        success, _ = run_energy_minimization(gmx, ionized_gro, complex_top, em_gro, output_dir)
        if not success:
            print("ERROR: Energy minimization failed")
            sys.exit(1)
        final_gro = em_gro
    else:
        print("\n" + "-"*60)
        print("Step 7: Skipping energy minimization")
        print("-"*60)
        final_gro = ionized_gro

    # Step 8: Generate position restraints
    print("\n" + "-"*60)
    print("Step 8: Generate position restraints")
    print("-"*60)

    posres_itp = output_dir / 'posres_protein.itp'
    success, _ = generate_posres(gmx, final_gro, posres_itp)
    if not success:
        print("WARNING: Position restraint generation failed")

    # Summary
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print(f"\nOutput files in {output_dir}:")
    print(f"  Final structure: {final_gro.name}")
    print(f"  Topology:        {complex_top.name}")
    print(f"  Ligand topology: {ligand_itp_local.name}")
    if atomtypes_itp:
        print(f"  Ligand atomtypes: {atomtypes_itp.name}")

    print(f"\nNext step - run stability test:")
    print(f"  python 06_test_binding_stability.py \\")
    print(f"      --complex_gro {final_gro} \\")
    print(f"      --topology {complex_top} \\")
    print(f"      --output_dir OUTPUT_DIR")
    print()


if __name__ == "__main__":
    main()
