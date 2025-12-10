# AutoDock Vina Docking Workflow

A Python script for automated molecular docking using AutoDock Vina, with streamlined ligand and receptor preparation.

## Overview

This script automates the complete docking workflow:

1. Extracts binding site coordinates from a reference structure
2. Prepares ligands using Meeko
3. Prepares receptors using Open Babel
4. Runs AutoDock Vina docking
5. Parses and saves results as JSON

## Installation

### Create Conda Environment

```bash
conda create -n vina_docking python=3.10
conda activate vina_docking
```

### Install Dependencies

```bash
# AutoDock Vina (Python bindings)
pip install vina

# Meeko for ligand preparation
pip install meeko

# Open Babel for receptor preparation
conda install -c conda-forge openbabel

# NumPy (usually installed as dependency, but just in case)
pip install numpy
```

### Verify Installation

```bash
# Check Vina
vina --help

# Check Meeko
mk_prepare_ligand.py --help

# Check Open Babel
obabel -H
```

## Usage

### Basic Usage

```bash
python3 vina_docking.py \
    --ligand LIGAND.sdf \
    --receptor RECEPTOR.pdb \
    --reference REFERENCE.pdb \
    --ligname LIG
```

### Multiple Receptors

Dock the same ligand against multiple receptor conformations:

```bash
python3 vina_docking.py \
    --ligand CP55940.sdf \
    --receptor ArrB1_CB1.pdb Gio_CB1.pdb \
    --reference 7fee.pdb \
    --ligname 9GF
```

### Custom Docking Parameters

```bash
python3 vina_docking.py \
    --ligand ligand.sdf \
    --receptor receptor.pdb \
    --reference reference.pdb \
    --ligname LIG \
    --exhaustiveness 64 \
    --num-modes 50 \
    --energy-range 5 \
    --output my_docking_results
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--ligand` | Yes | - | Ligand file in SDF format |
| `--receptor` | Yes | - | Receptor PDB file(s), can specify multiple |
| `--reference` | Yes | - | Reference PDB containing ligand for binding site definition |
| `--ligname` | No | None | Residue name of ligand in reference (e.g., 9GF). If not specified, uses all HETATM records excluding common solvents/ions |
| `--output` | No | vina_results | Output directory |
| `--exhaustiveness` | No | 32 | Search thoroughness (higher = slower but more accurate) |
| `--num-modes` | No | 20 | Maximum number of binding poses to generate |
| `--energy-range` | No | 4.0 | Energy range (kcal/mol) for pose clustering |

## Output Files

The script creates the following files in the output directory:

```
vina_results/
├── LIGAND_prepared.pdbqt      # Prepared ligand
├── RECEPTOR_prepared.pdbqt    # Prepared receptor
├── RECEPTOR_docked.pdbqt      # Docked poses
└── RECEPTOR_results.json      # Parsed docking scores
```

### Results JSON Format

```json
{
  "receptor": "ArrB1_CB1",
  "poses": [
    {"mode": 1, "affinity": -9.2},
    {"mode": 2, "affinity": -8.8},
    {"mode": 3, "affinity": -8.5}
  ],
  "best_affinity": -9.2
}
```

## Binding Site Definition

The binding site is automatically extracted from the reference structure:

- Centre: geometric centre of the ligand atoms
- Box size: ligand dimensions + 10 Å buffer (capped at 30 Å per dimension)

Common solvent molecules and ions (HOH, WAT, NA, CL, etc.) are automatically excluded from binding site calculation.

## Visualisation

View docking results in PyMOL:

```bash
pymol receptor.pdb vina_results/receptor_docked.pdbqt
```

## Tips

**Exhaustiveness**: The default value of 32 provides a good balance between speed and accuracy. For production runs or difficult binding sites, consider using 64 or higher.

**Ligand preparation**: Ensure your SDF file has correct 3D coordinates and proper protonation state for your target pH.

**Receptor preparation**: The script adds hydrogens automatically. For best results, ensure your receptor PDB has:
- No missing heavy atoms in the binding site
- Appropriate protonation states for key residues
- No alternate conformations (or only one selected)

## Troubleshooting

**"vina not found"**: Ensure the conda environment is activated and Vina is installed.

**"mk_prepare_ligand.py not found"**: Install Meeko with `pip install meeko`.

**"obabel not found"**: Install Open Babel with `conda install -c conda-forge openbabel`.

**Poor docking scores**: Try increasing exhaustiveness, checking ligand protonation, or verifying the binding site is correctly defined.

## Dependencies Summary

| Package | Purpose | Installation |
|---------|---------|--------------|
| Python 3.10+ | Runtime | conda |
| AutoDock Vina | Docking engine | `pip install vina` |
| Meeko | Ligand preparation | `pip install meeko` |
| Open Babel | Receptor preparation | `conda install -c conda-forge openbabel` |
| NumPy | Coordinate calculations | `pip install numpy` |