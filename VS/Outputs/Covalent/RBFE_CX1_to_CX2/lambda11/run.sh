#!/bin/bash
# FEP simulation for lambda window 11
set -e

# Run from lambda directory
GMX="gmx"

echo "Lambda window 11"
echo "========================"

# Energy minimization
if [ ! -f em.gro ]; then
    echo "Running energy minimization..."
    $GMX grompp -f em.mdp -c ../input/hybrid.gro -p ../input/topol.top -o em.tpr -maxwarn 5
    $GMX mdrun -deffnm em -v 
fi

# NVT equilibration
if [ ! -f nvt.gro ]; then
    echo "Running NVT equilibration..."
    $GMX grompp -f nvt.mdp -c em.gro -r em.gro -p ../input/topol.top -o nvt.tpr -maxwarn 5
    $GMX mdrun -deffnm nvt -v 
fi

# NPT equilibration
if [ ! -f npt.gro ]; then
    echo "Running NPT equilibration..."
    $GMX grompp -f npt.mdp -c nvt.gro -r nvt.gro -p ../input/topol.top -t nvt.cpt -o npt.tpr -maxwarn 5
    $GMX mdrun -deffnm npt -v 
fi

# Production
if [ ! -f prod.gro ]; then
    echo "Running production..."
    $GMX grompp -f prod.mdp -c npt.gro -p ../input/topol.top -t npt.cpt -o prod.tpr -maxwarn 5
    $GMX mdrun -deffnm prod -v 
fi

echo "Lambda 11 complete!"
