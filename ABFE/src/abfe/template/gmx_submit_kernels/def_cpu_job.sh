#!/usr/bin/env bash
# CPU job template for Gromacs with thread-MPI settings
# Load LUMI GROMACS module

export OMP_NUM_THREADS="${1}"
STEPNAME="${2}"
TOPOLOGY="${3}"
STRUCTURE="${4}"
#opts
GROMACS_TPR="${STEPNAME}.tpr"
CONFOUT="${STEPNAME}.gro"
#Grompp this:
gmx grompp -f ./${STEPNAME}.mdp -c ${STRUCTURE} -r ${STRUCTURE} -p ${TOPOLOGY} -o ${GROMACS_TPR} -maxwarn 2
# run gromacs command
gmx mdrun  -s $GROMACS_TPR -c $CONFOUT -deffnm ${STEPNAME}
