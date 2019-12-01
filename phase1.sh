#!/bin/bash
#SBATCH -J caqr
#SBATCH -o test_caqr.o%j
#SBATCH -t 00:20:00
#SBATCH -N 1 -n 8 --constraint=gen10 
#SBATCH -A pwu

#gcc -O3 -o caqr ca_qr.c -lm
#./caqr


module load OpenMPI/intel/3.1.2
mpicc -O2 ca_qr.c  -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -o caqr
mpirun caqr
