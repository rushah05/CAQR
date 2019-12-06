#!/bin/sh
#SBATCH -p skx-normal
#SBATCH -J caqr_1024
#SBATCH -o caqr.np1024.skx.o%j
#SBATCH -t 00:10:00
#SBATCH -N 32 -n 1024
#SBATCH -A COSC6365-Fall-2019

module add gcc/7.1.0
module add mkl/18.0.2
mpicc -O2 ca_qr.c  -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -o caqr -DGN=2621440

echo "Matrix size is 327680,512"
mpirun caqr 327680 512

echo "Matrix size is 655360,512"
mpirun caqr 655360 512

echo "Matrix size is 1310720,512"
mpirun caqr 1310720 512

echo "Matrix size is 2621440,512"
mpirun caqr 2621440 512
