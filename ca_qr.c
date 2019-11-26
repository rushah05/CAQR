#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>
#include <mkl.h>

/*
 *Generate Gaussian Random no
 *refernce link - http://c-faq.com/lib/gaussian.html
 *Used a method discussed in Knuth and due originally to Marsaglia
 * */
float guassrand()
{
	static float V1, V2, S;
	static int phase = 0;
	float num;

	if(phase == 0) 
	{
		do 
		{
			float U1 = (float)rand() / RAND_MAX;
			float U2 = (float)rand() / RAND_MAX;
			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while(S >= 1 || S == 0);

		num = V1 * sqrt(-2 * log(S) / S);
	} 
	else
	{
		num = V2 * sqrt(-2 * log(S) / S);
	}
	phase = 1 - phase;
	return num;
}


/* Generate distributed input matrix A of size n * d*/
void generate_ip_mat(int rank, int np, float *A, int n, int d)
{
	int i, j, rk;
	if(rank == 0)
	{
		for(rk=0; rk<np; ++rk)
		{
			int begin = (n * rk)/np;
        		int end = (n * (rk + 1)/np);
        		int block = end - begin;
			float *Abuf = (float*)malloc(sizeof(float)*block*d);
			for(i=0;i<block;++i)
        		{
                		for(j=0;j<d;++j)
                		{
                        		Abuf[i+(j*d)] = guassrand();
                		}
        		}
			//printf("Rank %d :: Abuf[%d,%d]\n",rank,block,d);
			if(rk == 0)
			{
				LAPACKE_slacpy(LAPACK_ROW_MAJOR,'P', d, block, Abuf, d,A, d);
			}
			else
			{
				MPI_Send(Abuf, (block * d), MPI_FLOAT, rk, 1, MPI_COMM_WORLD);
			}
		}
	}
	else
	{
		int begin = (n * rank)/np;
        	int end = (n * (rank + 1))/np;
        	int block = end - begin;
        	MPI_Recv(A, (block * d), MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	
	}
}

/*Utility function to check if the number of processor passed is power of 2*/
int isPowerOfTwo(int n) 
{ 
   if(n==0) 
   	return 0; 
  
   if(ceil(log2(n)) == floor(log2(n)))
	return 1;
   else
	return 0;
}

/*Communication avoidance QR
 * 1)First perform QR on each process
 * 2)Stack Rs by sharing accross alternate process
 * 3)Perform QR on the stacked Rs until you get the final Q
 * 4)Multiply all Qs back to process 0*/
void comm_avoidance_qr(int rank, float *A, int n, int d, int k)
{
	double *tau = (double*)malloc(sizeof(double)*k);	
}


int main(int argc, char *argv[])
{
	int i,j, rank,np;
	int n = 512;
	int d = 64;
	int k = 16;
	/** Initialize MPI **/
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &np);

	int begin = (n * rank)/np;
        int end = (n * (rank + 1))/np;
        int block = end - begin;
	float *A = (float*)malloc(sizeof(float)*block*d);
	
	/**Check if the number of processors passed is power of 2**/
	if(rank == 0)
	{
		if(isPowerOfTwo(np) == 0)
		{
			printf("Please pass the number of processors in the power of 2\n");
			return 0;
		}
	}

	/** Generate distributed input matrix A using Gaussian random no*/
	generate_ip_mat(rank, np, A, n, d);
		
	MPI_Finalize();
	free(A);	
	return 0;
}
