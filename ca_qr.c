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
void getIpMat(int rank, int np, float *A, int ldA, int ln, int d)
{
	int i, j;
        for(i=0;i<ln;++i)
        {
        	for(j=0;j<d;++j)
               	{
                	A[i+(j*ldA)] = guassrand();
               	}
        }
}

/*Check if the number of processor passed is power of 2*/
int isPowerOfTwo(int np) 
{ 
   if(np==0) 
   	return 0; 
  
   if(ceil(log2(np)) == floor(log2(np)))
	return 1;
   else
	return 0;
}

/*Extract R from Q*/
void extractR(int rank, float *A, int ldA, float *R, int ldR, int ln, int d)
{
	int i,j;
	for(i=0; i<d; ++i)
	{
		for(j=0; j<d; ++j)
		{
			R[i+(j*ldR)] = A[i+(j*ldA)];
		}
	}
}

/* Stack Rs */
void stack(float *R1, float *R2, float *RR, int k)
{
    int i,j;
    for (j=0; j<k; j++) {
        for (i=0; i<k; i++) {
            RR[i+j*(2*k)] = R1[i+j*k];
            RR[i+k+j*(2*k)] = R2[i+j*k];
        }
    }
}

/* Ustack Rs */
void unstack(float *R1, float *R2, float *RR, int k)
{
    int i,j;
    for (j=0; j<k; j++) {
        for (i=0; i<k; i++) {
            R1[i+j*k] = RR[i+j*(2*k)];
            R2[i+j*k] = RR[i+k+j*(2*k)];
        }
    }
}


/*Communication avoidance QR
 * 1)First perform QR on each process
 * 2)Stack Rs by sharing accross alternate process
 * 3)Perform QR on the stacked Rs until you get the final Q
 * 4)Multiply all Qs back to process 0
 * ln = local n
 * d = no of columns
 * */
void comm_avoidance_qr(int rank, int np, float *A, int ldA, float* R, int ldR, int ln, int d)
{
	float *tau = (float*)malloc(sizeof(float)*d);
    	float *Q1 = (float*)malloc(sizeof(float)*d*d);
    	float *Q2 = (float*)malloc(sizeof(float)*d*d);
	float *Q = (float*)malloc(sizeof(float)*2*d*d*log2(np));
	float *RR = (float*)malloc(sizeof(float)*2*d*d);
	float *R1 = (float*)malloc(sizeof(float)*d*d);
	float *R2 = (float*)malloc(sizeof(float)*d*d);
	float *stackedR = (float*)malloc(sizeof(float)*2*d*d);
	float *Qtmp = (float*)malloc(sizeof(float)*2*d*d);
    	float *Atmp = (float*)malloc(sizeof(float)*2*ln*d);

	
	/** MKL's library to perform QR 
 * 	The QR is replaced in A. The upper triangular forms R
 * 	Lower triangular along with tau forms Q **/
	LAPACKE_sgeqrf(LAPACK_COL_MAJOR, ln, d, A, ldA, tau);
	extractR(rank, A, ldA, R1, ldR, ln, d);
	/** MKL's library to form Q explicitly in A**/
        LAPACKE_sorgqr(LAPACK_COL_MAJOR, ln, d, d, A, ldA, tau);

	int r;
    	MPI_Status status;
    	for (r=0; r<log2(np); r++) {
        	if (r== 0 || !( rank & ((1<<r)-1) )) {
			/** Receiver **/
            		if ( !(rank & (1<<r))  ) { 
                		MPI_Recv(R2, d*d, MPI_FLOAT, rank ^ (1<<r), 0, MPI_COMM_WORLD, &status);
                		stack(R1, R2, RR, d);
				/** MKL's library to perform QR on the stacked R **/ 
                		LAPACKE_sgeqrf(LAPACK_COL_MAJOR, 2*d, d, RR, 2*d, tau); 
				/** MKL's library to extract R1 from  upper traingular in RR **/
                		LAPACKE_slacpy(LAPACK_COL_MAJOR, 'U', d, d, RR, 2*d, R1, d);
				/** MKL's library to form Q explicitly in RR **/
                		LAPACKE_sorgqr(LAPACK_COL_MAJOR, 2*d, d, d, RR, 2*d, tau);
				/** MKL's library to store Q in Q[r] **/
                		LAPACKE_slacpy(LAPACK_COL_MAJOR, 'A', 2*d, d, RR, 2*d, &Q[2*d*d*r], 2*d);
            		} 
			/** Sender **/
			else 
			{
                		MPI_Send(R1, d*d, MPI_FLOAT, rank ^ (1<<r), 0, MPI_COMM_WORLD);
            		}
        	}
    	}
    	extractR(rank, R, ldR, R1, ldR, ln, d);	
    	for (r=log2(np)-1; r>=0; r--) {
        	if (r== 0 || !( rank & ((1<<r)-1) )) {
		/** Send **/
            	if ( !(rank & (1<<r)) ) { 
                	/** Split Q{2d,d} into Q1{d,d} and Q2{d,d} **/
			unstack(Q1, Q2, &Q[2*d*d*r], d);
			/** Send Q2 to the next processor **/
                	MPI_Send(Q2, d*d, MPI_FLOAT, rank ^ (1<<r), 0, MPI_COMM_WORLD); 
            	} 
	    	else {
			/** Received Q1 from the previous processor **/
                	MPI_Recv(Q1, d*d, MPI_FLOAT, rank ^ (1<<r), 0, MPI_COMM_WORLD, &status);
            	}
            	if (r>0) {
			/** MKL's library to perform matrix multiplication of received Q1 and existing Q block **/
                	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2*d, d, d, 1.0, &Q[2*d*d*(r-1)], 2*d, Q1, d, 0, Qtmp, 2*d);
                	LAPACKE_slacpy(LAPACK_COL_MAJOR, 'A', 2*d, d, Qtmp, 2*d, &Q[2*d*d*(r-1)], 2*d);
            	}
        }
    }

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, d, d, 1.0, A, ldA, Q1, d, 0, Atmp, ln);
    LAPACKE_slacpy(LAPACK_COL_MAJOR, 'A', ln, d, Atmp, ln, A, ldA);
    free(R1);free(R2); free(RR);
    free(tau);
    free(Q1); free(Q2); free(Q);
    free(Atmp); free(Qtmp);
	
}


int main(int argc, char *argv[])
{
	int i,j, rank,np;
	int gn = 512;
	int d = 64;
	//int k = 16;

	/** Initialize MPI **/
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &np);

	/** Generate a block local to a process **/
	int begin = (gn * rank)/np;
        int end = (gn * (rank + 1))/np;
        int ln = end - begin; 
	float *A = (float*)malloc(sizeof(float)*ln*d);
	float *R =(float*)malloc(sizeof(float)*d*d);
	int ldA = ln;
	int ldR = d;

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
	getIpMat(rank, np, A, ldA, ln, d);

	/** Perform Communication avoidance QR **/
	comm_avoidance_qr(rank, np, A, ldA, R, ldR,  ln, d);		

	MPI_Finalize();
	free(A);	
	return 0;
}
