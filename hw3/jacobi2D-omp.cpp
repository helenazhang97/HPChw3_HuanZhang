//Gauss-Seldel method to solve 2d Poisson equation with homogeneous boundary condition with openmp.
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <cmath>
#include <iostream>
#include <omp.h>
#include <fstream>
using namespace std;
//initialize u and f vector
//update red points
//update black points
int main(int argc, char** argv)
{
  	long p=read_option<long>("-p",argc,argv,"4");
  	long N=read_option<long>("-n",argc,argv,"9");
	#ifdef _OPENMP
	omp_set_num_threads(p);
	#endif

	double h=1./(N+1);
	double* f=(double *) calloc(sizeof(double),(N+2)*(N+2));
	double* u=(double *)calloc(sizeof(double),(N+2)*(N+2));
	double* unew=(double *) calloc(sizeof(double),(N+2)*(N+2));

	for (long i=0;i<(N+2)*(N+2);i++){ f[i]=1.0;u[i]=0.0;unew[i]=0.0;}

	double tol=1e-5;
	int nstep=0;
	double res=0;
	int maxnstep=5e4;
	#ifdef _OPENMP
	printf("start with %d threads...\n", p);	
	double tt=omp_get_wtime();
	#endif
	
	do{
	#ifdef _OPENMP
	#pragma omp parallel for 
	#endif
		for (long i=1;i<N+1;i++){
			for (long j=1;j<N+1;j++){
				unew[i+(N+2)*j]=0.25*(h*h*f[i+(N+2)*j]+u[i-1+(N+2)*j]+u[i+(N+2)*(j-1)]+u[i+1+(N+2)*j]+u[i+(N+2)*(j+1)]);
			}
		}
	
		res=0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:res)
#endif
		for(long i=0;i<(N+2)*(N+2);i++){
			res+=fabs(u[i]-unew[i]);
		}
		
	/*	if (nstep%1000==0){
		printf("nstep=%d, res = %f\n",nstep,res);
		}
		*/
		nstep++;
		double* utemp=u;
		u=unew;
		unew=utemp;
	
	}while(nstep<maxnstep);
//	}while(res>tol);
	
	#ifdef _OPENMP
	printf("grid number N=%d, maxnstep=%d,thread number p = %d, parallel-executing time = %fs\n",N,nstep,p,omp_get_wtime()-tt);
	#endif

	ofstream myfile ("jacobi.txt");
	if (myfile.is_open()){
		for (int i=0;i<(N+2)*(N+2);i++){
			myfile <<u[i]<< " ";
		}
	}
	myfile.close();
	free(f);
	free(u);
	free(unew);

}
