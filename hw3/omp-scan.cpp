#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include "utils.h"
// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n,long p) {
	// TODO: implement multi-threaded OpenMP scan

	printf("%d threads executing iteration",p);
	omp_set_num_threads(p);//set number of threads	
	long interval=0;
	if (n%p==0){interval=int(n/p);}else{interval=floor(n/p)+1;}
	int ends[p+1];
	for(int i=0;i<p;i++){ends[i]=interval*i;}ends[p]=n;

#pragma omp parallel default(none) shared(n, A, prefix_sum,ends) 
	{
		long id=omp_get_thread_num();
		prefix_sum[ends[id]]=A[ends[id]];
		for(int i=ends[id]+1;i<ends[id+1];i++){
			prefix_sum[i]=prefix_sum[i-1]+A[i];
		}
	}
	for (int id=1;id<p;id++){
		for(int i=ends[id];i<ends[id+1];i++){
			prefix_sum[i]=prefix_sum[i]+prefix_sum[ends[id]-1];
		}
	}
}

int main(int argc, char** argv) {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  long p=read_option<long>("-p",argc,argv,"4");
  tt = omp_get_wtime();
  scan_omp(B1, A, N, p);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
