/******************************************************************************
* FILE: omp_bug5.c
* DESCRIPTION:
*   Using SECTIONS, two threads initialize their own array and then add
*   it to the other's array, however a deadlock occurs.
* AUTHOR: Blaise Barney  01/29/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1000000
#define PI 3.1415926535
#define DELTA .01415926535

int main (int argc, char *argv[]) 
{
int nthreads, tid, i;
float a[N], b[N];
omp_lock_t locka, lockb;

/* Initialize the locks */
omp_init_lock(&locka);
omp_init_lock(&lockb);

/* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel shared(a, b, nthreads, locka, lockb) private(tid)
  {

  /* Obtain thread number and number of threads */
  tid = omp_get_thread_num();
  #pragma omp master
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);
  #pragma omp barrier

  #pragma omp sections nowait
    {
    #pragma omp section
      {
//comments: the code has two nested locks in each thread. Before each thread enter its second loop, the newly used lock mush be available for it to use. When the program starts thread 1 grabs locka and thread 2 grabs lockb, and inside thread when it wants to use lockb, it must wait until lockb is available. However lockb is never available since it is held by thread2 waiting for the release of lock a. As a result the two threads each holds a lock and waits each other forever. I broke the nest and make them sequential in each set. Note that because lockb must be used after released, it ensures that before we use b to update a in thread1, the value of b has been properly initialized in thread2(otherwise the lock would not have been released).
	  printf("Thread %d initializing a[]\n",tid);
      omp_set_lock(&locka);
      for (i=0; i<N; i++)
        a[i] = i * DELTA;
      omp_unset_lock(&locka);
	  omp_set_lock(&lockb);
	  printf("Thread %d adding a[] to b[]\n",tid);
      for (i=0; i<N; i++)
        b[i] += a[i];
            omp_unset_lock(&lockb);
      }

    #pragma omp section
      {
      printf("Thread %d initializing b[]\n",tid);
      omp_set_lock(&lockb);
      for (i=0; i<N; i++)
        b[i] = i * PI;
      omp_unset_lock(&lockb);
	  omp_set_lock(&locka);
	  printf("Thread %d adding b[] to a[]\n",tid);
      for (i=0; i<N; i++)
        a[i] += b[i];
      
      omp_unset_lock(&locka);
      }
    }  /* end of sections */
  }  /* end of parallel region */

}

