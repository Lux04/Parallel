#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    #pragma omp parallel 
    {
        int nthreads, thread_id = 0 ;
        nthreads = omp_get_num_threads();
        thread_id = omp_get_thread_num();
        printf("Hello OpenMP \n");
        printf("I have %d threads and I am thread %d \n", nthreads, thread_id); 
    }   
}