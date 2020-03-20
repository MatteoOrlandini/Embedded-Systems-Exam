#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> //for chdir
#include <stdbool.h> //for bool type
#include "host_functions.h"

#define MAXCHAR 3000

float * B;
float * AUX1;

int main (int argc, char * argv[])
{
    char input [100], fileName[] = {"Matrix/"};
    chdir("../");
    FILE *fp;
    if (argc == 1){
        do
        {
            printf ("Insert matrix name: \n");
            scanf ("%s", input);
            strcat(fileName, input);
        }
        while(openFile(&fp, fileName, "r") == false);
    }
    else if (argc == 2){
        sprintf(fileName, "Matrix/%s", argv[1]);
        if (openFile(&fp, fileName, "r") == false)
        	exit(1);
	}        
    float * matrix;
    int rows, columns;
    fillRowMajorOrderMatrix(&fp, &matrix, &rows, &columns);
    fclose(fp);
    // Column order matrix
    createColumnMajorOrderMatrix(&B, matrix, rows, columns);
    // initialize AUX1 array to zero
    initializeArray (&AUX1, columns);
    // Open new file to store the singular values
	sprintf(fileName, "SingularValues/C/Singular values C %dX%d.txt", rows, columns);
    openFile(&fp, fileName, "w");
    //compute one sided jacobi
    int iterations = svd_one_sided_jacobi_C(rows, columns);
    printf("iterations: %d \n", iterations);
    fprintf(fp, "iterations: %d \n", iterations);
    descentOrdering(AUX1, columns);
    //print array and save on file
	printAndSaveArray (&fp, AUX1, columns);
    fclose(fp);
    //free the memory
    free(B);
    free(AUX1);
    return 0;
}
