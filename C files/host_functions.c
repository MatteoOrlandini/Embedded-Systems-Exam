#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> //for chdir
#include <stdbool.h> //for bool type
#include "host_functions.h"
#define MAXCHAR 30000

extern float * B;
extern float * AUX1;

bool openFile(FILE** fp, char * fileName, const char * mode){
    if ((*fp = fopen(fileName, mode)) != NULL)
    {
        //printf("File %s in %s opened \n", fileName, directory);
        return true;
    }

    else
    {
        printf("Error while opening file \n");
        return false;
    }
}

void fillRowMajorOrderMatrix(FILE** fp, float ** matrix, int * rows, int * columns){
    char buf[MAXCHAR]; //buffer for reading the file
    char * numChar; //elemento della matrice (salvato come ascii)
    *matrix = (float*)malloc(sizeof(float));
    (*rows) = 0;
    (*columns) = 0;
    int numMatrixEle = 0;
    while (fgets(buf, MAXCHAR, *fp) != NULL){
        numChar = strtok(buf, " ");
        *columns = 0;
        while (numChar != NULL){
            (*matrix)[numMatrixEle] = atof(numChar);
            //printf("A[%d][%d]: %2.9f \t", *rows, *columns, (*matrix)[numMatrixEle]);
            numChar = strtok(NULL, " ");
            (*columns)++;
            numMatrixEle++;
            *matrix=realloc(*matrix, (numMatrixEle+1)*sizeof(float));
        }
        //printf("\n");
        (*rows)++;
    }
}

void createColumnMajorOrderMatrix(float ** matCol, float * matRow, int rows, int columns){
    *matCol = (float*)malloc(rows * columns * sizeof(float));
    if(*matCol == NULL)
    {
        printf("Memoria esaurita\n");
        exit(1);
    }
    for (int i = 0; i < columns; i++){
        for (int j = 0; j < rows; j++){
            (*matCol)[i*rows+j] = matRow[j*columns+i];
            //printf("matCol[%d]: %2.9f \n", i*rows+j, matCol[i*rows+j]);
        }
    }
}

void initializeArray (float ** arr, int dim){
    *arr = (float*)malloc(dim*sizeof(float));
	memset(*arr, 0, dim*sizeof(*arr[0]));
}

void descentOrdering (float * arr, int dim){
    for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)             //Loop for comparing other values
		{
			if (arr[j] < arr[i])                //Comparing other array elements
			{
				float tmp = arr[i];         //Using temporary variable for storing last value
				arr[i] = arr[j];            //replacing value
				arr[j] = tmp;             //storing last value
			}
		}
	}
}

void printAndSaveArray (FILE** fp, float * arr, int dim){
    for (int i = 0; i < dim; i++){
        printf ("arr[%d]: %f \n", i, arr[i]);
        fprintf(*fp, "%f\n", arr[i]);
    }
}
