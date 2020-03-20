#include<stdio.h>
 
int main()
{
	int cols;
	printf ("Enter columns number:\t");
	scanf ("%d", &cols);
	int vector1 [cols/2];
	int vector2 [cols/2];
	for (int i = 0; i < cols/2; i++) {
		vector1[i] = i*2 + 1;
		vector2[i] = i*2 + 2;
	}
	for (int i = 0; i < cols/2; i++) 
		printf ("%d \t", vector1[i]);
	printf ("\n");
	for (int i = 0; i < cols/2; i++) 
		printf ("%d \t", vector2[i]);
	printf ("\n");
	int round;
	printf ("Insert round\n");
	scanf ("%d", &round);
	for (int i = 0; i < round; i++)
	{
		int tmp = vector2[0];
		for (int i = 0; i < cols/2 - 1; i++)
			vector2[i] = vector2[i+1];	
		vector2[cols/2 - 1] = vector1[cols/2 - 1];
		for (int i = (cols/2 -1); i > 1; i--)
			vector1[i] = vector1[i-1];	
		vector1[1] = tmp;
	}

	for (int i = 0; i < cols/2; i++) 
		printf ("%d \t", vector1[i]);
	printf ("\n");
	for (int i = 0; i < cols/2; i++) 
		printf ("%d \t", vector2[i]);
	printf ("\n");
	/*
	int index;
	printf ("Insert column\n");
	scanf ("%d", &index);
	printf ("%d \n", vector1[index]);
	printf ("%d \n", vector2[index]);
	*/

  	return 0; 
}
