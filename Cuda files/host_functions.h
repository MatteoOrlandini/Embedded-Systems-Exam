int sign_svd (float num);

int svd_one_sided_jacobi_C (int rows, int columns);

bool openFile(FILE** fp, char * fileName, const char * mode);

void fillRowMajorOrderMatrix (FILE** fp, float ** matrix, int * rows, int * columns);

void createColumnMajorOrderMatrix (float ** matCol, float * matRow, int rows, int columns);

void initializeArray (float ** arr, int dim);

void descentOrdering (float * arr, int dim);

void printAndSaveArray (FILE** fp, float * arr, int dim);
