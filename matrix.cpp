#include "matrix.h"
#include <math.h>


Matrix* initMatrix(int rows,int columns)				//��ʼ��һ������
{
	Matrix *matrix = NULL;
	if (rows>0 && columns>0)
	{
		matrix = (Matrix*)malloc(sizeof(Matrix));
		if (matrix == NULL) {
			fprintf(stderr, "Memory allocation failed\n");
			return NULL;
		}
		matrix->rows = rows;
		matrix->columns = columns;
		matrix->data = (double*)malloc(sizeof(double)*rows*columns);
		
		if (matrix->data == NULL) {
			fprintf(stderr, "Memory allocation failed\n");
			free(matrix); // �ͷ��ѷ����matrix�ṹ���ڴ�
			return NULL;
		}
		memset(matrix->data, 0, sizeof(double)*rows*columns);
		return matrix;
	}
	else 
		return NULL;
} 

// ��֤array��С��matrix.row*columnsһ��
void assignMatrix(Matrix *matrix,double *array) 		//������ֵ
{
	if (matrix->data != NULL && array != NULL)
	{
		memcpy(matrix->data, array, (matrix->rows * matrix->columns) * sizeof(double));
	}
}
 
int capacityMatrix(Matrix *matrix)
{
	return matrix->rows * matrix->columns;
}
 
void freeMatrix(Matrix *matrix)
{
	if (matrix != NULL) {
        //free(matrix->data); // �ͷž�������ݴ洢��
		SAFE_FREE(matrix->data); // ��ȫ�ͷž�������ݴ洢��
        free(matrix); // �ͷž���ṹ�屾��
        printf("ptr released successfully\n");
    }
}
 
void copyMatrix(Matrix *matrix_A, Matrix *matrix_B)
{
	if (matrix_B == NULL || matrix_A == NULL) return;
    matrix_B->rows = matrix_A->rows;
    matrix_B->columns = matrix_A->columns;
    if (matrix_B->data != NULL) {
        SAFE_FREE(matrix_B->data); // �ͷžɵ��ڴ�
    }
    matrix_B->data = (double*)malloc(capacityMatrix(matrix_A) * sizeof(double));
    if (matrix_B->data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    memcpy(matrix_B->data, matrix_A->data, capacityMatrix(matrix_A) * sizeof(double));
}

//��Ƭ��������A[left:right] [left-1, right)
/*
Matrix* indexVector(Matrix *vector_A, int left, int right){
	Matrix* vector_slice = NULL;

	if (vector_slice == NULL || vector_A == NULL) return;
		fprintf(stderr, "Memory allocation failed or the vector is Null\n");

	vector_slice = initMatrix();

}
*/
 
void printMatrix(Matrix *matrix)
{	
	int i =0;
	int j =0;
	for (i=0; i<matrix->rows; i++)
	{
		for(j=0; j<matrix->columns; j++){
			printf("%lf\t", matrix->data[i * matrix->columns + j]);	
		}
		//if (i != matrix->rows -1)
			printf("\n");
	}
}

//�����ԽǾ���
Matrix* diagMatrix(double* diag_elements, int n) {
    // ����һ�� n x n �ľ�������Ԫ�س�ʼ��Ϊ��
	int i = 0;
    Matrix* result = initMatrix(n, n); // ʹ�� initMatrix ��ʼ������

    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // �� diag_elements �е�Ԫ�����õ��������ĶԽ���
    for (i = 0; i < n; i++) {
        result->data[i * n + i] = diag_elements[i];  // ���öԽ���Ԫ��
    }

    return result;
}


//�ӷ�
Matrix* addMatrix(Matrix *matrix_A,Matrix *matrix_B)
{
	int i =0;
	int j =0;
	Matrix *matrix_C = NULL;

	if (matrix_A->rows == matrix_B->rows && matrix_A->columns == matrix_B->columns)
	{
		matrix_C = initMatrix(matrix_A->rows,matrix_A->columns);
		if (matrix_C == NULL) {
            fprintf(stderr, "Memory allocation failed for matrix_C\n");
            return NULL;
        }
		for (i=0;i<matrix_A->rows;i++)
		{
			for (j=0;j<matrix_A->columns;j++)
			{
				matrix_C->data[i*matrix_C->columns + j] = \
				matrix_A->data[i*matrix_A->columns + j] + matrix_B->data[i*matrix_B->columns + j];
			}
		}
		return matrix_C;
	}
	else 
	{
		printf("������״��һ��..\n");
		return NULL;
	}
}

//����
Matrix* subMatrix(Matrix *matrix_A,Matrix *matrix_B)
{
	int i =0;
	int j =0;
	Matrix *matrix_C = NULL;

	if (matrix_A->rows == matrix_B->rows && matrix_A->columns == matrix_B->columns)
	{
		matrix_C = initMatrix(matrix_A->rows,matrix_A->columns);
		if (matrix_C == NULL) {
            fprintf(stderr, "Memory allocation failed!\n");
            return NULL;
        }
		for (i=0;i<matrix_A->rows;i++)
		{
			for (j=0;j<matrix_A->columns;j++)
			{
				matrix_C->data[i*matrix_C->columns + j] = \
				matrix_A->data[i*matrix_A->columns + j] - matrix_B->data[i*matrix_B->columns + j];
			}
		}
		return matrix_C;
	}
	else 
	{
		printf("������״��һ��..\n");
		return NULL;
	}
}
 
//����˷�
Matrix* mulMatrix(Matrix *matrix_A,Matrix *matrix_B)
{
	int i =0;
	int j =0;
	int k =0;
	double sum = 0.0;
	Matrix *matrix_C = NULL;

	if (matrix_A->columns == matrix_B->rows)		//��==��
	{
		matrix_C = initMatrix(matrix_A->rows, matrix_B->columns);  
		if (matrix_C == NULL) {
            fprintf(stderr, "Memory allocation failed!\n");
            return NULL;
        }
		for (i=0; i<matrix_A->rows; i++)
		{
			for (j=0; j<matrix_B->columns; j++)
			{
				sum = 0.0;
				for (k=0; k<matrix_A->columns; k++)
				{
					sum += matrix_A->data[i * matrix_A->columns + k] * matrix_B->data[k * matrix_B->rows + j];
				}
				matrix_C->data[i*matrix_C->columns + j] = sum;
			}
		}
		return matrix_C;
	}
	else
	{
		printf("�������\n");
		return NULL;
	}
}
 
// Element-wise product
Matrix* dotProductMatrix(Matrix *matrix_A, Matrix *matrix_B) {
	int i = 0;
	int j = 0;
	Matrix *matrix_C = NULL;

    // ������A�;���B����״�Ƿ�һ��
    if (matrix_A->rows != matrix_B->rows || matrix_A->columns != matrix_B->columns) {
        printf("����A�;���B����״��һ�£��޷����е������\n");
        return NULL;
    }

    // ��ʼ���������C��������������A��B��ͬ
    matrix_C = initMatrix(matrix_A->rows, matrix_A->columns);
    if (matrix_C == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // ִ��Ԫ�ؼ���ĳ˷���Hadamard����
    for (i = 0; i < matrix_A->rows; i++) {
        for (j = 0; j < matrix_A->columns; j++) {
            matrix_C->data[i * matrix_C->columns + j] = 
                matrix_A->data[i * matrix_A->columns + j] * matrix_B->data[i * matrix_B->columns + j];
        }
    }

    return matrix_C;
}


//��ȡ�Ӿ���
Matrix* getsubMatrix(Matrix *matrix, int row, int col) {
	int i = 0;
	int j = 0;
	int minorRow = 0, minorCol = 0;
	Matrix *minor = NULL;

    // �������
    if (!matrix || matrix->rows <= 1 || matrix->columns <= 1) {
        fprintf(stderr, "Error: Invalid matrix or matrix dimensions are too small.\n");
        return NULL;
    }
    if (row < 0 || row >= matrix->rows || col < 0 || col >= matrix->columns) {
        fprintf(stderr, "Error: Row or column index out of bounds.\n");
        return NULL;
    }

    // �����µ��Ӿ���
    minor = initMatrix(matrix->rows - 1, matrix->columns - 1);
    if (!minor) {
        fprintf(stderr, "Error: Memory allocation failed for minor matrix.\n");
        return NULL;
    }

    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->columns; j++) {
            if (i != row && j != col) {
                // ���Ʒ�ָ���к��е�Ԫ��
                minor->data[minorRow * minor->columns + minorCol] = matrix->data[i * matrix->columns + j];
                minorCol++;

                // ���һ��������������������������һ��
                if (minorCol == minor->columns) {
                    minorCol = 0;
                    minorRow++;
                }
            }
        }
    }

    return minor;
}

// �����������ʽ
double detMatrix(Matrix *matrix) {
	double det = 0.0;
	int col = 0;
	double minorDet = 0.0;
	double cofactor = 0.0;

    // �������
    if (!matrix) {
        fprintf(stderr, "Error: Null matrix passed to Determinant.\n");
        return 0.0;
    }
    if (matrix->rows != matrix->columns) {
        fprintf(stderr, "Error: Non-square matrix has no determinant.\n");
        return 0.0;
    }
    if (matrix->rows == 0) {
		fprintf(stderr, "Error: The matrix has no element.\n");
        return 0.0; // �վ���
    }

    // ���������1x1 ����
    if (matrix->rows == 1) {
        return matrix->data[0];
    }

    // ���������2x2 ����
    if (matrix->rows == 2) {
        return matrix->data[0] * matrix->data[3] - matrix->data[1] * matrix->data[2];
    }

    // ͨ��������ݹ����
    
    for (col = 0; col < matrix->columns; col++) {
        // ��ȡ�Ӿ���
        Matrix *minor = getsubMatrix(matrix, 0, col);
        if (!minor) {
            fprintf(stderr, "Error: Failed to compute minor matrix.\n");
            return 0.0;
        }

        // �ݹ�����Ӿ��������ʽ
        minorDet = detMatrix(minor);
        freeMatrix(minor);

        // ʹ��������˹չ����ʽ
        cofactor = matrix->data[col]; // ��Ԫ��
        if (col % 2 != 0) {
            cofactor = -cofactor; // �������
        }
        det += cofactor * minorDet;
    }

    return det;
}

//����ת��
void transMatrix(Matrix *matrix)			//����
{
	int i =0;
	int j =0;

	if (matrix->rows == matrix->columns)
	{
		Matrix *matrixTemp = initMatrix(matrix->rows,matrix->columns);       	//����һ����ʱ����
		if (matrixTemp == NULL) {
            fprintf(stderr, "Memory allocation failed!\n");
            return;
        }
		copyMatrix(matrix,matrixTemp);	
 
		for (i = 0; i < matrix->rows; i++)
        {
            for (j = 0; j < matrix->columns; j++)
            {
                matrix->data[i * matrix->columns + j] = matrixTemp->data[j * matrix->columns + i];
            }
        }
		// �ͷ���ʱ������ڴ�
        freeMatrix(matrixTemp);
	}
	else
	{
		printf("ת�õľ������Ϊ����\n");
	}
}

//����ת�� �Ƿ���
Matrix* transMatrix2(Matrix *matrix) // �Ƿ����ת��
{
	int i =0;
	int j =0;
	Matrix *matrix_T = NULL;

	if (matrix->rows == matrix->columns){
		transMatrix(matrix);
		return matrix;
	}

    if (matrix->rows != matrix->columns)
    {
        // ����һ���µ�Matrix�ṹ�����洢ת�ú�ľ���
        matrix_T = initMatrix(matrix->columns, matrix->rows);
        if (matrix_T == NULL) {
            fprintf(stderr, "Memory allocation failed!\n");
            return NULL;
        }

        for (int i = 0; i < matrix->columns; i++)
        {
            for (int j = 0; j < matrix->rows; j++)
            {
                matrix_T->data[i * matrix_T->columns + j] = matrix->data[j * matrix->columns + i];
            }
        }
        return matrix_T;
    }
    else
    {
        printf("����ľ����Ƿ��󣬲���Ҫת��\n");
        return NULL;
    }
}

// 2-3�׾����Ƿ����
bool isInvertible(Matrix *matrix) {
	double det = 0.0;

    if (matrix->rows != matrix->columns) {
        return 0; // �Ƿ��󲻿���
    }

    // 2x2���������ʽ
    if (matrix->rows == 2) {
        det = matrix->data[0] * matrix->data[3] - matrix->data[1] * matrix->data[2];
        return fabs(det) > 1e-9;  // �������ʽ�Ƿ�ӽ���0
    }

    // 3x3���������ʽ
    if (matrix->rows == 3) {
        det = matrix->data[0] * (matrix->data[4] * matrix->data[8] - matrix->data[5] * matrix->data[7]) -
                     matrix->data[1] * (matrix->data[3] * matrix->data[8] - matrix->data[5] * matrix->data[6]) +
                     matrix->data[2] * (matrix->data[3] * matrix->data[7] - matrix->data[4] * matrix->data[6]);
        return fabs(det) > 1e-9;
    }

    // ���ڸ���ľ���������Ҫ�����ӵ��㷨
    return false; // Ŀǰֻ֧��2x2��3x3����
}

// �ж�n�׾����Ƿ����ĺ���
bool isInvertiblenx_n(Matrix *matrix) {
	double det = 0.0;

    if (matrix == NULL || matrix->rows != matrix->columns) {
        // �Ƿ����վ���ֱ�ӷ��ز�����
		fprintf(stderr, "Memory allocation failed or matrix is non-square\n");
        return false;
    }

    det = detMatrix(matrix);
    // �������ʽΪ0�����󲻿��棻���򣬾������
    return !FLOAT_EQUAL(det, 0.0);
}

// ������� 2-3��
Matrix* invMatrix(Matrix *matrix) {
	Matrix *inverse = NULL;
	double det = 0.1; //��֤������ȫ
	double inv_det = 0.0;

    if (!isInvertible(matrix)) {
        return NULL;
    }

    inverse = initMatrix(matrix->rows, matrix->columns);
    if (inverse == NULL) {
		fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }

    // 2x2�������
    if (matrix->rows == 2) {
        det = matrix->data[0] * matrix->data[3] - matrix->data[1] * matrix->data[2];
        if (det == 0) {
            return NULL;  // ��ֹ������
        }
        inverse->data[0] = matrix->data[3] / det;
        inverse->data[1] = -matrix->data[1] / det;
        inverse->data[2] = -matrix->data[2] / det;
        inverse->data[3] = matrix->data[0] / det;
    }

    // 3x3�������
    if (matrix->rows == 3) {
        det = matrix->data[0] * (matrix->data[4] * matrix->data[8] - matrix->data[5] * matrix->data[7]) -
                     matrix->data[1] * (matrix->data[3] * matrix->data[8] - matrix->data[5] * matrix->data[6]) +
                     matrix->data[2] * (matrix->data[3] * matrix->data[7] - matrix->data[4] * matrix->data[6]);
        if (det == 0) {
            return NULL;  // ��ֹ������
        }
        inv_det = 1 / det;

        inverse->data[0] = (matrix->data[4] * matrix->data[8] - matrix->data[5] * matrix->data[7]) * inv_det;
        inverse->data[1] = -(matrix->data[1] * matrix->data[8] - matrix->data[2] * matrix->data[7]) * inv_det;
        inverse->data[2] = (matrix->data[1] * matrix->data[5] - matrix->data[2] * matrix->data[4]) * inv_det;
        inverse->data[3] = -(matrix->data[3] * matrix->data[8] - matrix->data[5] * matrix->data[6]) * inv_det;
        inverse->data[4] = (matrix->data[0] * matrix->data[8] - matrix->data[2] * matrix->data[6]) * inv_det;
        inverse->data[5] = -(matrix->data[0] * matrix->data[5] - matrix->data[3] * matrix->data[2]) * inv_det;
        inverse->data[6] = (matrix->data[3] * matrix->data[7] - matrix->data[4] * matrix->data[6]) * inv_det;
        inverse->data[7] = -(matrix->data[0] * matrix->data[7] - matrix->data[1] * matrix->data[6]) * inv_det;
        inverse->data[8] = (matrix->data[0] * matrix->data[4] - matrix->data[1] * matrix->data[3]) * inv_det;
    }

    return inverse;
}

// ��ȫ�ĸ�˹��Ԫ��������� n�׾���
Matrix* safeMatrixInv(Matrix *matrix, int n) {
	int i, j, k, principal;
	double* aug = NULL; // �������
	double max = 0.0;
	double tmp = 0.0;
	double factor = 0.0;
	double diag_factor = 0.0;
	Matrix* inverse = NULL;

    if (matrix == NULL || n <= 0) {
        fprintf(stderr, "Invalid input matrix.\n");
        return NULL;
    }

	aug = (double*)malloc(sizeof(double) * n * n * 2); // �������
    if (aug == NULL) {
        fprintf(stderr, "Memory allocation failed for augmented matrix.\n");
        return NULL;
    }

    // ��ʼ���������
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            aug[i * 2 * n + j] = matrix->data[i * n + j]; // ԭ���󲿷�
            aug[i * 2 * n + (j + n)] = (i == j) ? 1.0 : 0.0; // ��λ���󲿷�
        }
    }

    // ��˹��Ԫ
    for (j = 0; j < n; j++) {
        // ѡ��Ԫ
        principal = j;
        max = fabs(aug[principal * 2 * n + j]);
        for (i = j + 1; i < n; i++) {
            if (fabs(aug[i * 2 * n + j]) > max) {
                principal = i;
                max = fabs(aug[i * 2 * n + j]);
            }
        }

        // �����Ԫ�Ƿ�Ϊ0
        if (fabs(max) < 1e-9) {
            fprintf(stderr, "Matrix is singular or nearly singular.\n");
            SAFE_FREE(aug);
            return NULL;
        }

        // ������
        if (j != principal) {
            for (k = 0; k < 2 * n; k++) {
                tmp = aug[j * 2 * n + k];
                aug[j * 2 * n + k] = aug[principal * 2 * n + k];
                aug[principal * 2 * n + k] = tmp;
            }
        }

        // ��Ϊ�н�����ʽ
        for (i = 0; i < n; i++) {
            if (i != j) {
                factor = aug[i * 2 * n + j] / aug[j * 2 * n + j];
                for (k = 0; k < 2 * n; k++) {
                    aug[i * 2 * n + k] -= factor * aug[j * 2 * n + k];
                }
            }
        }

        // ���Խ��߻�Ϊ1
        diag_factor = aug[j * 2 * n + j];
        for (k = 0; k < 2 * n; k++) {
            aug[j * 2 * n + k] /= diag_factor;
        }
    }

    // ��ȡ�����
	inverse = initMatrix(n, n);
    if (inverse == NULL) {
        fprintf(stderr, "Memory allocation failed for inverse matrix.\n");
        SAFE_FREE(aug);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            inverse->data[i * n + j] = aug[i * 2 * n + (j + n)];
        }
    }

    SAFE_FREE(aug); // �ͷ��������
    return inverse;
}


//������� �ҳ�  2-3��
Matrix* rightDivMatrix(Matrix *matrix_A, Matrix *matrix_B) {
	Matrix *inverse_B = NULL;
	Matrix *result = NULL;

    // ��� B �Ƿ��Ƿ���
    if (matrix_B->rows != matrix_B->columns) {
        printf("���� B �����Ƿ���\n");
        return NULL;
    }

    // ��� A �Ƿ����
    if (!isInvertible(matrix_A)) {
        printf("���� A ������\n");
        return NULL;
    }

    // ���� B �������
    inverse_B = invMatrix(matrix_B);
    if (inverse_B == NULL) {
        printf("������� B �������ʧ��\n");
        return NULL;
    }

    // ���� A * B^-1 ���õ���� X
    result = mulMatrix(matrix_A, inverse_B);
    if (result == NULL) {
        printf("����˷�ʧ��\n");
        freeMatrix(inverse_B);
        return NULL;
    }

    // �ͷ��м������ڴ�
    freeMatrix(inverse_B);

    return result;
}


//������� ���  2-3��
Matrix* leftDivMatrix(Matrix *matrix_A, Matrix *matrix_B) {
	Matrix *inverse_A = NULL;
	Matrix *result = NULL;

    // ��� A �Ƿ��Ƿ���
    if (matrix_A->rows != matrix_A->columns) {
        printf("���� A �����Ƿ���\n");
        return NULL;
    }

    // ��� A �Ƿ����
    if (!isInvertible(matrix_A)) {
        printf("���� A ������\n");
        return NULL;
    }

    // ���� A �������
    inverse_A = invMatrix(matrix_A);
    if (inverse_A == NULL) {
        printf("������� A �������ʧ��\n");
        return NULL;
    }

    // ���� A^-1 * B ���õ���� X
    result = mulMatrix(inverse_A, matrix_B);
    if (result == NULL) {
        printf("����˷�ʧ��\n");
        freeMatrix(inverse_A);
        return NULL;
    }

    // �ͷ��м������ڴ�
    freeMatrix(inverse_A);

    return result;
}

// n�׾������ �޷�������
Matrix* leftDivMatrix_n(Matrix *matrix_A,Matrix *matrix_B){
	Matrix *inverse_A = NULL;
	Matrix *result = NULL;

	// ��� A �Ƿ��Ƿ���
    if (matrix_A->rows != matrix_A->columns) {
        printf("���� A �����Ƿ���\n");
        return NULL;
    }

    // ��� A �Ƿ����
    if (!isInvertiblenx_n(matrix_A)) {
        printf("���� A ������\n");
        return NULL;
    }

    // ���� A �������
	inverse_A = safeMatrixInv(matrix_A, matrix_A->rows);
    if (inverse_A == NULL) {
        printf("������� A �������ʧ��\n");
        return NULL;
    }

    // ���� A^-1 * B ���õ���� X
    result = mulMatrix(inverse_A, matrix_B);
    if (result == NULL) {
        printf("����˷�ʧ��\n");
        freeMatrix(inverse_A);
        return NULL;
    }

    // �ͷ��м������ڴ�
    freeMatrix(inverse_A);

    return result;
}

// n�׾����ҳ� �޷�������
Matrix* rightDivMatrix_n(Matrix *matrix_A,Matrix *matrix_B){
	Matrix *inverse_B = NULL;
	Matrix *result = NULL;

	// ��� B �Ƿ��Ƿ���
    if (matrix_B->rows != matrix_B->columns) {
        printf("���� B �����Ƿ���\n");
        return NULL;
    }

    // ��� B �Ƿ����
    if (!isInvertiblenx_n(matrix_B)) {
        printf("���� B ������\n");
        return NULL;
    }

    // ���� B �������
	inverse_B = safeMatrixInv(matrix_B, matrix_B->rows);
    if (inverse_B == NULL) {
        printf("������� B �������ʧ��\n");
        return NULL;
    }

    // ���� A * B^-1 ���õ���� X
    result = mulMatrix(matrix_A, inverse_B);
    if (result == NULL) {
        printf("����˷�ʧ��\n");
        freeMatrix(inverse_B);
        return NULL;
    }

    // �ͷ��м������ڴ�
    freeMatrix(inverse_B);

    return result;
}

// �������
Matrix* crossProduct(Matrix *vector_A, Matrix *vector_B) {
	Matrix *result = NULL;

    // ����vector_A��vector_B��3x1�ľ���
    if (vector_A->rows != 3 || vector_A->columns != 1 || vector_B->rows != 3 || vector_B->columns != 1) {
        printf("Only 3D vectors are supported for cross product.\n");
        return NULL;
    }

    result = initMatrix(3, 1);
    if (result == NULL) {
		fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }

    // �����˵�ÿ������
    result->data[0] = vector_A->data[1] * vector_B->data[2] - vector_A->data[2] * vector_B->data[1]; // x
    result->data[1] = vector_A->data[2] * vector_B->data[0] - vector_A->data[0] * vector_B->data[2]; // y
    result->data[2] = vector_A->data[0] * vector_B->data[1] - vector_A->data[1] * vector_B->data[0]; // z

    return result;
}

// ������ A��B�������پ���һ������Ϊ 3 ��ά�ȡ�
Matrix* crossMatrix(Matrix *matrix_A, Matrix *matrix_B) {
	Matrix *result = NULL;
	int i = 0;
	Matrix vec_A;
	Matrix vec_B;
	Matrix *cross = NULL;
    // �������Ƿ�������Ҫ��3xN �� Nx3
    if (matrix_A->columns == 3 && matrix_B->columns == 3) {
        // 3x3������
        if (matrix_A->rows != matrix_B->rows) {
            printf("�����������ƥ�䣬�޷����в��\n");
            return NULL;
        }
        result = initMatrix(matrix_A->rows, 3);
        if (result == NULL) {
			fprintf(stderr, "Memory allocation failed!\n");
            return NULL;
        }

        // ��ÿһ�н����������
        for (i = 0; i < matrix_A->rows; i++) {
           //Matrix vec_A = {3, 1, &matrix_A->data[i * 3]};
           //Matrix vec_B = {3, 1, &matrix_B->data[i * 3]};
			vec_A.rows = 3;
            vec_A.columns = 1;
            vec_A.data = &matrix_A->data[i * 3];
            vec_B.rows = 3;
            vec_B.columns = 1;
            vec_B.data = &matrix_B->data[i * 3];
            cross = crossProduct(&vec_A, &vec_B);
            if (cross == NULL) {
				fprintf(stderr, "Memory allocation failed!\n");
                freeMatrix(result);
                return NULL;
            }
            memcpy(&result->data[i * 3], cross->data, 3 * sizeof(double));
            freeMatrix(cross);
        }
        return result;
    } else if (matrix_A->columns == 1 && matrix_B->columns == 1) {
        // ������� (3x1 ����)
        if (matrix_A->rows != 3 || matrix_B->rows != 3) {
            printf("ֻ�ܶ�3D���������в�ˡ�\n");
            return NULL;
        }
        return crossProduct(matrix_A, matrix_B);
    } else {
        printf("��֧�ֵľ�����״��\n");
        return NULL;
    }
}

// ����������һ����
Matrix* normVector1(Matrix *vector) {
	Matrix *norm = NULL;
	int i = 0;
	double sum = 0.0;

    if (vector->columns != 1) {
        return NULL; // ��֧����������һ�У�
    }

    norm = initMatrix(1, 1); // һ�������Ϊ����
    if (!norm) {
        return NULL;
    }

    for (i = 0; i < vector->rows; i++) {
        sum += fabs(vector->data[i]);
    }

    norm->data[0] = sum;
    return norm;
}

// ���������Ķ�����
Matrix* normVector2(Matrix *vector) {
	Matrix *norm = NULL;
	int i = 0;
	double sum = 0.0;
	
    if (vector->columns != 1) {
        return NULL; // ��֧����������һ�У�
    }

    norm = initMatrix(1, 1); // ���������Ϊ����
    if (!norm) {
        return NULL;
    }

    for (i = 0; i < vector->rows; i++) {
        sum += vector->data[i] * vector->data[i];
    }

    norm->data[0] = sqrt(sum);
    return norm;
}


//��x����ת����
Matrix* rotationMatrix_x (double element){
    Matrix* result = initMatrix(3, 3); 

    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }

    // �� elements �е�Ԫ�����õ��������ĶԽ���
	result->data[0] = 1;
	result->data[1] = 0;
	result->data[2] = 0;

	result->data[3] = 0;
	result->data[4] = cos(element);
	result->data[5] = sin(element);

	result->data[6] = 0;
	result->data[7] = -sin(element);
	result->data[8] = cos(element);
    return result;
}

//��y����ת����
Matrix* rotationMatrix_y (double element){
	Matrix* result = initMatrix(3, 3); 

    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }

    // �� elements �е�Ԫ�����õ��������ĶԽ���
	result->data[0] = cos(element);
	result->data[1] = 0;
	result->data[2] = -sin(element);

	result->data[3] = 0;
	result->data[4] = 1;
	result->data[5] = 0;

	result->data[6] = sin(element);
	result->data[7] = 0;
	result->data[8] = cos(element);
    return result;
}

//��z����ת����
Matrix* rotationMatrix_z (double element){
	Matrix* result = initMatrix(3, 3); 

    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }

    // �� elements �е�Ԫ�����õ��������ĶԽ���
	result->data[0] = cos(element);
	result->data[1] = sin(element);
	result->data[2] = 0;

	result->data[3] = -sin(element);
	result->data[4] = cos(element);
	result->data[5] = 0;

	result->data[6] = 0;
	result->data[7] = 0;
	result->data[8] = 1;
    return result;
}