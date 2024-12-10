#include "matrix.h"
#include <math.h>


Matrix* initMatrix(int rows,int columns)				//初始化一个矩阵
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
			free(matrix); // 释放已分配的matrix结构体内存
			return NULL;
		}
		memset(matrix->data, 0, sizeof(double)*rows*columns);
		return matrix;
	}
	else 
		return NULL;
} 

// 保证array大小与matrix.row*columns一致
void assignMatrix(Matrix *matrix,double *array) 		//给矩阵赋值
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
        //free(matrix->data); // 释放矩阵的数据存储区
		SAFE_FREE(matrix->data); // 安全释放矩阵的数据存储区
        free(matrix); // 释放矩阵结构体本身
        printf("ptr released successfully\n");
    }
}
 
void copyMatrix(Matrix *matrix_A, Matrix *matrix_B)
{
	if (matrix_B == NULL || matrix_A == NULL) return;
    matrix_B->rows = matrix_A->rows;
    matrix_B->columns = matrix_A->columns;
    if (matrix_B->data != NULL) {
        SAFE_FREE(matrix_B->data); // 释放旧的内存
    }
    matrix_B->data = (double*)malloc(capacityMatrix(matrix_A) * sizeof(double));
    if (matrix_B->data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    memcpy(matrix_B->data, matrix_A->data, capacityMatrix(matrix_A) * sizeof(double));
}

//切片索引向量A[left:right] [left-1, right)
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

//创建对角矩阵
Matrix* diagMatrix(double* diag_elements, int n) {
    // 创建一个 n x n 的矩阵，所有元素初始化为零
	int i = 0;
    Matrix* result = initMatrix(n, n); // 使用 initMatrix 初始化矩阵

    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // 将 diag_elements 中的元素设置到结果矩阵的对角线
    for (i = 0; i < n; i++) {
        result->data[i * n + i] = diag_elements[i];  // 设置对角线元素
    }

    return result;
}


//加法
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
		printf("矩阵形状不一致..\n");
		return NULL;
	}
}

//减法
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
		printf("矩阵形状不一致..\n");
		return NULL;
	}
}
 
//矩阵乘法
Matrix* mulMatrix(Matrix *matrix_A,Matrix *matrix_B)
{
	int i =0;
	int j =0;
	int k =0;
	double sum = 0.0;
	Matrix *matrix_C = NULL;

	if (matrix_A->columns == matrix_B->rows)		//列==行
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
		printf("不可相乘\n");
		return NULL;
	}
}
 
// Element-wise product
Matrix* dotProductMatrix(Matrix *matrix_A, Matrix *matrix_B) {
	int i = 0;
	int j = 0;
	Matrix *matrix_C = NULL;

    // 检查矩阵A和矩阵B的形状是否一致
    if (matrix_A->rows != matrix_B->rows || matrix_A->columns != matrix_B->columns) {
        printf("矩阵A和矩阵B的形状不一致，无法进行点积运算\n");
        return NULL;
    }

    // 初始化结果矩阵C，行数和列数与A、B相同
    matrix_C = initMatrix(matrix_A->rows, matrix_A->columns);
    if (matrix_C == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // 执行元素级别的乘法（Hadamard积）
    for (i = 0; i < matrix_A->rows; i++) {
        for (j = 0; j < matrix_A->columns; j++) {
            matrix_C->data[i * matrix_C->columns + j] = 
                matrix_A->data[i * matrix_A->columns + j] * matrix_B->data[i * matrix_B->columns + j];
        }
    }

    return matrix_C;
}


//提取子矩阵
Matrix* getsubMatrix(Matrix *matrix, int row, int col) {
	int i = 0;
	int j = 0;
	int minorRow = 0, minorCol = 0;
	Matrix *minor = NULL;

    // 参数检查
    if (!matrix || matrix->rows <= 1 || matrix->columns <= 1) {
        fprintf(stderr, "Error: Invalid matrix or matrix dimensions are too small.\n");
        return NULL;
    }
    if (row < 0 || row >= matrix->rows || col < 0 || col >= matrix->columns) {
        fprintf(stderr, "Error: Row or column index out of bounds.\n");
        return NULL;
    }

    // 创建新的子矩阵
    minor = initMatrix(matrix->rows - 1, matrix->columns - 1);
    if (!minor) {
        fprintf(stderr, "Error: Memory allocation failed for minor matrix.\n");
        return NULL;
    }

    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->columns; j++) {
            if (i != row && j != col) {
                // 复制非指定行和列的元素
                minor->data[minorRow * minor->columns + minorCol] = matrix->data[i * matrix->columns + j];
                minorCol++;

                // 如果一行填满，重置列索引并进入下一行
                if (minorCol == minor->columns) {
                    minorCol = 0;
                    minorRow++;
                }
            }
        }
    }

    return minor;
}

// 计算矩阵行列式
double detMatrix(Matrix *matrix) {
	double det = 0.0;
	int col = 0;
	double minorDet = 0.0;
	double cofactor = 0.0;

    // 参数检查
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
        return 0.0; // 空矩阵
    }

    // 基本情况：1x1 矩阵
    if (matrix->rows == 1) {
        return matrix->data[0];
    }

    // 基本情况：2x2 矩阵
    if (matrix->rows == 2) {
        return matrix->data[0] * matrix->data[3] - matrix->data[1] * matrix->data[2];
    }

    // 通用情况：递归计算
    
    for (col = 0; col < matrix->columns; col++) {
        // 获取子矩阵
        Matrix *minor = getsubMatrix(matrix, 0, col);
        if (!minor) {
            fprintf(stderr, "Error: Failed to compute minor matrix.\n");
            return 0.0;
        }

        // 递归计算子矩阵的行列式
        minorDet = detMatrix(minor);
        freeMatrix(minor);

        // 使用拉普拉斯展开公式
        cofactor = matrix->data[col]; // 主元素
        if (col % 2 != 0) {
            cofactor = -cofactor; // 交替符号
        }
        det += cofactor * minorDet;
    }

    return det;
}

//矩阵转置
void transMatrix(Matrix *matrix)			//方阵
{
	int i =0;
	int j =0;

	if (matrix->rows == matrix->columns)
	{
		Matrix *matrixTemp = initMatrix(matrix->rows,matrix->columns);       	//创建一个临时矩阵
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
		// 释放临时矩阵的内存
        freeMatrix(matrixTemp);
	}
	else
	{
		printf("转置的矩阵必须为方阵\n");
	}
}

//矩阵转置 非方阵
Matrix* transMatrix2(Matrix *matrix) // 非方阵的转置
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
        // 创建一个新的Matrix结构体来存储转置后的矩阵
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
        printf("输入的矩阵是方阵，不需要转置\n");
        return NULL;
    }
}

// 2-3阶矩阵是否可逆
bool isInvertible(Matrix *matrix) {
	double det = 0.0;

    if (matrix->rows != matrix->columns) {
        return 0; // 非方阵不可逆
    }

    // 2x2矩阵的行列式
    if (matrix->rows == 2) {
        det = matrix->data[0] * matrix->data[3] - matrix->data[1] * matrix->data[2];
        return fabs(det) > 1e-9;  // 检查行列式是否接近于0
    }

    // 3x3矩阵的行列式
    if (matrix->rows == 3) {
        det = matrix->data[0] * (matrix->data[4] * matrix->data[8] - matrix->data[5] * matrix->data[7]) -
                     matrix->data[1] * (matrix->data[3] * matrix->data[8] - matrix->data[5] * matrix->data[6]) +
                     matrix->data[2] * (matrix->data[3] * matrix->data[7] - matrix->data[4] * matrix->data[6]);
        return fabs(det) > 1e-9;
    }

    // 对于更大的矩阵，我们需要更复杂的算法
    return false; // 目前只支持2x2和3x3矩阵
}

// 判断n阶矩阵是否可逆的函数
bool isInvertiblenx_n(Matrix *matrix) {
	double det = 0.0;

    if (matrix == NULL || matrix->rows != matrix->columns) {
        // 非方阵或空矩阵直接返回不可逆
		fprintf(stderr, "Memory allocation failed or matrix is non-square\n");
        return false;
    }

    det = detMatrix(matrix);
    // 如果行列式为0，矩阵不可逆；否则，矩阵可逆
    return !FLOAT_EQUAL(det, 0.0);
}

// 矩阵的逆 2-3阶
Matrix* invMatrix(Matrix *matrix) {
	Matrix *inverse = NULL;
	double det = 0.1; //保证除法安全
	double inv_det = 0.0;

    if (!isInvertible(matrix)) {
        return NULL;
    }

    inverse = initMatrix(matrix->rows, matrix->columns);
    if (inverse == NULL) {
		fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }

    // 2x2矩阵的逆
    if (matrix->rows == 2) {
        det = matrix->data[0] * matrix->data[3] - matrix->data[1] * matrix->data[2];
        if (det == 0) {
            return NULL;  // 防止除以零
        }
        inverse->data[0] = matrix->data[3] / det;
        inverse->data[1] = -matrix->data[1] / det;
        inverse->data[2] = -matrix->data[2] / det;
        inverse->data[3] = matrix->data[0] / det;
    }

    // 3x3矩阵的逆
    if (matrix->rows == 3) {
        det = matrix->data[0] * (matrix->data[4] * matrix->data[8] - matrix->data[5] * matrix->data[7]) -
                     matrix->data[1] * (matrix->data[3] * matrix->data[8] - matrix->data[5] * matrix->data[6]) +
                     matrix->data[2] * (matrix->data[3] * matrix->data[7] - matrix->data[4] * matrix->data[6]);
        if (det == 0) {
            return NULL;  // 防止除以零
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

// 安全的高斯消元法求矩阵逆 n阶矩阵
Matrix* safeMatrixInv(Matrix *matrix, int n) {
	int i, j, k, principal;
	double* aug = NULL; // 增广矩阵
	double max = 0.0;
	double tmp = 0.0;
	double factor = 0.0;
	double diag_factor = 0.0;
	Matrix* inverse = NULL;

    if (matrix == NULL || n <= 0) {
        fprintf(stderr, "Invalid input matrix.\n");
        return NULL;
    }

	aug = (double*)malloc(sizeof(double) * n * n * 2); // 增广矩阵
    if (aug == NULL) {
        fprintf(stderr, "Memory allocation failed for augmented matrix.\n");
        return NULL;
    }

    // 初始化增广矩阵
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            aug[i * 2 * n + j] = matrix->data[i * n + j]; // 原矩阵部分
            aug[i * 2 * n + (j + n)] = (i == j) ? 1.0 : 0.0; // 单位矩阵部分
        }
    }

    // 高斯消元
    for (j = 0; j < n; j++) {
        // 选主元
        principal = j;
        max = fabs(aug[principal * 2 * n + j]);
        for (i = j + 1; i < n; i++) {
            if (fabs(aug[i * 2 * n + j]) > max) {
                principal = i;
                max = fabs(aug[i * 2 * n + j]);
            }
        }

        // 检查主元是否为0
        if (fabs(max) < 1e-9) {
            fprintf(stderr, "Matrix is singular or nearly singular.\n");
            SAFE_FREE(aug);
            return NULL;
        }

        // 交换行
        if (j != principal) {
            for (k = 0; k < 2 * n; k++) {
                tmp = aug[j * 2 * n + k];
                aug[j * 2 * n + k] = aug[principal * 2 * n + k];
                aug[principal * 2 * n + k] = tmp;
            }
        }

        // 化为行阶梯形式
        for (i = 0; i < n; i++) {
            if (i != j) {
                factor = aug[i * 2 * n + j] / aug[j * 2 * n + j];
                for (k = 0; k < 2 * n; k++) {
                    aug[i * 2 * n + k] -= factor * aug[j * 2 * n + k];
                }
            }
        }

        // 主对角线化为1
        diag_factor = aug[j * 2 * n + j];
        for (k = 0; k < 2 * n; k++) {
            aug[j * 2 * n + k] /= diag_factor;
        }
    }

    // 提取逆矩阵
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

    SAFE_FREE(aug); // 释放增广矩阵
    return inverse;
}


//矩阵除法 右除  2-3阶
Matrix* rightDivMatrix(Matrix *matrix_A, Matrix *matrix_B) {
	Matrix *inverse_B = NULL;
	Matrix *result = NULL;

    // 检查 B 是否是方阵
    if (matrix_B->rows != matrix_B->columns) {
        printf("矩阵 B 必须是方阵\n");
        return NULL;
    }

    // 检查 A 是否可逆
    if (!isInvertible(matrix_A)) {
        printf("矩阵 A 不可逆\n");
        return NULL;
    }

    // 计算 B 的逆矩阵
    inverse_B = invMatrix(matrix_B);
    if (inverse_B == NULL) {
        printf("计算矩阵 B 的逆矩阵失败\n");
        return NULL;
    }

    // 计算 A * B^-1 来得到结果 X
    result = mulMatrix(matrix_A, inverse_B);
    if (result == NULL) {
        printf("矩阵乘法失败\n");
        freeMatrix(inverse_B);
        return NULL;
    }

    // 释放中间结果的内存
    freeMatrix(inverse_B);

    return result;
}


//矩阵除法 左除  2-3阶
Matrix* leftDivMatrix(Matrix *matrix_A, Matrix *matrix_B) {
	Matrix *inverse_A = NULL;
	Matrix *result = NULL;

    // 检查 A 是否是方阵
    if (matrix_A->rows != matrix_A->columns) {
        printf("矩阵 A 必须是方阵\n");
        return NULL;
    }

    // 检查 A 是否可逆
    if (!isInvertible(matrix_A)) {
        printf("矩阵 A 不可逆\n");
        return NULL;
    }

    // 计算 A 的逆矩阵
    inverse_A = invMatrix(matrix_A);
    if (inverse_A == NULL) {
        printf("计算矩阵 A 的逆矩阵失败\n");
        return NULL;
    }

    // 计算 A^-1 * B 来得到结果 X
    result = mulMatrix(inverse_A, matrix_B);
    if (result == NULL) {
        printf("矩阵乘法失败\n");
        freeMatrix(inverse_A);
        return NULL;
    }

    // 释放中间结果的内存
    freeMatrix(inverse_A);

    return result;
}

// n阶矩阵左除 无方阵限制
Matrix* leftDivMatrix_n(Matrix *matrix_A,Matrix *matrix_B){
	Matrix *inverse_A = NULL;
	Matrix *result = NULL;

	// 检查 A 是否是方阵
    if (matrix_A->rows != matrix_A->columns) {
        printf("矩阵 A 必须是方阵\n");
        return NULL;
    }

    // 检查 A 是否可逆
    if (!isInvertiblenx_n(matrix_A)) {
        printf("矩阵 A 不可逆\n");
        return NULL;
    }

    // 计算 A 的逆矩阵
	inverse_A = safeMatrixInv(matrix_A, matrix_A->rows);
    if (inverse_A == NULL) {
        printf("计算矩阵 A 的逆矩阵失败\n");
        return NULL;
    }

    // 计算 A^-1 * B 来得到结果 X
    result = mulMatrix(inverse_A, matrix_B);
    if (result == NULL) {
        printf("矩阵乘法失败\n");
        freeMatrix(inverse_A);
        return NULL;
    }

    // 释放中间结果的内存
    freeMatrix(inverse_A);

    return result;
}

// n阶矩阵右除 无方阵限制
Matrix* rightDivMatrix_n(Matrix *matrix_A,Matrix *matrix_B){
	Matrix *inverse_B = NULL;
	Matrix *result = NULL;

	// 检查 B 是否是方阵
    if (matrix_B->rows != matrix_B->columns) {
        printf("矩阵 B 必须是方阵\n");
        return NULL;
    }

    // 检查 B 是否可逆
    if (!isInvertiblenx_n(matrix_B)) {
        printf("矩阵 B 不可逆\n");
        return NULL;
    }

    // 计算 B 的逆矩阵
	inverse_B = safeMatrixInv(matrix_B, matrix_B->rows);
    if (inverse_B == NULL) {
        printf("计算矩阵 B 的逆矩阵失败\n");
        return NULL;
    }

    // 计算 A * B^-1 来得到结果 X
    result = mulMatrix(matrix_A, inverse_B);
    if (result == NULL) {
        printf("矩阵乘法失败\n");
        freeMatrix(inverse_B);
        return NULL;
    }

    // 释放中间结果的内存
    freeMatrix(inverse_B);

    return result;
}

// 向量叉积
Matrix* crossProduct(Matrix *vector_A, Matrix *vector_B) {
	Matrix *result = NULL;

    // 假设vector_A和vector_B是3x1的矩阵
    if (vector_A->rows != 3 || vector_A->columns != 1 || vector_B->rows != 3 || vector_B->columns != 1) {
        printf("Only 3D vectors are supported for cross product.\n");
        return NULL;
    }

    result = initMatrix(3, 1);
    if (result == NULL) {
		fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }

    // 计算叉乘的每个分量
    result->data[0] = vector_A->data[1] * vector_B->data[2] - vector_A->data[2] * vector_B->data[1]; // x
    result->data[1] = vector_A->data[2] * vector_B->data[0] - vector_A->data[0] * vector_B->data[2]; // y
    result->data[2] = vector_A->data[0] * vector_B->data[1] - vector_A->data[1] * vector_B->data[0]; // z

    return result;
}

// 矩阵叉乘 A和B必须至少具有一个长度为 3 的维度。
Matrix* crossMatrix(Matrix *matrix_A, Matrix *matrix_B) {
	Matrix *result = NULL;
	int i = 0;
	Matrix vec_A;
	Matrix vec_B;
	Matrix *cross = NULL;
    // 检查矩阵是否满足叉乘要求：3xN 或 Nx3
    if (matrix_A->columns == 3 && matrix_B->columns == 3) {
        // 3x3矩阵叉乘
        if (matrix_A->rows != matrix_B->rows) {
            printf("矩阵的行数不匹配，无法进行叉乘\n");
            return NULL;
        }
        result = initMatrix(matrix_A->rows, 3);
        if (result == NULL) {
			fprintf(stderr, "Memory allocation failed!\n");
            return NULL;
        }

        // 对每一行进行向量叉乘
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
        // 向量叉乘 (3x1 矩阵)
        if (matrix_A->rows != 3 || matrix_B->rows != 3) {
            printf("只能对3D列向量进行叉乘。\n");
            return NULL;
        }
        return crossProduct(matrix_A, matrix_B);
    } else {
        printf("不支持的矩阵形状。\n");
        return NULL;
    }
}

// 计算向量的一范数
Matrix* normVector1(Matrix *vector) {
	Matrix *norm = NULL;
	int i = 0;
	double sum = 0.0;

    if (vector->columns != 1) {
        return NULL; // 仅支持行向量（一列）
    }

    norm = initMatrix(1, 1); // 一范数结果为标量
    if (!norm) {
        return NULL;
    }

    for (i = 0; i < vector->rows; i++) {
        sum += fabs(vector->data[i]);
    }

    norm->data[0] = sum;
    return norm;
}

// 计算向量的二范数
Matrix* normVector2(Matrix *vector) {
	Matrix *norm = NULL;
	int i = 0;
	double sum = 0.0;
	
    if (vector->columns != 1) {
        return NULL; // 仅支持行向量（一列）
    }

    norm = initMatrix(1, 1); // 二范数结果为标量
    if (!norm) {
        return NULL;
    }

    for (i = 0; i < vector->rows; i++) {
        sum += vector->data[i] * vector->data[i];
    }

    norm->data[0] = sqrt(sum);
    return norm;
}


//绕x轴旋转矩阵
Matrix* rotationMatrix_x (double element){
    Matrix* result = initMatrix(3, 3); 

    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }

    // 将 elements 中的元素设置到结果矩阵的对角线
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

//绕y轴旋转矩阵
Matrix* rotationMatrix_y (double element){
	Matrix* result = initMatrix(3, 3); 

    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }

    // 将 elements 中的元素设置到结果矩阵的对角线
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

//绕z轴旋转矩阵
Matrix* rotationMatrix_z (double element){
	Matrix* result = initMatrix(3, 3); 

    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }

    // 将 elements 中的元素设置到结果矩阵的对角线
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