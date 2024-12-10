#include <assert.h>
#include <math.h>
#include "matrix.h"

// 测试用例 1: 初始化矩阵
void testinitMatrix() {
	printf("testinitMatrix\n");
    Matrix *matrix = initMatrix(2, 2);

	// 使用 FLOAT_EQUAL 宏替代直接比较
    assert(matrix != NULL);
    //assert(FLOAT_EQUAL(matrix->rows, 2.0)); //无法重载宏 故强制类型转换
    //assert(FLOAT_EQUAL(matrix->columns, 2.0));
	assert(matrix->rows == 2);
    assert(matrix->columns == 2);

    freeMatrix(matrix);
}

//测试用例 2: 矩阵赋值
void testValueMatrix() {
	printf("testValueMatrix\n");
    Matrix *matrix = initMatrix(2, 2);
    double array[] = {1, 2, 3, 4};
    assignMatrix(matrix, array);

    assert(FLOAT_EQUAL(matrix->data[0], 1));
    assert(FLOAT_EQUAL(matrix->data[1], 2));
    assert(FLOAT_EQUAL(matrix->data[2], 3));
    assert(FLOAT_EQUAL(matrix->data[3], 4));

    freeMatrix(matrix);
}

//测试用例 3: 矩阵加法
void testaddMatrix() {
	printf("testaddMatrix\n");
    Matrix *matrixA = initMatrix(2, 2);
    Matrix *matrixB = initMatrix(2, 2);
    double arrayA[] = {1, 2, 3, 4};
    double arrayB[] = {5, 6, 7, 8};
    assignMatrix(matrixA, arrayA);
    assignMatrix(matrixB, arrayB);

    Matrix *result = addMatrix(matrixA, matrixB);

    assert(FLOAT_EQUAL(result->data[0], 6));
    assert(FLOAT_EQUAL(result->data[1], 8));
    assert(FLOAT_EQUAL(result->data[2], 10));
    assert(FLOAT_EQUAL(result->data[3], 12));

    freeMatrix(matrixA);
    freeMatrix(matrixB);
    freeMatrix(result);
}

//测试用例 4: 矩阵减法
void testsubMatrix() {
	printf("testsubMatrix\n");
    Matrix *matrixA = initMatrix(2, 2);
    Matrix *matrixB = initMatrix(2, 2);
    double arrayA[] = {5, 6, 7, 8};
    double arrayB[] = {1, 2, 3, 4};
    assignMatrix(matrixA, arrayA);
    assignMatrix(matrixB, arrayB);

    Matrix *result = subMatrix(matrixA, matrixB);

    assert(FLOAT_EQUAL(result->data[0], 4));
    assert(FLOAT_EQUAL(result->data[1], 4));
    assert(FLOAT_EQUAL(result->data[2], 4));
    assert(FLOAT_EQUAL(result->data[3], 4));

    freeMatrix(matrixA);
    freeMatrix(matrixB);
    freeMatrix(result);
}

//测试用例 5: 矩阵乘法
void testmulMatrix() {
	printf("testmulMatrix\n");
    Matrix *matrixA = initMatrix(2, 2);
    Matrix *matrixB = initMatrix(2, 2);
    double arrayA[] = {1, 2, 3, 4};
    double arrayB[] = {5, 6, 7, 8};
    assignMatrix(matrixA, arrayA);
    assignMatrix(matrixB, arrayB);
    
	Matrix *result = mulMatrix(matrixA, matrixB);

    assert(FLOAT_EQUAL(result->data[0], 19));
    assert(FLOAT_EQUAL(result->data[1], 22));
    assert(FLOAT_EQUAL(result->data[2], 43));
    assert(FLOAT_EQUAL(result->data[3], 50));
    
	freeMatrix(matrixA);
    freeMatrix(matrixB);
    freeMatrix(result);
}

void testDotProductMatrix() {
    printf("testDotProductMatrix\n");

    // 测试1：矩阵形状一致，执行点积
    Matrix *matrix_A = initMatrix(2, 3);
    double array_A[] = {1, 2, 3, 4, 5, 6};
    assignMatrix(matrix_A, array_A);

    Matrix *matrix_B = initMatrix(2, 3);
    double array_B[] = {6, 5, 4, 3, 2, 1};
    assignMatrix(matrix_B, array_B);

    Matrix *result = dotProductMatrix(matrix_A, matrix_B);
    assert(result != NULL);
    assert(result->rows == 2 && result->columns == 3);

    // 验证点积结果：对应元素相乘
    assert(FLOAT_EQUAL(result->data[0], 1 * 6));
    assert(FLOAT_EQUAL(result->data[1], 2 * 5));
    assert(FLOAT_EQUAL(result->data[2], 3 * 4));
    assert(FLOAT_EQUAL(result->data[3], 4 * 3));
    assert(FLOAT_EQUAL(result->data[4], 5 * 2));
    assert(FLOAT_EQUAL(result->data[5], 6 * 1));

    freeMatrix(matrix_A);
    freeMatrix(matrix_B);
    freeMatrix(result);

    // 测试2：矩阵形状不一致，无法执行点积
    Matrix *matrix_C = initMatrix(3, 2);
    double array_C[] = {1, 2, 3, 4, 5, 6};
    assignMatrix(matrix_C, array_C);

    Matrix *matrix_D = initMatrix(2, 3);
    double array_D[] = {6, 5, 4, 3, 2, 1};
    assignMatrix(matrix_D, array_D);

    result = dotProductMatrix(matrix_C, matrix_D);
    assert(result == NULL);  // 矩阵形状不一致，返回NULL

    freeMatrix(matrix_C);
    freeMatrix(matrix_D);

    // 测试3：矩阵包含零元素，执行点积
    Matrix *matrix_E = initMatrix(2, 3);
    double array_E[] = {0, 2, 3, 4, 0, 6};
    assignMatrix(matrix_E, array_E);

    Matrix *matrix_F = initMatrix(2, 3);
    double array_F[] = {1, 0, 4, 3, 0, 5};
    assignMatrix(matrix_F, array_F);

    result = dotProductMatrix(matrix_E, matrix_F);
    assert(result != NULL);

    // 验证结果
    assert(FLOAT_EQUAL(result->data[0], 0 * 1));
    assert(FLOAT_EQUAL(result->data[1], 2 * 0));
    assert(FLOAT_EQUAL(result->data[2], 3 * 4));
    assert(FLOAT_EQUAL(result->data[3], 4 * 3));
    assert(FLOAT_EQUAL(result->data[4], 0 * 0));
    assert(FLOAT_EQUAL(result->data[5], 6 * 5));

    freeMatrix(matrix_E);
    freeMatrix(matrix_F);
    freeMatrix(result);

    // 测试4：矩阵包含负数，执行点积
    Matrix *matrix_G = initMatrix(2, 3);
    double array_G[] = {-1, 2, -3, 4, -5, 6};
    assignMatrix(matrix_G, array_G);

    Matrix *matrix_H = initMatrix(2, 3);
    double array_H[] = {6, -5, 4, -3, 2, -1};
    assignMatrix(matrix_H, array_H);

    result = dotProductMatrix(matrix_G, matrix_H);
    assert(result != NULL);

    // 验证结果：对应元素相乘，负数处理是否正确
    assert(FLOAT_EQUAL(result->data[0], (-1) * 6));
    assert(FLOAT_EQUAL(result->data[1], 2 * (-5)));
    assert(FLOAT_EQUAL(result->data[2], (-3) * 4));
    assert(FLOAT_EQUAL(result->data[3], 4 * (-3)));
    assert(FLOAT_EQUAL(result->data[4], (-5) * 2));
    assert(FLOAT_EQUAL(result->data[5], 6 * (-1)));

    freeMatrix(matrix_G);
    freeMatrix(matrix_H);
    freeMatrix(result);
}

void testDeterminant() {
    printf("testDeterminant\n");

    // 测试 1x1 矩阵
    Matrix *matrix1x1 = initMatrix(1, 1);
    double array1x1[] = {5.0};
    assignMatrix(matrix1x1, array1x1);
    assert(FLOAT_EQUAL(detMatrix(matrix1x1), 5.0));
    freeMatrix(matrix1x1);

    // 测试 2x2 矩阵（行列式非零）
    Matrix *matrix2x2 = initMatrix(2, 2);
    double array2x2[] = {1.0, 2.0, 3.0, 4.0}; // 行列式为 -2
    assignMatrix(matrix2x2, array2x2);
    assert(FLOAT_EQUAL(detMatrix(matrix2x2), -2.0));
    freeMatrix(matrix2x2);

    // 测试 2x2 矩阵（行列式为 0）
    Matrix *matrix2x2Zero = initMatrix(2, 2);
    double array2x2Zero[] = {1.0, 2.0, 2.0, 4.0}; // 行列式为 0
    assignMatrix(matrix2x2Zero, array2x2Zero);
    assert(FLOAT_EQUAL(detMatrix(matrix2x2Zero), 0.0));
    freeMatrix(matrix2x2Zero);

    // 测试 3x3 矩阵（行列式非零）
    Matrix *matrix3x3 = initMatrix(3, 3);
    double array3x3[] = {1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0}; // 行列式为 1
    assignMatrix(matrix3x3, array3x3);
    assert(FLOAT_EQUAL(detMatrix(matrix3x3), 1.0));
    freeMatrix(matrix3x3);

    // 测试 3x3 矩阵（行列式为 0）
    Matrix *matrix3x3Zero = initMatrix(3, 3);
    double array3x3Zero[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}; // 行列式为 0
    assignMatrix(matrix3x3Zero, array3x3Zero);
    assert(FLOAT_EQUAL(detMatrix(matrix3x3Zero), 0.0));
    freeMatrix(matrix3x3Zero);

    // 测试非方阵
    Matrix *nonSquareMatrix = initMatrix(2, 3);
    double arrayNonSquare[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    assignMatrix(nonSquareMatrix, arrayNonSquare);
    assert(FLOAT_EQUAL(detMatrix(nonSquareMatrix), 0.0)); // 非方阵应返回 0.0
    freeMatrix(nonSquareMatrix);

    // 测试空矩阵
    Matrix *nullMatrix = NULL;
	assert(FLOAT_EQUAL(detMatrix(nullMatrix), 0.0));
}

//测试用例 6: 矩阵转置（方阵）
void testTransMatrix() {
	printf("testTransMatrix\n");
    Matrix *matrix = initMatrix(2, 2);
    double array[] = {1, 2, 3, 4};
    assignMatrix(matrix, array);

    transMatrix(matrix);
    
	assert(FLOAT_EQUAL(matrix->data[0], 1));
    assert(FLOAT_EQUAL(matrix->data[1], 3));
    assert(FLOAT_EQUAL(matrix->data[2], 2));
    assert(FLOAT_EQUAL(matrix->data[3], 4));

    freeMatrix(matrix);
}

//测试用例 7: 矩阵转置（非方阵）
void testTransMatrix2() {
	printf("testTransMatrix2\n");
    Matrix *matrix = initMatrix(2, 3);
    double array[] = {1, 2, 3, 4, 5, 6};
    assignMatrix(matrix, array);

    Matrix *result = transMatrix2(matrix);

    assert(result->rows == 3);
    assert(result->columns == 2);
    
	assert(FLOAT_EQUAL(result->data[0], 1));
    assert(FLOAT_EQUAL(result->data[1], 4));
    assert(FLOAT_EQUAL(result->data[2], 2));
    assert(FLOAT_EQUAL(result->data[3], 5));
    assert(FLOAT_EQUAL(result->data[4], 3));
    assert(FLOAT_EQUAL(result->data[5], 6));

    freeMatrix(matrix);
    freeMatrix(result);
}


//判断2x2矩阵是否可逆
void testIsInvertible2x2IsInvertible() {
	printf("testIsInvertible2x2IsInvertible\n");
    Matrix *matrix = initMatrix(2, 2);
    double array[] = {1, 2, 3, 4}; // 行列式为 -2，非零
	Matrix *matrix2 = initMatrix(2, 2);
    double array2[] = {1, 0, 1, 0}; // 行列式为 0

    assignMatrix(matrix, array);
	assignMatrix(matrix2, array2);
    assert(isInvertible(matrix) == true);
    freeMatrix(matrix);

    assert(isInvertible(matrix2) == false);    
    freeMatrix(matrix2);
}


//判断3x3矩阵是否可逆
void testIsInvertible3x3IsInvertible() {
	printf("testIsInvertible3x3IsInvertible\n");
    Matrix *matrix = initMatrix(3, 3);
    double array[] = {1, 2, 5, 2, 3, 4, 3, 4, 5}; // 行列式为 -2，非零
	Matrix *matrix2 = initMatrix(3, 3);
    double array2[] = {1, 2, 3, 2, 4, 3, 3, 6, 5}; // 行列式为 0
    assignMatrix(matrix, array);
	assignMatrix(matrix2, array2);

    assert(isInvertible(matrix) == true);
    freeMatrix(matrix);

    assert(isInvertible(matrix2) == false);
    freeMatrix(matrix2);
}

//测试n阶矩阵是否可逆
void testIsInvertiblenxn() {
    printf("testIsInvertiblenxn\n");

    // 测试 1x1 矩阵（可逆）
    Matrix *matrix1x1 = initMatrix(1, 1);
    double array1x1[] = {5.0}; // 行列式为 5.0，不为零
    assignMatrix(matrix1x1, array1x1);
    assert(isInvertiblenx_n(matrix1x1) == true);  // 可逆
    freeMatrix(matrix1x1);

    // 测试 2x2 矩阵（行列式非零）
    Matrix *matrix2x2 = initMatrix(2, 2);
    double array2x2[] = {1.0, 2.0, 3.0, 4.0}; // 行列式为 -2，不为零
    assignMatrix(matrix2x2, array2x2);
    assert(isInvertiblenx_n(matrix2x2) == true);  // 可逆
    freeMatrix(matrix2x2);

    // 测试 2x2 矩阵（行列式为零）
    Matrix *matrix2x2Zero = initMatrix(2, 2);
    double array2x2Zero[] = {1.0, 2.0, 2.0, 4.0}; // 行列式为 0，不可逆
    assignMatrix(matrix2x2Zero, array2x2Zero);
    assert(isInvertiblenx_n(matrix2x2Zero) == false); // 不可逆
    freeMatrix(matrix2x2Zero);

    // 测试 3x3 矩阵（行列式非零）
    Matrix *matrix3x3 = initMatrix(3, 3);
    double array3x3[] = {1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0}; // 行列式为 1，不为零
    assignMatrix(matrix3x3, array3x3);
    assert(isInvertiblenx_n(matrix3x3) == true);  // 可逆
    freeMatrix(matrix3x3);

    // 测试 3x3 矩阵（行列式为零）
    Matrix *matrix3x3Zero = initMatrix(3, 3);
    double array3x3Zero[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}; // 行列式为 0，不可逆
    assignMatrix(matrix3x3Zero, array3x3Zero);
    assert(isInvertiblenx_n(matrix3x3Zero) == false);  // 不可逆
    freeMatrix(matrix3x3Zero);

    // 测试非方阵（2x3）
    Matrix *nonSquareMatrix = initMatrix(2, 3);
    double arrayNonSquare[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    assignMatrix(nonSquareMatrix, arrayNonSquare);
    assert(isInvertiblenx_n(nonSquareMatrix) == false); // 非方阵，不可逆
    freeMatrix(nonSquareMatrix);

    // 测试空矩阵
    Matrix *nullMatrix = NULL;
    assert(isInvertiblenx_n(nullMatrix) == false); // 空矩阵，不可逆
}


//测试用例 x：求矩阵的逆
void testInvMatrix2x2() {
    printf("testInvMatrix2x2\n");

    // 测试可逆的2x2矩阵
    Matrix *matrix = initMatrix(2, 2);
    double array[] = {4, 7, 2, 6}; // 行列式为 10，非零
    assignMatrix(matrix, array);

    Matrix *inverse = invMatrix(matrix);
    assert(inverse != NULL);

    // 验证逆矩阵
    double expected[] = {0.6, -0.7, -0.2, 0.4};
    for (int i = 0; i < 4; i++) {
        assert(FLOAT_EQUAL(inverse->data[i], expected[i]));
    }

    freeMatrix(matrix);
    freeMatrix(inverse);

    // 测试不可逆的2x2矩阵
    Matrix *matrix2 = initMatrix(2, 2);
    double array2[] = {1, 2, 2, 4}; // 行列式为 0
    assignMatrix(matrix2, array2);

    Matrix *inverse2 = invMatrix(matrix2);
    assert(inverse2 == NULL);

    freeMatrix(matrix2);
}

void testInvMatrix3x3() {
    printf("testInvMatrix3x3\n");

    // 测试可逆的3x3矩阵
    Matrix *matrix = initMatrix(3, 3);
    double array[] = {1, 2, 3, 0, 1, 4, 5, 6, 0}; // 行列式为 1，非零
    assignMatrix(matrix, array);

    Matrix *inverse = invMatrix(matrix);
    assert(inverse != NULL);

    // 验证逆矩阵
    double expected[] = {-24, 18, 5, 20, -15, -4, -5, 4, 1}; // 经过计算得出的逆矩阵
    for (int i = 0; i < 9; i++) {
        assert(FLOAT_EQUAL(inverse->data[i], expected[i]));
    }

    freeMatrix(matrix);
    freeMatrix(inverse);

    // 测试不可逆的3x3矩阵
    Matrix *matrix2 = initMatrix(3, 3);
    double array2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9}; // 行列式为 0
    assignMatrix(matrix2, array2);

    Matrix *inverse2 = invMatrix(matrix2);
    assert(inverse2 == NULL);

    freeMatrix(matrix2);
}

void testInvMatrixNonSquare() {
    printf("testInvMatrixNonSquare\n");

    // 测试非方阵
    Matrix *matrix = initMatrix(2, 3);
    double array[] = {1, 2, 3, 4, 5, 6};
    assignMatrix(matrix, array);

    Matrix *inverse = invMatrix(matrix);
    assert(inverse == NULL);

    freeMatrix(matrix);
}

// 测试n阶安全矩阵求逆函数
void testSafeMatrixInver() {
    int n = 3;
	Matrix* matrix = initMatrix(3, 3);
    double src[] = {2, -1, 0, 
                    -1, 2, -1, 
                    0, -1, 2};
	assignMatrix(matrix, src);

	Matrix* matrix2 = safeMatrixInv(matrix, n);
    double* inverse =  matrix2->data;

    if (inverse) {
        printf("Inverse matrix:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%8.4f ", inverse[i * n + j]);
            }
            printf("\n");
        }
        SAFE_FREE(inverse);
    } else {
        printf("Failed to compute the inverse matrix.\n");
    }
}


//测试用例 8: 矩阵除法（2x2）
void testDivMatrix2x2() {
	printf("testDivMatrix2x2\n");
	Matrix *matrixI = initMatrix(2, 2);
    Matrix *matrixA = initMatrix(2, 2);
    Matrix *matrixB = initMatrix(2, 2);
	double arrayI[] = {1, 0, 0, 1}; // 单位矩阵I的元素
    double arrayA[] = {1, 2, 3, 4}; // 矩阵A的元素
    double arrayB[] = {-2, 1, 1.5, -0.5}; // 矩阵B的元素，这里应该是矩阵A的逆矩阵

    // 确保矩阵A是可逆的，并且矩阵B是矩阵A的逆矩阵
	assignMatrix(matrixI, arrayI);
    assignMatrix(matrixA, arrayA);
    assignMatrix(matrixB, arrayB);
    Matrix *result = rightDivMatrix(matrixA, matrixB);
    
	printMatrix(result);
    // 检查结果
    assert(FLOAT_EQUAL(result->data[0], 7));
    assert(FLOAT_EQUAL(result->data[1], 10));
    assert(FLOAT_EQUAL(result->data[2], 15));
    assert(FLOAT_EQUAL(result->data[3], 22));
    
    freeMatrix(matrixA);
    freeMatrix(matrixB);
    freeMatrix(result);
}

//测试用例 9: 矩阵除法（3x3）
void testDivMatrix3x3() {
	printf("testDivMatrix3x3\n");
    Matrix *matrixA = initMatrix(3, 3);
    Matrix *matrixB = initMatrix(3, 3);
    double arrayA[] = {1, 1, 0, 0, 1, 0, 1, 0, 1};
    double arrayB[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    assignMatrix(matrixA, arrayA);
    assignMatrix(matrixB, arrayB);

    Matrix *result = leftDivMatrix(matrixA, matrixB);
	printMatrix(result);

    assert(FLOAT_EQUAL(result->data[0], 1.0));
    assert(FLOAT_EQUAL(result->data[1], -1.0));
    assert(FLOAT_EQUAL(result->data[2], 0.0));
    assert(FLOAT_EQUAL(result->data[3], 0.0));
    assert(FLOAT_EQUAL(result->data[4], 1.0));
    assert(FLOAT_EQUAL(result->data[5], 0.0));
    assert(FLOAT_EQUAL(result->data[6], -1.0));
    assert(FLOAT_EQUAL(result->data[7], 1.0));
    assert(FLOAT_EQUAL(result->data[8], 1.0));


    freeMatrix(matrixA);
    freeMatrix(matrixB);
    freeMatrix(result);
}

void testNormMatrix1() {
    printf("testNormMatrix1\n");

    // 测试向量的一范数
    Matrix *vector = initMatrix(4, 1);
    double array[] = {1.0, -2.0, 3.0, -4.0};
    assignMatrix(vector, array);

    Matrix *norm = normVector1(vector);
    assert(norm != NULL);
    assert(FLOAT_EQUAL(norm->data[0], 10.0)); // 一范数为 各元素绝对值和

    freeMatrix(vector);
    freeMatrix(norm);
}

void testNormMatrix2() {
    printf("testNormMatrix2\n");

    // 测试向量的二范数
    Matrix *vector = initMatrix(4, 1);
    double array[] = {1.0, -2.0, 3.0, -4.0};
    assignMatrix(vector, array);

    Matrix *norm = normVector2(vector);
    assert(norm != NULL);

    assert(FLOAT_EQUAL(norm->data[0], sqrt(30.0))); // 二范数为 各元素平方和开方

    freeMatrix(vector);
    freeMatrix(norm);
}


int main() {
	int num = _MSC_VER; // get the version
	printf( "My MSVC version is: %d\n", num);

	printf("Let's test the matrix operators..\n");

    testinitMatrix();

    testValueMatrix();

    testaddMatrix();

    testsubMatrix();

    testmulMatrix();

	testDotProductMatrix();

	testDeterminant();

    testTransMatrix();

    testTransMatrix2();

	testIsInvertible2x2IsInvertible();

	testIsInvertible3x3IsInvertible();

	testIsInvertiblenxn();

	testInvMatrix2x2();

    testInvMatrix3x3();

    testInvMatrixNonSquare();

	testSafeMatrixInver();

    testDivMatrix2x2();

    testDivMatrix3x3();

	testNormMatrix1();

    testNormMatrix2();

    printf("All tests passed.\n");
	system("pause");
    return 0;
}

