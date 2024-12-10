/**********************************************************
** FreeMatrix-C: An open source C/C++ matrix and vector library for Safe and High Performance Computations(support version C99+)
** Matrix_hub 是一个由Gao，Xing开发的C语言矩阵运算库，它提供了安全、高效的矩阵操作函数与矩阵算法，包括矩阵生成、基础运算、逆运算、行列式、范数运算、秩、奇异值分解等。
** 该项目的目标是简化C语言环境下的矩阵运算，语法支持早期C语言版本及早期Microsoft Visual Studio 2010编译器，便于科研和教学使用。
** 该项目遵循Apache-3.0许可证，可在GitHub地址上找到完整源代码与测试用例。

** Last revised date: 2024-12-9
** Author: Gao，Xing
** Email: xinggao163@163.com

** 测试平台：
** 操作系统：Windwos 11 
** 编译器：Microsoft Visual Studio 2010， _MSC_VER >= 1600
**********************************************************/
/**********************************************************
** Refs 
** [1] https://github.com/PX4/PX4-Matrix
** [2] https://github.com/EricLengyel/Terathon-Math-Library
** [3] https://gitcode.com/gh_mirrors/ma/Matrix_hub
** [4] https://learn.microsoft.com/cpp/overview/compiler-versions
**********************************************************/
#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 一维指针版

// 安全释放内存的宏
#define SAFE_FREE(ptr) do { if (ptr) { free(ptr); ptr = NULL; } } while (0)
// 自定义精度的两数大小比较 用于取代 == 运算符
#define FLOAT_EQUAL(a, b) (fabs((a) - (b)) < 1e-6)

// 存放矩阵的数据结构
typedef struct 
{
	int rows,columns;		//rows为行，columns为列,
	double *data; //提高索引效率
}Matrix;

//矩阵的基本操作
Matrix* initMatrix(int rows,int columns);		//初始化矩阵
void assignMatrix(Matrix *matrix,double *array);				//给一个矩阵赋值
int capacityMatrix(Matrix *matrix);								//获得一个矩阵的容量
void freeMatrix(Matrix *matrix);							//释放一个矩阵
void copyMatrix(Matrix *matrix_A, Matrix *matrix_B);		//复制一个矩阵的值 A to B
Matrix* indexVector(Matrix *vector_A, int left, int right);		//索引向量A[left:right] [left-1, right)
Matrix* indexMatrix(Matrix *matrix_A, int left, int right);		//索引矩阵A[left:right] [left-1, right)
void printMatrix(Matrix *matrix);							//打印一个矩阵

//矩阵的基本运算
Matrix* diagMatrix(double* diag_elements, int n);  //创建对角阵
Matrix* addMatrix(Matrix *matrix_A,Matrix *matrix_B);		//矩阵的加法
Matrix* subMatrix(Matrix *matrix_A,Matrix *matrix_B);		//矩阵的减法
Matrix* mulMatrix(Matrix *matrix_A,Matrix *matrix_B);		//矩阵的乘法
Matrix* rightDivMatrix(Matrix *matrix_A,Matrix *matrix_B);		//矩阵的除法 2-3阶矩阵 右除 /：A/B表示矩阵A乘以矩阵B的逆
Matrix* leftDivMatrix(Matrix *matrix_A,Matrix *matrix_B);		//矩阵的除法 2-3阶矩阵 左除 \：A\B表示矩阵A的逆乘以矩阵B
Matrix* rightDivMatrix_n(Matrix *matrix_A,Matrix *matrix_B);		//矩阵的除法 n阶矩阵 右除 /：A/B表示矩阵A乘以矩阵B的逆
Matrix* leftDivMatrix_n(Matrix *matrix_A,Matrix *matrix_B);		//矩阵的除法 n阶矩阵 左除 \：A\B表示矩阵A的逆乘以矩阵B

Matrix* dotProductMatrix(Matrix *matrix_A,Matrix *matrix_B); //矩阵的点积
Matrix* crossProduct(Matrix *vector_A, Matrix *vector_B) ;// 向量叉乘函数
Matrix* crossMatrix(Matrix *matrix_A,Matrix *matrix_B); //矩阵的叉乘

Matrix* getsubMatrix(Matrix *matrix, int row, int col); // 获取子矩阵（去掉指定行row和列col	）
double detMatrix(Matrix *matrix); //矩阵行列式 n阶

void transMatrix(Matrix *matrix);			//转置 条件为方阵
Matrix* transMatrix2(Matrix *matrix);  //转置 非方阵

bool isInvertible(Matrix *matrix);// 2-3阶矩阵是否可逆
bool isInvertiblenx_n(Matrix *matrix);// // 判断n阶矩阵是否可逆的函数

Matrix* invMatrix(Matrix *matrix); //2-3阶矩阵的逆
Matrix* safeMatrixInv(Matrix *matrix, int n); //n阶安全求逆

// 向量范数 rows or columns为1
Matrix* normVector1(Matrix *vector);		//向量的一范数
Matrix* normVector2(Matrix *vector);		//向量的二范数

// 初等变换阵 （旋转矩阵 顺时针）
Matrix* rotationMatrix_x (double element);
Matrix* rotationMatrix_y (double element);
Matrix* rotationMatrix_z (double element);

#endif // MATRIX_H