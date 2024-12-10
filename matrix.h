/**********************************************************
** FreeMatrix-C: An open source C/C++ matrix and vector library for Safe and High Performance Computations(support version C99+)
** Matrix_hub ��һ����Gao��Xing������C���Ծ�������⣬���ṩ�˰�ȫ����Ч�ľ����������������㷨�������������ɡ��������㡢�����㡢����ʽ���������㡢�ȡ�����ֵ�ֽ�ȡ�
** ����Ŀ��Ŀ���Ǽ�C���Ի����µľ������㣬�﷨֧������C���԰汾������Microsoft Visual Studio 2010�����������ڿ��кͽ�ѧʹ�á�
** ����Ŀ��ѭApache-3.0���֤������GitHub��ַ���ҵ�����Դ���������������

** Last revised date: 2024-12-9
** Author: Gao��Xing
** Email: xinggao163@163.com

** ����ƽ̨��
** ����ϵͳ��Windwos 11 
** ��������Microsoft Visual Studio 2010�� _MSC_VER >= 1600
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

// һάָ���

// ��ȫ�ͷ��ڴ�ĺ�
#define SAFE_FREE(ptr) do { if (ptr) { free(ptr); ptr = NULL; } } while (0)
// �Զ��徫�ȵ�������С�Ƚ� ����ȡ�� == �����
#define FLOAT_EQUAL(a, b) (fabs((a) - (b)) < 1e-6)

// ��ž�������ݽṹ
typedef struct 
{
	int rows,columns;		//rowsΪ�У�columnsΪ��,
	double *data; //�������Ч��
}Matrix;

//����Ļ�������
Matrix* initMatrix(int rows,int columns);		//��ʼ������
void assignMatrix(Matrix *matrix,double *array);				//��һ������ֵ
int capacityMatrix(Matrix *matrix);								//���һ�����������
void freeMatrix(Matrix *matrix);							//�ͷ�һ������
void copyMatrix(Matrix *matrix_A, Matrix *matrix_B);		//����һ�������ֵ A to B
Matrix* indexVector(Matrix *vector_A, int left, int right);		//��������A[left:right] [left-1, right)
Matrix* indexMatrix(Matrix *matrix_A, int left, int right);		//��������A[left:right] [left-1, right)
void printMatrix(Matrix *matrix);							//��ӡһ������

//����Ļ�������
Matrix* diagMatrix(double* diag_elements, int n);  //�����Խ���
Matrix* addMatrix(Matrix *matrix_A,Matrix *matrix_B);		//����ļӷ�
Matrix* subMatrix(Matrix *matrix_A,Matrix *matrix_B);		//����ļ���
Matrix* mulMatrix(Matrix *matrix_A,Matrix *matrix_B);		//����ĳ˷�
Matrix* rightDivMatrix(Matrix *matrix_A,Matrix *matrix_B);		//����ĳ��� 2-3�׾��� �ҳ� /��A/B��ʾ����A���Ծ���B����
Matrix* leftDivMatrix(Matrix *matrix_A,Matrix *matrix_B);		//����ĳ��� 2-3�׾��� ��� \��A\B��ʾ����A������Ծ���B
Matrix* rightDivMatrix_n(Matrix *matrix_A,Matrix *matrix_B);		//����ĳ��� n�׾��� �ҳ� /��A/B��ʾ����A���Ծ���B����
Matrix* leftDivMatrix_n(Matrix *matrix_A,Matrix *matrix_B);		//����ĳ��� n�׾��� ��� \��A\B��ʾ����A������Ծ���B

Matrix* dotProductMatrix(Matrix *matrix_A,Matrix *matrix_B); //����ĵ��
Matrix* crossProduct(Matrix *vector_A, Matrix *vector_B) ;// ������˺���
Matrix* crossMatrix(Matrix *matrix_A,Matrix *matrix_B); //����Ĳ��

Matrix* getsubMatrix(Matrix *matrix, int row, int col); // ��ȡ�Ӿ���ȥ��ָ����row����col	��
double detMatrix(Matrix *matrix); //��������ʽ n��

void transMatrix(Matrix *matrix);			//ת�� ����Ϊ����
Matrix* transMatrix2(Matrix *matrix);  //ת�� �Ƿ���

bool isInvertible(Matrix *matrix);// 2-3�׾����Ƿ����
bool isInvertiblenx_n(Matrix *matrix);// // �ж�n�׾����Ƿ����ĺ���

Matrix* invMatrix(Matrix *matrix); //2-3�׾������
Matrix* safeMatrixInv(Matrix *matrix, int n); //n�װ�ȫ����

// �������� rows or columnsΪ1
Matrix* normVector1(Matrix *vector);		//������һ����
Matrix* normVector2(Matrix *vector);		//�����Ķ�����

// ���ȱ任�� ����ת���� ˳ʱ�룩
Matrix* rotationMatrix_x (double element);
Matrix* rotationMatrix_y (double element);
Matrix* rotationMatrix_z (double element);

#endif // MATRIX_H