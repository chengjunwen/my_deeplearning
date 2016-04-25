#include <cmath>
#include <cstdlib>

#ifndef _UTILITY_H_
#define _UTILITY_H_

const int maxLayer = 5;
const double expThreshold = 45.0;
const int maxUnit = 6000;
const double L2Reg = 0.0001;

int changeEndian(int x);
int maxElemIndex(double * arr, int size);
void initWeightSigmoid(double *weight, int numIn, int numOut);
void initWeightTanh(double *weight, int numIn, int numOut);
void softmax(double *arr, int size);

double sigmoidc(double x);//考虑超过阈值的sigmoid和tanh ，避免计算出现 nan
double tanhc(double x);

double squareNorm(double *sample, int col, int size);

inline double sigmoid(double x){
	return 1.0 / (1.0 + exp(-x));
}
inline double tanh(double x){
	double a= exp(x) * exp(x);
	return (a - 1.0) / (a + 1.0);
}
inline double relu(double x){
	if(x<0)
		return 0;
	else
		return x;
}
inline double get_sigmoid_derivate(double y){
	return y * (1.0 - y);
}
inline double get_tanh_derivate(double y){
	return (1.0 - y*y);
}
inline double get_relu_derivate(double y){
	if(y==0)
		return 0;
	else
		return 1;
}
//产生 int 或者 double 型随机数
inline int random_int(int low, int high){
	return rand() % (high -low + 1) + low;
}
inline double random_double(double low, double high){
	return ((double)rand() / RAND_MAX) *(high - low) + low;
}
#endif
