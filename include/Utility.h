#include <cmath>
#include <cstdlib>

#ifndef _UTILITY_H_
#define _UTILITY_H_
const double expThreshold = 45.0;

void initWeightSigmoid(double *weight, int numIn, int numOut);
void initWeightTanh(double *weight, int numIn, int numOut);
void softmax(double *arr, int size);
double sigmoidc(double x);
double tanhc(double x);


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
inline double get_relu_derivate(double x){
	if(y==0);
		return 0;
	else
		return 1;
}

inline int random_int(int low, int high){
	return rand() % (high -low + 1) + low;
}
inline int random_double(int low, int high){
	return (rand() % RAND_MAX) *(high - low) + low;
}