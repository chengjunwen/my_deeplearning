#include "Utility.h"


void sofamax(double *arr, int size){
	double sum=0;
	for(int i=0; i<size; ++i){
		sum += exp(arr[i])
	}
	for(int i=0; i<size; ++i){
		arr[i] = arr[i] / sum;
	}
}

double sigmoidc(double x){
	if(x > expThreshold){
		return 1;
	}
	else if(x < -1*expThreshold){
		return 0;
	}
	else{
		return 1.0 / (1.0 + exp(-x));
	}
}

double tanhc(double x){
	if(x > expThreshold)
		return 1;
	else if(x < -expThreshold)
		return -1;
	else{
    	double a= exp(x) * exp(x);                                             
	    return (a - 1.0) / (a + 1.0);
	}
}

void initWeightSigmoid(double *weight, int numIn, int numOut){
	double low,high;
	high = 4 * sqrt(6 / (numIn + numOut));
	low = -1 * high;
	for(int i=0; i<numIn * numOut; ++i){
		weight[i] = random_double(low, high);
	}
}

void initWeightTanh(double *weight, int numIn, int numOut){
	double low,high;
	high = sqrt(6 / (numIn + numOut));
	low = -1 * high;
	for(int i=0; i<numIn * numOut; ++i){
		weight[i] = random_double(low, high);
	}
}
