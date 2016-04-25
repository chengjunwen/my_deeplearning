#include "Utility.h"


void softmax(double *arr, int size){
	double sum=0;
	for(int i=0; i<size; ++i){
		sum += exp(arr[i]);
	}
	for(int i=0; i<size; ++i){
		arr[i] = exp(arr[i]) / sum;
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
	high = 4 * sqrt(6. / (numIn + numOut));
	low = -1 * high;
	for(int i=0; i<numIn * numOut; ++i){
		weight[i] = random_double(low, high);
	}
}

void initWeightTanh(double *weight, int numIn, int numOut){
	double low,high;
	high = sqrt(6. / (numIn + numOut));
	low = -1 * high;
	for(int i=0; i<numIn * numOut; ++i){
		weight[i] = random_double(low, high);
	}
}

int changeEndian(int x){
	int nbyte = sizeof(int);
    char *p = (char*) &x;
    char tmp;
    for(int i = 0, j = nbyte - 1; i < j; i++, j--){
        tmp = p[i]; p[i] = p[j]; p[j] = tmp;
    }
    return x;

	
}

int maxElemIndex(double *arr, int size){
	int maxIndex = 0;
	for(int i=1; i<size; ++i){
		if(arr[i] > arr[maxIndex])
			maxIndex=i;
	}
	return maxIndex;
}

double squareNorm(double *sample, int col, int size){
	double sum;
	for(int i=0; i<size; i++){
		double sq =0;
		for(int j=0; j<col; j++){
			sq += sample[i*col+col] * sample[i*col+col];
		}
		sum += sqrt(sq);
	}
	return sum/size;
}
