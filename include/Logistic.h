#ifndef _LOGISTCU_H_
#define _LOGISTIC_H_

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "cblas_mkl.h"

class Logistic : public SoftmaxLayer, public SuperviseModel {
	public :
		Logistic(int numIn, int numOut);
		Logistic(const char * fileName);
		void setLearningRate(double lr){ Layer::setLearningRate(lr); }
		void setBatchSize(int size){ Layer::setBatchSize(size); }
		
		void trainBatch();
		void runBatch();
		void setInput(double *in){ Layer::setInput(in); }
		void setLabel(double *la){ label = la; }
		
		int getInputNumber(){ return Layer::getInputNumber(); }
		int getOutputNumber(){ return Layer::getOutputNumber(); }
		double * getOutput(){ return Layer::getOutput(); }
		double getTrainingCost();
		void saveModel(FILE *fp);
	private :
		double *label;
		void computeDelta(Layer * prevLayer); //覆盖softmax的computeDelta函数 
};
