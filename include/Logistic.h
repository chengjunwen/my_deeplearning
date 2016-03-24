#ifndef _LOGISTIC_H_
#define _LOGISTIC_H_

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "mkl_cblas.h"
#include "Utility.h"
#include "Layer.h"
#include "IModel.h"

class Logistic : public SoftmaxLayer, public IModel {
	public :
		Logistic(int numIn, int numOut);
		Logistic(const char * fileName);
		Logistic(FILE * fd);
		void setLearningRate(double lr){ Layer::setLearningRate(lr); }
		void setBatchSize(int size){ Layer::setBatchSize(size); }
		
		void trainBatch();
		void runBatch();
		void setInput(double *in){ Layer::setInput(in); }
		void setLabel(double *la){ label = la; }
		
		int getInputNumber(){ return Layer::getInputNumber(); }
		int getOutputNumber(){ return Layer::getOutputNumber(); }
		double * getOutput(){ return Layer::getOutput(); }
		double * getLabel(){ return label;}
		double getTrainingCost();
		void saveModel(FILE *fp);
	private :
		double *label;
		void computeDelta(Layer * prevLayer); //覆盖softmax的computeDelta函数 
};
#endif
