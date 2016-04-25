#ifndef _MLP_H_
#define _MLP_H_

#include <cstdio>
#include <cstdlib>
#include "mkl_cblas.h"
#include "Utility.h"
#include "Layer.h"
#include "IModelComponent.h"
#include "Logistic.h"
/*
 *	class MLP
 */
class MLP :public SuperviseModel {
	public: 
		MLP();
		~MLP();
		MLP(const char * fileName);
		void setLearningRate(double lr);
		void setBatchSize(int size);
		void setInput(double *in);
		void setLabel(double *la);
		void trainBatch();
		void runBatch();

		void setLabelNumber(int n){numLayer = n;} 
		void addLayer(Layer *l){ layers[numLayer++]=l; } //增加层
		void setLayer(Layer *l, int k){ layers[k]=l; }
		Layer * getLayer(int k){ return layers[k]; }
//		void setGaussian(bool f){ guass=b; }
		
		int getLayerNumber(){return numLayer;}
		int getInputNumber();
		int getOutputNumber();
		double *getOutput();
		double *getLabel(){ return label;}
		double getTrainingCost();
		void saveModel(FILE *fp);
	private:
		double learningRate;
		int batchSize;
		double *label;
//		bool gauss;
		int numLayer;
		Layer * layers[maxLayer];	

		void loadModel(FILE *fp);
};

#endif
