#ifndef _ENCODERLAYER_H_
#define _ENCODERLAYER_H_

#include <ctime>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "Dataset.h"
#include "Utility.h"
#include "IModelComponent.h"

class DeepAutoEncoder;
class AutoEncoder;

class EncoderLayer {
	public:
		EncoderLayer(int numIn, int numOut);
		EncoderLayer(const char * fileName);
		EncoderLayer(FILE *fp);
		EncoderLayer(int numIn, int numOut, double *w, double *b, double *c);
		virtual ~EncoderLayer();
		void setLearningRate(double lrt){ lr = lrt; }
		void setBatchSize(int size){ batchSize = size; }
		void setInput(double *in){ x = in; }
		void setGaussIn(bool f){ binIn = !f; }
		void setGaussOut(bool f){ binOut = !f; }

		int getInputNumber(){ return numIn; }
		int getOutputNumber(){ return numOut; }
		double *getOutput(){ return y; }
		void saveModel(FILE *fp);

		friend class DeepAutoEncoder;
		friend class AutoEncoder;
	private:
		void init();
		void initWeight();
		void mallocMemory();
		void getHFromX(double *x, double *h);
		void getYFromH(double *h, double *y);
		void getDeltaH(EncoderLayer * prevLayer);
		void getDeltaY(EncoderLayer * prevLayer);
		
		void updateWeight(double *prevH);
		void loadModel(FILE *fp);
		
		double *x, *y, *h, *dh, *dy;
		double *w, *b, *c ,*bI;
		bool binIn, binOut;

		int numIn, numOut;
		double lr;
		int batchSize;

//Activition maximization
        double *AMdelta;
		void getAMDelta(int unitIdx, double *lastAMdelta);
};

#endif
