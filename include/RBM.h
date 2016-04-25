#ifndef _RBM_H_ 
#define _RBM_H_

#include <ctime>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "Dataset.h"
#include "Utility.h"
#include "IModel.h"

class LayerWiseRBMs; 

class RBM : public UnsuperviseModel {
	public:
		RBM(int numIn, int numOut);
		RBM(const char * fileName);
		RBM(FILE *fp);
		~RBM();
		void setLearningRate(double lrt){ lr = lrt; }
		void setBatchSize(int size){ batchSize = size; }
		void setStep(int st){ step = st; }
		void setPersist(bool f){ persist=f; }
		void setGaussVisible( bool f ){ binVis=!f; }
		void setGaussHidden( bool f ){ binHid=!f; }

		void trainBatch();	//训练
		void runBatch();
		void setInput(double *in){ v = in; }

		int getInputNumber(){ return numVis; }
		int getOutputNumber(){ return numHid; }
		double *getOutput(){ return h1; }
		double *getWeight(){ return w; }
		double *getBias(){ return bh; }
		double getTrainingCost();

		void saveModel(FILE *fp);
		friend class LayerWiseRBMs;
	private:

		void init();
		void initWeight();
		void mallocMemory();
		void getProbH(double *v, double *ph);
		void getProbV(double *h ,double *pv);
		void getSampleH(double *ph, double *h);
		void getSampleV(double *pv, double *v);
//		void gibbs_vhv();		//gibbs 采样
		void gibbs_hvh(double *hstart, double *h, double *ph, double *v, double *pv);
		void runChain();

		void updateWeight();
		void updateBias();
		double getReconstructCost();
		double getPseudoCost();
		void getFreeEnergy(double *v, double *fe);

		void loadModel(FILE *fp);

		double *ph1, *ph2, *h1, *h2;
		double *v, *v2, *pv;
		double *w, *bv, *bh;
		double *bI;
		int xi;
		bool persist;
		double *chainStart;

		int numVis, numHid;
		bool binVis, binHid;
		double lr;
		int batchSize;
		int step;

		// Activition maximization
		void getAMDelta(int idx, double * lastAMDelta);

		double * AMDelta;
};

#endif
