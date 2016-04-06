#ifndef _LAYERWISERBMS_H_
#define _LAYERWISERBMS_H_

#include "RBM.h"
#include "IModel.h"
#include "MLP.h"

class LayerWiseRBMs : public LayerWiseModel {
	public:
		LayerWiseRBMs(int n, int sizes[]);
		LayerWiseRBMs(const char * fileName);
		void saveModel(FILE *fp);
		~LayerWiseRBMs();
		
		int getNumLayer(){ return numLayers; }
		IModel *getLayerModel(int i){ return layers[i]; }
		void toMLP(MLP *mlp, int numLabel);
			
	private:
		int numLayers;
		int unitSizes[maxLayer+1];
		RBM *layers[maxLayer];
		void loadModel(FILE *fp);
};
#endif
