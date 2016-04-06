#ifndef _LAYERWISERBMS_H_
#define _LAYERWISERBMS_H_

#include "RBM.h"
#include "IModel.h"
#include "MLP.h"

/*
 * stackRBMs, 也即是DBN的与训练部分
 */
class LayerWiseRBMs : public LayerWiseModel {
	public:
		LayerWiseRBMs(int n, int sizes[]);
		LayerWiseRBMs(const char * fileName);
		void saveModel(FILE *fp);
		~LayerWiseRBMs();
		
		int getNumLayer(){ return numLayers; }
		IModel *getLayerModel(int i){ return layers[i]; }
		void toMLP(MLP *mlp, int numLabel);	//预训练结束之后,权重值赋给MLP
			
	private:
		int numLayers;	    				//模型层数
		int unitSizes[maxLayer+1];			//每层节点单元数
		RBM *layers[maxLayer];
		void loadModel(FILE *fp);
};
#endif
