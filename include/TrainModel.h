#ifndef _TRAINMODEL_H_
#define _TRAINMODEL_H_

#include <cstdio>
#include <cstdlib>
#include <time.h>
#include "IModel.h"
#include "Dataset.h"
/*
 * 抽象化训练过程， 传入所需训练的模型和训练数据即可,TrainModel class用于是所有layer是整体的模型
 */
class TrainModel {
	public :
		TrainModel(IModel * model);
		void train(Dataset * dataSet, double lr, int miniBatch, int maxEpoch);
		double getErrorRate(Dataset *, int , bool f=1); //计算validation 或者train 数据集的错误率
		~TrainModel(){}
	private:
		IModel* model;

};
/*
 * 该类用于layers间是相互独立的模型，例如 stackRBMs, layer间的学习率和batchsize可以不同
 */

class LayerWiseTrainModel {
public:	
	LayerWiseTrainModel(LayerWiseModel * model);
	void train(Dataset * dataSet, double lr, int miniBatch, int maxEpoch);
	void train(Dataset * dataSet, double lrs[], int miniBatch, int maxEpoch);
	void train(Dataset * dataSet, double lr, int miniBatch, int maxEpochs[]);
	void train(Dataset * dataSet, double lrs[], int miniBatch, int maxEpochs[]);
	~LayerWiseTrainModel(){}
private:
	LayerWiseModel* model;	
	
};
#endif

