#ifndef _TRAINMODEL_H_
#define _TRAINMODEL_H_

#include <cstdio>
#include <cstdlib>
#include <time.h>
#include "IModel.h"
#include "Dataset.h"
/*
 * 抽象化训练过程， 传入所需训练的模型和训练数据即可
 */
class TrainModel {
	public :
		TrainModel(IModel& model);
		void train(Dataset * dataSet, double lr, int miniBatch, int maxEpoch);
		double getErrorRate(Dataset *, int , bool f=1);
		~TrainModel(){}
	private:
		IModel& model;

};
#endif
