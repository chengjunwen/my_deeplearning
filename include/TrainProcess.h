#ifndef _TRAINPROCESS_H_
#define _TRAINPROCESS_H_

#include <cstdio>
#include <cstdlib>
#include "IModel.h"
/*
 * 抽象化训练过程， 传入所需训练的模型和训练数据即可
 */
class TrainProcess	{
	public :
		TrainProcess(IModel &model);
		void train(Dataset * dataSet, double lr, int miniBatch, int maxEpoch);
		void getErrorRate(Dataset * dataset, int miniBatch, bool f=0);
		~TrainProcess(){}
	provite:
		IModel &model;

};
