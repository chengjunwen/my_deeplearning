#include "TrainProcess.h"

TrainProcess(Imodel &model):model(model){}

void TrainProcess::train(Dataset * dataSet, double lr, int miniBatch, int maxEpoch){

	SubDataset trainDataSet = dataSet->getTrainDataset();
	SubDataset validDataSet = dataSet->getValidDataset();
	int numBatch = dataset->getTrainingNumber;
	BatchIterator * dataIter;
	dataIter = new BatchIterator(&trainDataSet, miniBatch);
	
	int patience = 10000;	//下列参数均用于计算early stop
	int patience_increase = 2;
	double bestError = 100;
	double improvement_threshold = 0.995;
	bool stop_flag = 0;		//用于early stop 标记

	for( int epoch=0; (epoch<maxEpoch)&&!stop_flag; ++epoch ){
		double error =0; 
		double cost =0;
		time_t start, finish;
		start= time(NULL);
		for(dataIter->first(); !dataIter->isDone(); dataIter->next();){
			
		}
	}
}
