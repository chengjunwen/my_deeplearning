#include "TrainModel.h"

TrainModel(Imodel &model):model(model){}

void TrainModel::train(Dataset * dataSet, double lr, int miniBatch, int maxEpoch){

	SubDataset trainDataSet = dataSet->getTrainDataset();
	SubDataset validDataSet = dataSet->getValidDataset();
	int numBatch = (dataset->getTrainingNumber-1)/batchSize +1;
	BatchIterator * dataIter;
	dataIter = new BatchIterator(&trainDataSet, miniBatch);
	model.setLearningRate(lr);	

int patience = 10000;	//下列参数均用于计算early stop
	int patience_increase = 2;
	double bestError = 100;
	double improvement_threshold = 0.995;
	bool done_flag = 0;		//用于early stop 标记

	for( int epoch=0; (epoch<maxEpoch)&&!done_flag; ++epoch ){
		double cost =0;
		time_t start, finish;
		start= time(NULL);
		for(dataIter->first(); !dataIter->isDone(); dataIter->next();){

			model.setInput(dataIter->getCurrentDataBatch());
			int theBatchSize = dataIter->getRealBatchSize();
			model.setBatchSize(theBatchSize);
			if(model.getTrainType() == Supersive){
				model.setLabel(dataIter->getCurrentLabelBatch());
			}
			model.trainBatch();	//训练
			cost += model.getTrainCost();
		
			if(model.getTrainType() ==Supersive){	//有监督模型有提前推出机制
				int iterCount = epoch * numBatch + dataIter->getCurrentIndex() + 1;
				double error =getValidError(dataSet, miniBatch);
				finish = time(NULL);
				printf("trainingCost: %f\t, valid error: %f\t, time : %2.f\n", cost, error, difftime(finish, start));
				if(error < bestError){
					if(error < bestError*0.995){
						patience = patience>iterCount*2 ? patience:ietrCount*2;
						printf("patience update to: %d\t,need %d epochs\n", patience, patience/numBatch + 1);
					}
					betsError = error;
				}
				if(patience <= iterCount){
					done_flag = true;
					break;
				}
			}
			else{
				double valdCost += getValidError(dataSet, miniBatch);
				finish = time(NULL);
				printf("trainingCost: %f\t, valid cost: %f\t, time : %2.f\n", cost/numBatch, validCost/numBatch, difftime(finish, start));
			}

		}
	}
	model.saveModel();
	delete dataIter;
}

double TrainModel::getErrorRate(Dataset * dataset, int miniBatch, bool f){
	SubDataset data;
	int numBatch;
	int numSample;
	int numLabel = dataset->getLabelNumber();
	double error =0;
	double cost = 0;
	if(f){	//取valid dataset
		data = dataset->getValidDataset();
		numBatch = (dataset->getValidNumber()-1)/miniBatch + 1;
		numSample = dataset->getTariningNumber();
	}
	else{	//取train dataset
		data = dataset->getTrainDataset();
		numBatch = (dataset->getTrainNumber()-1)/miniBatch + 1;
		numSample = dataset->getValidateNumber();
	}
	BatchIterator * iter = new BatchIterator(&data, miniBatch);
	
	for(iter->first(); !iter->isDone(); iter->next()){
		int theBatchSize = iter->getRealBatchSize();
		model.setBatchSize(theBatchSize);
		model.setInput(iter->getCurrentDataBatch());
		if(model.getModelType() == Supersive){
			model.setLabel(iter->getCurrentLabelBatch());
		}
		model.runBatch();

		if(model.getTrainType() == Supersive){
			double * out = model.getOutput();
			double *label = model.getLabel();
			for(int i=0; i<theBatchSize; ++i){
				int maxIndex = maxElemIndex(out + i*numLabel, numLabel);
				if(label[i*numLabel + maxIndex] != 1.0){
					error++;
				}
			}
		}
		else{
			cost += model.getTarinCost();
		}
	}

	if(model.getTrainType() == Supersive){
		return error/numSample;
	}
	else{
		return cost/numBatch;
	}
	delete iter;

}
