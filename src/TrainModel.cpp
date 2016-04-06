#include "TrainModel.h"

TrainModel::TrainModel(IModel * model):model(model){}

void TrainModel::train(Dataset * dataSet, double lr, int miniBatch, int maxEpoch){

	SubDataset trainDataSet = dataSet->getTrainDataset();
	SubDataset validDataSet = dataSet->getValidDataset();
	int numBatch = (dataSet->getTrainNumber()-1)/miniBatch +1;
	BatchIterator * dataIter;
	dataIter = new BatchIterator(&trainDataSet, miniBatch);
	model->setLearningRate(lr);	

	int patience = 10000;	//下列参数均用于计算early stop
	int patience_increase = 2;
	double bestError = 100;
	double improvement_threshold = 0.995;
	bool done_flag = 0;		//用于early stop 标记

	for( int epoch=0; (epoch<maxEpoch)&&!done_flag; ++epoch ){
		double cost =0;
		time_t start, finish;
		start= time(NULL);
		for(dataIter->first(); !dataIter->isDone(); dataIter->next()){

			model->setInput(dataIter->getCurrentDataBatch());
			int theBatchSize = dataIter->getRealBatchSize();
			model->setBatchSize(theBatchSize);
			if(model->getModelType() == Supervise){
				model->setLabel(dataIter->getCurrentLabelBatch());
			}
			model->trainBatch();	//训练
			cost += model->getTrainingCost();
		}	
		if(model->getModelType() ==Supervise){	//有监督模型有提前推出机制
			int iterCount = epoch * numBatch + dataIter->getCurrentIndex() + 1;
			double error =getErrorRate(dataSet, miniBatch, 1);
			finish = time(NULL);
			printf("epoch: %d trainingCost: %f valid error: %.4lf%% time : %.2fs\n", epoch, cost/numBatch, error*100, difftime(finish, start));
			if(error < bestError){
			if(error < bestError*improvement_threshold){
				patience = patience>iterCount*2 ? patience:iterCount*patience_increase;
					printf("patience update to: %d,need %d epochs\n", patience, patience/numBatch + 1);
				}
				bestError = error;
			}
			if(patience <= iterCount){
				done_flag = true;
				break;
			}
		}
		else{
			double validCost = getErrorRate(dataSet, miniBatch);
			finish = time(NULL);
			printf("epoch: %d trainingCost: %f valid cost: %f time : %2.f\n", epoch, cost/numBatch, validCost, difftime(finish, start));
		}

	}
	model->saveModel();
	delete dataIter;
}

double TrainModel::getErrorRate(Dataset * dataset, int miniBatch, bool f){
	SubDataset data;
	int numBatch;
	int numSample;
	int numLabel = dataset->getLabelNumber();
	double error =0;
	double cost = 0;
	double xx;
	if(f){	//取valid dataset
		data = dataset->getValidDataset();
		numBatch = (dataset->getValidNumber()-1)/miniBatch + 1;
		numSample = dataset->getValidNumber();
	}
	else{	//取train dataset
		data = dataset->getTrainDataset();
		numBatch = (dataset->getTrainNumber()-1)/miniBatch + 1;
		numSample = dataset->getTrainNumber();
	}
	BatchIterator * iter = new BatchIterator(&data, miniBatch);
	
	for(iter->first(); !iter->isDone(); iter->next()){
		int theBatchSize = iter->getRealBatchSize();
		model->setBatchSize(theBatchSize);
		model->setInput(iter->getCurrentDataBatch());
		if(model->getModelType() == Supervise){
			model->setLabel(iter->getCurrentLabelBatch());
		}
		model->runBatch();

		if(model->getModelType() == Supervise){
			double * out = model->getOutput();
			double *label = model->getLabel();
			for(int i=0; i<theBatchSize; ++i){
				int maxIndex = maxElemIndex(out + i*numLabel, numLabel);
				if(label[i*numLabel + maxIndex] != 1.0){
					error++;
				}
			}
		}
		else{
			cost += model->getTrainingCost();
		}
	}

	if(model->getModelType() == Supervise){
		xx = error/numSample;
	}
	else{
		xx = cost/numBatch;
	}
	delete iter;
	return xx;
}

//layers独立时的训练模型
LayerWiseTrainModel::LayerWiseTrainModel(LayerWiseModel * model):model(model){}

void LayerWiseTrainModel::train(Dataset * dataset, double lr, int miniBatch, int maxEpoch){
	int numLayer = model->getNumLayer();
	int * maxEpochs = new int[numLayer];
	double *lrs = new double[numLayer];
	for(int i=0; i<numLayer; i++){
		maxEpochs[i] = maxEpoch;
		lrs[i] = lr;
	}
	train(dataset, lrs, miniBatch, maxEpochs);
	delete[] lrs;
	delete[] maxEpochs;
}

void LayerWiseTrainModel::train(Dataset * dataset, double lrs[], int miniBatch, int maxEpoch){
	int numLayer = model->getNumLayer();
	int *maxEpochs = new int[numLayer];
	for(int i=0; i<numLayer; i++){
		maxEpochs[i] = maxEpoch;
	}
	train(dataset, lrs, miniBatch, maxEpochs);
	delete[] maxEpochs;
}

void LayerWiseTrainModel::train(Dataset * dataset, double lr, int miniBatch, int maxEpochs[]){    int numLayer = model->getNumLayer();
    double * lrs = new double[numLayer];
    for(int i=0; i<numLayer; i++){
        lrs[i] = lr;
    }
    train(dataset, lrs, miniBatch, maxEpochs);
    delete[] lrs;
}

void LayerWiseTrainModel::train(Dataset * dataset, double lrs[], int miniBatch, int maxEpochs[]){    
	int numLayer = model->getNumLayer();
	Dataset *curData = dataset;
    for(int i=0; i<numLayer; i++){
	printf("pretrain layer %d\n", i);
		TrainModel trainModel(model->getLayerModel(i));
		if(i != 0){
			Dataset * tmpData = new TransmissionDataset(curData, model->getLayerModel(i-1));
			if(curData!=dataset)
				delete curData;
			curData = tmpData;
		}
		trainModel.train(curData, lrs[i], miniBatch, maxEpochs[i]);
    }
	if(curData!=dataset){
		delete curData;
	}
	model->saveModel();
}


