#include "MultiModalComponent.h"

/*
* MultiModalModel 
*
*/
MultiModalModel::MultiModalModel(int n):numModel(n){}

void MultiModalModel::setModelFile(const string modelFileNames[]){
    for(int i=0; i<numModel; i++){
        multiModels[i]->setModelFile(modelFileNames[i].c_str());
    }
}

void MultiModalModel::setGaussVisible(bool gauss[]){
    for(int i=0; i<numModel; i++){
        multiModels[i]->setGaussVisible(0, gauss[i]);
    }
}


/*
 * MultiModal RBMs
 *
 */
MultiModalRBMs::MultiModalRBMs():MultiModalModel(0){}
MultiModalRBMs::MultiModalRBMs(int numModel, const string modelFileNames[]):MultiModalModel(numModel){
	for(int i=0; i<numModel; i++){
		multiModels[i] = new LayerWiseRBMs(modelFileNames[i].c_str());
	}
}
MultiModalRBMs::~MultiModalRBMs(){
	for(int i=0; i<numModel; i++){
		delete multiModels[i];
	}
}

void MultiModalRBMs::trainModel(Dataset *dataset, double lr[], int batchSize, int epoch[]){
	for(int i =0; i <numModel; i++){

		LayerWiseTrainModel preTrainModel(multiModels[i]);
        printf("pretain model: %d\n",i+1);
        preTrainModel.train(&dataset[i], lr[i], batchSize, epoch[i]);
	}
}

Dataset * MultiModalRBMs::runModel(Dataset *dataset){
	
	Dataset * hiddenDataset[numModel];
	for(int i=0; i<numModel; i++){
		int nLayers = multiModels[i]->getLayerNumber();
		Dataset *curData = &dataset[i];
        for(int j=0; j<nLayers; j++){
        	Dataset * tmData = new TransmissionDataset(curData, multiModels[i]->getLayer(j) );

            if(curData!=&dataset[i]){
            	delete curData;
            }
            curData = tmData;
		}
		hiddenDataset[i] = curData;
	}

	Dataset *tmp = new MergeDataset(hiddenDataset, numModel);

	for(int i=0; i<numModel; i++){
		delete hiddenDataset[i];
	}
	return tmp; 
}

/*
 * Spiral MultiModal 
 *
 */

SpiralMultiModal::SpiralMultiModal():MultiModalModel(0){}
SpiralMultiModal::SpiralMultiModal(int numModel, const string modelFileNames[]):MultiModalModel(numModel){
    for(int i=0; i<numModel; i++){
        multiModels[i] = new LayerWiseRBMs(modelFileNames[i].c_str());
    }
}
SpiralMultiModal::~SpiralMultiModal(){
	for(int i=0; i<numModel; i++){
		delete multiModels[i];
		delete hiddenDataset[i];
	}
}

void SpiralMultiModal::addModel(int numLayer, int unitSize[] ){
	multiModels[numModel] = new LayerWiseRBMs(numLayer, unitSize);
	multiModels[numModel]->setPersistent(false);
	numModel++;
}

//set dataset used for train one model, set the bottom model data
void SpiralMultiModal::setDataset(Dataset *dataset, int n){
	for(int i=0; i<n; i++){
		datas[i] = &dataset[i];
	}
}
/*
 * 训练整体模型
 *
 */
void SpiralMultiModal::trainModel(Dataset *dataset, double lr, int batchSize, int epoch){
	double lrs[5] = {lr, lr, lr, lr, lr};
	int epochs[5] = {epoch, epoch, epoch, epoch, epoch};

	trainModel(dataset, lrs, batchSize, epochs);

}
void SpiralMultiModal::trainModel(Dataset *dataset, double lr[], int batchSize, int epoch[]){
	Dataset *curData = &dataset[0];
	Dataset *tmpData[2];

	for(int i=0; i<numModel; i++){
		printf("pretrain model: %d\n", i+1);
		printf("input size: %d\n", curData->getFeatureNumber());
		trainOneModel(i, curData, lr[i], batchSize, epoch[i]);

		tmpData[0] = runOneModel(i, curData);
		if(i != (numModel-1) ){
			tmpData[1] = &dataset[i+1];
			hiddenDataset[i] = new MergeDataset(tmpData, 2);
			delete tmpData[0];
		}
		else{
			hiddenDataset[i] = tmpData[0];
		}
			
		curData = hiddenDataset[i];
	}
}

Dataset * SpiralMultiModal::runModel(Dataset *dataset){
    Dataset *curData = &dataset[0];
    Dataset *tmpData[2];
    
	for(int i=0; i<numModel; i++){
        tmpData[0] = runOneModel(i, curData);
        if(i != (numModel-1) ){
            tmpData[1] = &dataset[i+1];
            hiddenDataset[i] = new MergeDataset(tmpData, 2);
            delete tmpData[0];
        }
        else{
            hiddenDataset[i] = tmpData[0];
        }
            
        curData = hiddenDataset[i];
    }
	return curData;
}

void SpiralMultiModal::trainOneModel(int k, Dataset *dataset, double lr, int batchSize, int epoch){
	LayerWiseTrainModel preTrainModel(multiModels[k]);
    preTrainModel.train(dataset, lr, batchSize, epoch);
}

Dataset * SpiralMultiModal::runOneModel(int k, Dataset *dataset){
	Dataset *curData = dataset;
    int nLayers = multiModels[k]->getLayerNumber();
	for(int j=0; j<nLayers; j++){
    	Dataset * tmpData = new TransmissionDataset(curData, multiModels[k]->getLayer(j) );

        if(curData!=dataset){
        	delete curData;
        }
        curData = tmpData;
	}
	return curData;
}
