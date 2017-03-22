#include "Dataset.h"
#include <iostream>
#include <ctring>
#include <cstdlib>
using namespace std;

Dataset::Dataset(){
	numFeature =0;
	numLabel=0;
	numTrain=0;
	numValid=0;
	trainData=validData=trainLabel=validLabel=NULL;
}

void Dataset::dumpTrainData(const char * savefile){
	dumpData(savefile, numTrain, trainData, trainLabel);
}

void Dataset::dumpData(const char *savefile, int numData, double*data, double * label){
	FILE *fp=fopen(savefile,"wb+");
	if(fp == NULL){
		printf("file can not open: %s\n", savefile);
		exit(1);
	}
	fwrite(&numData, sizeof(int), 1, fp);
	fwrite(&numFeature, sizeof(int), 1, fp);
	fwrite(&numLabel, sizeof(int), 1, fp);
	if(data != NULL)
		fwrite(data, sizeof(double), numData*numFeature, fp);
	if(label != NULL)	
		fwrite(label, sizeof(double), numData*numLabel, fp);
	fclose(fp);
}
Dataset::~Dataset(){
	delete[] trainData;
	delete[] trainLabel;
	delete[] validData;
	delete[] validLabel;
}

SubDataset Dataset::getTrainDataset(){
	return SubDataset(numTrain, numFeature, numLabel, trainData, trainLabel);
}

SubDataset Dataset::getValidDataset(){
	return SubDataset(numValid, numFeature, numLabel, validData, validLabel);
}

int BatchIterator::getRealBatchSize(){
	if(cur==(size-1))
		return data->numSample - batchSize*cur;
	else
		return batchSize;
}
bool BatchIterator::isDone(){
	if(cur<size)
		return 0;
	else
		return 1;
}

void MNISTDataset::loadData(const char* DataFileName, const char* LabelFileName){
	int magicNum,numImage, numRow, numCol;
	uint8_t pixel, label;
	FILE *dataFile = fopen(DataFileName, "rb");
	if(dataFile == NULL){
		printf("can not open file : %s\n", DataFileName);
		exit(1);
	}
	printf("load data ...\n");
	fread(&magicNum, sizeof(int),1, dataFile);
	magicNum = changeEndian(magicNum);
	printf("magic number: %d\n", magicNum);
	
	fread(&numImage, sizeof(int),1, dataFile);
	numImage = changeEndian(numImage);
	printf("number of image: %d\n", numImage);
	
	fread(&numRow, sizeof(int),1, dataFile);
	numRow = changeEndian(numRow);
	printf("number of rows: %d\n", numRow);
	
	fread(&numCol, sizeof(int),1, dataFile);
	numCol = changeEndian(numCol);
	printf("number of cols: %d\n", numCol);

	numFeature = numRow*numCol;
	numValid = numImage/6;
	numTrain = numImage - numValid;
	trainData = new double[numTrain*numFeature];
	validData = new double[numValid*numFeature];

	for(int i=0; i<numTrain; ++i){
		for(int j=0; j<numFeature; ++j){
			fread(&pixel, sizeof(uint8_t),1 ,dataFile);
			trainData[numFeature*i+j] = double(pixel)/255.0;
		}
	}
	for(int i=0; i<numValid; ++i){
		for(int j=0; j<numFeature; ++j){
			fread(&pixel, sizeof(uint8_t),1 ,dataFile);
			validData[numFeature*i+j] = double(pixel)/255.0;
		}
	}

	fclose(dataFile);

	FILE* labelFile = fopen(LabelFileName, "rb");
	if(labelFile == NULL){
		printf("can not open file : %s\n", LabelFileName);
		exit(1);
	}
	fread(&magicNum, sizeof(int),1 ,labelFile);
	magicNum = changeEndian(magicNum);
	printf("number of magic: %d\n", magicNum);
	
	fread(&numImage, sizeof(int),1 ,labelFile);
	numImage = changeEndian(numImage);
	printf("number of image: %d\n", numImage);
	
	numLabel=10;
	trainLabel = new double [numTrain*numLabel];
	validLabel = new double [numValid*numLabel];
	memset(trainLabel, 0, numTrain*numLabel*sizeof(double));
	memset(validLabel, 0, numValid*numLabel*sizeof(double));
	
	for(int i=0; i<numTrain; ++i){
		fread(&label, sizeof(uint8_t), 1, labelFile);
		trainLabel[i*numLabel+label]=1.0;
	}
	for(int i=0; i<numValid; ++i){
		fread(&label, sizeof(uint8_t), 1, labelFile);
		validLabel[i*numLabel+label]=1.0;
	}

	fclose(labelFile);

	printf("load done\n");
}

void BinDataset::loadData(const char *DataFileName, const char *LabelFileName){

    uint8_t label;

	FILE *trainDataFd = fopen(DataFileName, "rb");

	printf("loading training data...\n");
	
	fread(&numTrain, sizeof(int), 1, trainDataFd);
	printf("number of training sample : %d\n", numTrain);

	fread(&numValid, sizeof(int), 1, trainDataFd);
	printf("number of validate sample : %d\n", numValid);

	fread(&numFeature, sizeof(int), 1, trainDataFd);
	printf("number of feature : %d\n", numFeature);

	trainData = new double[numTrain*numFeature];
	validData = new double[numValid*numFeature];

	for(int i = 0; i < numTrain; i++)
		for(int j = 0; j < numFeature; j++){
			fread(&trainData[numFeature*i+j], sizeof(double), 1, trainDataFd);
	}
	for(int i = 0; i < numValid; i++)
		for(int j = 0; j < numFeature; j++){
			fread(&validData[numFeature*i+j], sizeof(double), 1, trainDataFd);
	}
	fclose(trainDataFd);

	printf("loading training label...\n");
	
	FILE *trainLabelFd = fopen(LabelFileName, "rb");

	fread(&numLabel, sizeof(int), 1, trainLabelFd);
	printf("number of label : %d\n", numLabel);

	trainLabel = new double[numTrain*numLabel];
	validLabel = new double[numValid*numLabel];
	memset(trainLabel, 0, numTrain*numLabel*sizeof(double));
	memset(validLabel, 0, numValid*numLabel*sizeof(double));

	for(int i = 0; i < numTrain; i++){
		fread(&label, sizeof(uint8_t), 1, trainLabelFd);
		trainLabel[i*numLabel+label] = 1.0;
	}
	for(int i = 0; i < numValid; i++){
		fread(&label, sizeof(uint8_t), 1, trainLabelFd);
		validLabel[i*numLabel+label] = 1.0;
	}
	fclose(trainLabelFd);
	printf("loading ok...\n");
}

void SVMDataset::loadData(const char* trainDataFileName, const char * validDataFileName){

	char line[40000];
	char *saveptr1;
	char *saveptr2; 

	FILE *fp = fopen(trainDataFileName, 'r');
	fscanf(fp, "%d", &numTrain);
	fscanf(fp, "%d", &numFeature);
	fscanf(fp, "%d", &numLabel);
	printf("numTrain: %d, numFeature: %d, numLabel: %d\n", numTrain, numFeature, numLabel);
	
	trainData = new double[numTrain*numFeature];
	trainLabel = new double[numTrain*numLabel];
	
	memset(trainData, 0, numTrain*numFeature*sizeof(double));
	memset(trainLabel, 0, numTrain*numLabel*sizeof(double));

	for(int i=0; i<numTrain; ++i){
		fgets(line, 40000, fp);
		char * token = strtok_r(line, " ", saveptr1);
		int label = atoi(token)-1;
		trainLabel[i*numLabel + label] = 1.0;
		if(saveptr[0]!='\n'){
			while(token=strtok_r(NULL, " ", saveptr1)!=NULL){
				int index;
				double value;
				sscanf(token, "%d:%lf", &index, &value);
				trainData[i*numFeature + index-1] = value;
				
			}
			
		}
	}
	fclose(fp);


	FILE *fp = fopen(validDataFileName, 'r');
	fscanf(fp, "%d", &numValid);
	fscanf(fp, "%d", &numFeature);
	fscanf(fp, "%d", &numLabel);
	printf("numValidate: %d, numFeature: %d, numLabel: %d\n", numValid, numFeature, numLabel);
	
	validData = new double[numValid*numFeature];
	validLabel = new double[numValid*numLabel];
	
	memset(validData, 0, numValidn*numFeature*sizeof(double));
	memset(validLabel, 0, numValid*numLabel*sizeof(double));

	char *saveptr1;
	char *saveptr2; 
	for(int i=0; i<numValid; ++i){
		fgets(line, 40000, fp);
		char * token = strtok_r(line, " ", saveptr1);
		int label = atoi(token)-1;
		validLabel[i*numLabel + label] = 1.0;
		if(saveptr[0]!='\n'){
			while(token=strtok_r(NULL, " ", saveptr1)!=NULL){
				int index;
				double value;
				sscanf(token, "%d:%lf", &index, &value);
				validData[i*numFeature + index-1] = value;
				
			}
			
		}
	}
	fclose(fp);

}

TransmissionDataset::TransmissionDataset(Dataset *data, IModel* model){
	numTrain = data->getTrainNumber();
	numValid = data->getValidNumber();
	numFeature = model->getOutputNumber();
	numLabel = data->getLabelNumber();
	trainData = new double[numTrain*numFeature];
	validData = new double[numValid*numFeature];
	trainLabel = new double[numTrain*numLabel];
	validLabel = new double[numValid*numLabel];
	SubDataset tmpData; 
	int batchSize = 10;

//train data
	tmpData = data->getTrainDataset();
	BatchIterator *iter = new BatchIterator(&tmpData, batchSize);
	for( iter->first(); !iter->isDone(); iter->next()){
		int theBatchSize = iter->getRealBatchSize();
		int maskInd = iter->getCurrentIndex();
		model->setBatchSize(theBatchSize);
		model->setInput(iter->getCurrentDataBatch());
		model->runBatch();
		memcpy(trainData + maskInd*numFeature*batchSize, model->getOutput(), numFeature*theBatchSize*sizeof(double));
	}
	delete iter;

//valid data
    tmpData = data->getValidDataset();
    iter = new BatchIterator(&tmpData, batchSize);
    for( iter->first(); !iter->isDone(); iter->next()){
        int theBatchSize = iter->getRealBatchSize();
        int maskInd = iter->getCurrentIndex();
        model->setBatchSize(theBatchSize);
        model->setInput(iter->getCurrentDataBatch());
        model->runBatch();
        memcpy(validData + maskInd*numFeature*batchSize, model->getOutput(), numFeature*theBatchSize*sizeof(double));
    }
	delete iter;
// label
	memcpy(trainLabel, data->getTrainLabelBatch(0), numTrain*numLabel*sizeof(double));
	memcpy(validLabel, data->getValidLabelBatch(0), numValid*numLabel*sizeof(double));
}

MergeDataset::MergeDataset(Dataset * originDatas[], int numSets){
    numFeature = 0;
    numLabel = originDatas[0]->getLabelNumber();
    numTrain = originDatas[0]->getTrainNumber();
    numValid = originDatas[0]->getValidNumber();
    for(int i=0; i<numSets; i++){
        numFeature += originDatas[i] -> getFeatureNumber();
    }
    trainData = new double[numTrain*numFeature];
    trainLabel = new double[numTrain*numLabel];
    validData = new double[numValid*numFeature];
    validLabel = new double[numValid*numLabel];
    memset(trainData, 0, sizeof(double)*numTrain*numFeature);
    memset(trainLabel, 0, sizeof(double)*numLabel*numTrain);
    memset(validData, 0, sizeof(double)*numValid*numFeature);
    memset(validLabel, 0, sizeof(double)*numLabel*numValid);
    int maskOffset =0;
    for(int i=0; i<numTrain; i++){
        for(int j=0; j<numSets; j++){
            memcpy(trainData+maskOffset,
                    originDatas[j]->getTrainDataBatch(i),
                    originDatas[j]->getFeatureNumber()*sizeof(double));
            maskOffset += originDatas[j]->getFeatureNumber();
        }
    }
    maskOffset =0;
    for(int i=0; i<numValid; i++){
        for(int j=0; j<numSets; j++){
            memcpy(validData+maskOffset,
                    originDatas[j]->getValidDataBatch(i),
                    originDatas[j]->getFeatureNumber()*sizeof(double));
            maskOffset += originDatas[j]->getFeatureNumber();
        }
    }
    memcpy( trainLabel, originDatas[0]->getTrainLabelBatch(0), numTrain*numLabel*sizeof(double) );
    memcpy( validLabel, originDatas[0]->getValidLabelBatch(0), numValid*numLabel*sizeof(double) );
//todo
}
MergeDataset::~MergeDataset(){}


