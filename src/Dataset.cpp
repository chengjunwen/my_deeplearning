#include "Dataset.h"

Dataset::Dataset(){
	numFeature =0;
	numLabel=0;
	numTrain=0;
	numValid=0;
	trainData=validData=trainLabel=validLabel=NULL;
}

void Dataset::dumpTrainingData(const char * savefile){
	dumpData(savefile, numTrain, trainData, trainLabel);
}

void Dataset::dumpData(const char *savefile, int numData, double*data, double * label){
	FILE *fp=fopen(savefile,"wb");
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
	numTrain = numImage/6;
	numValid = numImage - numTrain;
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
	printf("number of cols: %d\n", magicNum);
	
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


