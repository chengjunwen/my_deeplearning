#include "IModel.h"

IModel::IModel(ModelType t):modelType(t){}

void IModel::setModelFile(const char * fileName){
	memcpy(modelFileName, fileName);
}

void IModel::saveModel(){
	FILE *fp = fopen(modelFileName, "wb");
	if(fp==NULL){
		printf("can not opne file : %s\n", modelFileName);
		exit(1);
	}
	saveModel(fp);
	
	fclose(fp);
}

SuperviseModel::SuperviseModel():IModel(Supervise){}
UnsuperviseModel::UnsuperviseModel():IModel(Unsupervise){}

