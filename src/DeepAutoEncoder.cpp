#include "DeepAutoEncoder.h"

DeepAutoEncoder::DeepAutoEncoder():numLayers(0){}

DeepAutoEncoder::DeepAutoEncoder(int n, int sizes[]){
	numLayers = n;
	unitSizes[0] = sizes[0];
	for(int i=0; i<numLayers; i++){
		unitSizes[i+1] = sizes[i+1];
		layers[i] = new EncoderLayer(unitSizes[i], unitSizes[i+1]);
	}
}

DeepAutoEncoder::DeepAutoEncoder(const char * filename){
	FILE *fp = fopen(filename, "rb");
	if(fp==NULL){
		printf("file cnat not open : %s\n", filename);
		exit(1);
	}
	loadModel(fp);
	fclose(fp);
}

DeepAutoEncoder::~DeepAutoEncoder(){
	for(int i=0; i<numLayers; i++){
		delete layers[i];
	}
}

void DeepAutoEncoder::saveModel(FILE *fp){
	fwrite(&numLayers, sizeof(int), 1, fp);
	for(int i=0; i<numLayers; i++){
		layers[i]->saveModel(fp);
	}
}

void DeepAutoEncoder::loadModel(FILE *fp){
	fread(&numLayers, sizeof(int), 1, fp);
	for(int i=0; i<numLayers; i++){
		layers[i]->loadModel(fp);
	}
}

/*
 *	以下三个分别设置学习率， batchize， 内存分配
 *
 */
void DeepAutoEncoder::setLearningRate(double lrt){
	for(int i=0; i<numLayers; i++){
		layers[i]->setLearningRate(lrt);
	}
}
void DeepAutoEncoder::setBatchSize(int size){
	batchSize = size;
	for(int i=0; i<numLayers; i++){
		layers[i]->setBatchSize(size);
	}
}

void DeepAutoEncoder::mallocMemory(){
	for(int i=0; i<numLayers; i++){
		layers[i]->mallocMemory();
	}
}

/*
 * 训练模型
 *
 */
void DeepAutoEncoder::trainBatch(){
	mallocMemory();
	forward();
	backpropagate();
}

void DeepAutoEncoder::runBatch(){
	mallocMemory();
	forward();
}

void DeepAutoEncoder::forward(){
	for(int i=0; i<numLayers; i++){
		if(i!=0){
			layers[i]->setInput(layers[i-1]->h);
		}
		layers[i]->getHFromX(layers[i]->x, layers[i]->h);
	}
	for(int i=numLayers-1; i>=0; i--){
		if(i==numLayers-1)
			layers[i]->getYFromH(layers[i]->h, layers[i]->y);
		else
			layers[i]->getYFromH(layers[i+1]->y, layers[i]->y);
	}
}

void DeepAutoEncoder::backpropagate(){
	for(int i=0; i<numLayers; i++){
		if(i==0)
			layers[i]->getDeltaY(NULL);
		else
			layers[i]->getDeltaY(layers[i-1]);
	}
	for(int i=numLayers-1; i>=0; i--){
		if(i==numLayers-1){
			layers[i]->getDeltaH(NULL);
			layers[i]->updateWeight(layers[i]->h);
		}
		else{
			layers[i]->getDeltaH(layers[i+1]);
			layers[i]->updateWeight(layers[i+1]->y);
		}
		
	}

}

double DeepAutoEncoder::getTrainingCost(){
	double cost =0.0;
	int n = layers[0]->numIn;
	double *x = layers[0]->x;
	double *y = layers[0]->y;
	for(int i=0; i<batchSize*n; ++i){
		cost += -1*x[i] *log(y[i] + 1e-10) -1*(1-x[i]) *log(1-y[i] + 1e-10);
	}
	return cost /batchSize;
}
