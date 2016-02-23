#include "MLP.h"
MLP::MLP():numLayer(0){}
MLP::MLP(const char * fileName){
	numLayer=0;

	FILE *fp = fopen(fileName,"rb");
	if(fp==NULL){
		printf("can not open the file: %s\n", fileName);
		exit(1);
	}
	loadModel(fp);
	fclose(fp);
}
MLP::~MLP(){
	for(int i=0; i<numLayer; ++i){
		delete layers[i];
	}
}

void MLP::setLearningRate(double lr){
	learningRate = lr;
	for(int i=0; i<numLayer; ++i){
		layers[i]->setLearningRate(lr);
	}
}
void MLP::setBatchSize(int size){
	batchSize = size;
        for(int i=0; i<numLayer; ++i){
                layers[i]->setBatchSize(size);
        }
}
void MLP::setInput(double *in){
	layers[0]->setInput(in);
}
void MLP::setLabel(double *la){
	label = la;
	layers[numLayer-1]->setLabel(la);
}
/*
 * 训练模型，前向， 后向
 */
void MLP::trainBatch(){
	for(int i=0; i<numLayer; ++i){
		if(i!=0){
			layers[i]->setInput( layers[i-1]->getOutput() );
		}		
		layers[i]->forward();
	}
	for(int i=numLayer-1; i>=0; --i){
		if(i==(numLayer-1))
			layers[i]->backpropagate(NULL);
		else
			layers[i]->backpropagate(layers[i+1]);
	}
}
void MLP::runBatch(){
        for(int i=0; i<numLayer; ++i){
                if(i!=0){
                        layers[i]->setInput( layers[i-1]->getOutput() );
                } 
                layers[i]->forward();
        }
}

int MLP::getInputNumber(){
	return layers[0]->getInputNumber();	
}
int MLP::getOutputNumber(){
	return layers[numLayer-1]->getOutputNumber();
}
double* MLP::getOutput(){
	return layers[numLayer-1]->getOutput();
}
double MLP::getTrainingCost(){
        double cost =0.0;
        double *y = layers[numLayer-1]->getOutput();
        for(int i=0; i<batchSize*getOutputNumber(); ++i){
                cost += -1 * label[i] * log(y[i] + 1e-10);
        }
        return cost/batchSize;
}

void MLP::saveModel(FILE *fp){
	fwrite(&numLayer, sizeof(int), 1, fp);
	char layerName[15];
	for(int i=0; i<numLayer; ++i){
		strcpy(layerName, layers[i]->getLayerName());
		fwrite(layerName, sizeof(layerName), 1, fp);
		layers[i]->saveModel(fp);
	}
}

/*
 * 加载模型，不同类型的layer
 */
void MLP::loadModel(FILE *fp){
	fread(&numLayer, sizeof(int), 1, fp);
	char layerName[15];
	for(int i=0; i<numLayer; ++i){
		fread(layerName, sizeof(layerName), 1, fp);
		if(strcmp(layerName,"sigmoid")==0)
			layers[i]=new SigmoidLayer(fp);
		else if(strcmp(layerName, "relu")==0)
			layers[i]=new ReLULayer(fp);
		else if(strcmp(layerName, "tanh")==0)
			layers[i]=new TanhLayer(fp);
		else if(strcmp(layerName, "softmax")==0)
			layers[i]=new SoftmaxLayer(fp);
			
	}
}
