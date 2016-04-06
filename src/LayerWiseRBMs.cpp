#include "LayerWiseRBMs.h"

LayerWiseRBMs::LayerWiseRBMs(int n, int sizes[]):numLayers(n){
	for(int i=0; i<numLayers+1; i++){
		unitSizes[i] = sizes[i];
		if(i!=0){
			layers[i-1]=new RBM(unitSizes[i-1], unitSizes[i]);
		}
	}
}
LayerWiseRBMs::LayerWiseRBMs(const char * fileName){
	FILE *fp =fopen(fileName, "rb");
	if(fp==NULL){
		printf("can not open the file: %s\n", fileName);
		exit(1);
	}
	loadModel(fp);
	fclose(fp);
}
LayerWiseRBMs::~LayerWiseRBMs(){
	for(int i=0; i<numLayers; i++){
		delete layers[i];
	}
}


void LayerWiseRBMs::loadModel(FILE *fp){
	fread(&numLayers, sizeof(int), 1, fp);
	for(int i=0; i<numLayers; i++){
		layers[i] = new RBM(fp);
	}
}
void LayerWiseRBMs::saveModel(FILE *fp){
	fwrite(&numLayers, sizeof(int), 1, fp);
	for(int i=0; i<numLayers; i++){
		layers[i]->saveModel(fp);
	}
}

void LayerWiseRBMs::toMLP(MLP *mlp, int numLabel){
	for(int i=0; i<numLayers; i++){
		double *w = layers[i]->getWeight();
		double *b = layers[i]->getBias();
		Layer *layer=new SigmoidLayer(unitSizes[i], unitSizes[i+1], w, b);
		mlp->addLayer(layer);
	}
	mlp->addLayer(new Logistic(unitSizes[numLayers], numLabel));
}
