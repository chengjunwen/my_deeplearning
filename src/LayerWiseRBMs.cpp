#include "LayerWiseRBMs.h"

LayerWiseRBMs::LayerWiseRBMs():numLayers(0), AMSample(NULL){}
LayerWiseRBMs::LayerWiseRBMs(int n, int sizes[]):numLayers(n), AMSample(NULL){
	for(int i=0; i<numLayers+1; i++){
		unitSizes[i] = sizes[i];
		if(i!=numLayers){
			layers[i]=new RBM(sizes[i], sizes[i+1]);
		}
	}
}
LayerWiseRBMs::LayerWiseRBMs(const char * fileName):AMSample(NULL){
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

void LayerWiseRBMs::setPersistent(bool f){
	for(int i=0; i<numLayers; i++){
		layers[i]->setPersist(f);
	}
}

void LayerWiseRBMs::addLayer(RBM * layer){
	layers[numLayers]=layer;
	if(numLayers ==0){
		unitSizes[0] = layer->getInputNumber();
	}
	numLayers++;
	unitSizes[numLayers] = layer->getOutputNumber();
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


/*
 * 存储激励最大化的最终样本
 *
 */
void LayerWiseRBMs::saveSample(FILE *fp, double *s, int n){
	fwrite(s, sizeof(double), n, fp);
}
/*
 * 激励最大化第 layerdx 层，
 */
void LayerWiseRBMs::activationMaximization(int layerIdx, double argvNorm, int epoch, char * AMSampleFile){
	int AMnumOut = layers[layerIdx]->numHid;
	int AMnumIn = layers[0]->numVis;
	
	if(AMSample == NULL){
		AMSample = new double[AMnumOut*AMnumIn];
	}
	for(int i=0; i<AMnumOut*AMnumIn; i++){
		AMSample[i] = random_double(0, 1);
	}

	FILE * fp = fopen(AMSampleFile, "w+");
	fwrite(&AMnumIn, sizeof(int), 1, fp);
	fwrite(&AMnumOut, sizeof(int), 1, fp);
	for(int i=0; i<AMnumOut; i++){
		double * unitSample = AMSample + i*AMnumIn;
		time_t start, stop;
		start = time(NULL);
		double maxValue = maximizeUnit(layerIdx, i, unitSample, argvNorm, epoch);
		stop = time(NULL);
		printf("layer: %d , unit: %d, max value : %.6lf\t time: %.2lf s\n", layerIdx, i, maxValue, difftime(stop, start));
		saveSample(fp, unitSample, AMnumIn);
	}

	fclose(fp);
}

/*
 * 最大最大激励化 第 unitdx 个单元
 *
 */
double LayerWiseRBMs::maximizeUnit(int layerIdx, int unitIdx, double * unitSample, double argvNorm, int epoch){

    int AMnumIn = layers[0]->numVis;                                            

    // unitsample 归一化
    double curNorm = squareNorm(unitSample, AMnumIn, 1);
    cblas_dscal(AMnumIn, argvNorm / curNorm, unitSample, 1);
	
	double maxValue =0;

	for(int k=0; k<epoch; k++){
	// forward
		for(int i=0; i<=layerIdx; i++){
			if(i==0)
				layers[i]->setInput(unitSample);
			else
				layers[i]->setInput(layers[i-1]->getOutput());
			layers[i]->setBatchSize(1);
			layers[i]->runBatch();
		}
		maxValue = layers[layerIdx]->getOutput()[unitIdx];
	//back propagate
		for(int i=layerIdx; i>=0; i--){
			if(i==layerIdx)
				layers[i]->getAMDelta(unitIdx, NULL)	;
			else
				layers[i]->getAMDelta(-1, layers[i+1]->AMDelta);
		}
        double lr = 0.01 * cblas_dasum(AMnumIn, unitSample, 1) /                
                    cblas_dasum(AMnumIn, layers[0]->AMDelta, 1);
		
	// update unitSample
		cblas_daxpy(AMnumIn, lr, layers[0]->AMDelta, 1, unitSample, 1);
	//归一化 unitSample
		curNorm = squareNorm(unitSample, AMnumIn, 1);
	    cblas_dscal(AMnumIn, argvNorm / curNorm, unitSample, 1);
	
	}
	return maxValue;
}
