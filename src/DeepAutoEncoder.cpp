#include "DeepAutoEncoder.h"
#include "mkl_cblas.h"

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

/*
*  存储激励最大化的最终输入sample
*/
void DeepAutoEncoder::saveSample(FILE *fp, double *s, int n){                                                  
    fwrite(s, sizeof(double), n, fp);
}

/*
 * 激励最大化第 layerdx 层，
 */
void DeepAutoEncoder::activationMaximization(int layerIdx, double argvNorm, int epoch, char * AMSampleFile){
    int AMnumOut = layers[layerIdx]->numIn;
    int AMnumIn = layers[0]->numOut;
    
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
 *  最大化第 unitIdx个单元
 */
double DeepAutoEncoder::maximizeUnit(int layerIdx, int unitIdx,
        double* unitSample, double avgNorm, int epoch)
{
    int AMnumIn = layers[0]->numIn;

    // average norm
    double curNorm = squareNorm(unitSample, AMnumIn, 1);
    cblas_dscal(AMnumIn, avgNorm / curNorm, unitSample, 1);

    double curval;

    for(int k=0; k<epoch; k++){

        // forward
        for(int i = 0; i <= layerIdx; i++){
            if(i==0){
                layers[i]->setInput(unitSample);
            }
            else{
                layers[i]->setInput(layers[i-1]->h);
            }
            layers[i]->getHFromX(layers[i]->x, layers[i]->h);
        }
        curval = layers[layerIdx]->h[unitIdx];
        //printf("unit index %d epoch %d current maximal : %.8lf\n", unitIdx+1, k, curval);

        // back-propagate
        for(int i = layerIdx; i >= 0; i--){
            if(i == layerIdx){
                layers[i]->getAMDelta(unitIdx, NULL);
            }else{
                layers[i]->getAMDelta(-1, layers[i+1]->AMdelta);
            }
        }

        // 自适应learning rate
        double lr = 0.01 * cblas_dasum(AMnumIn, unitSample, 1) /
                    cblas_dasum(AMnumIn, layers[0]->AMdelta, 1);

        // update sample
        cblas_daxpy(AMnumIn, lr,
                    layers[0]->AMdelta, 1,
                    unitSample, 1);

        // average norm
        curNorm = squareNorm(unitSample, AMnumIn, 1);
        cblas_dscal(AMnumIn, avgNorm / curNorm, unitSample, 1);
    }
    return curval;
}                                           
