#include "mkl_cblas.h"
#include "Layer.h"

static LayerBuff temp1; 	//存储中间计算结果的临时存储

Layer::Layer(int numIn, int numOut, const char *Name):numIn(numIn),numOut(numOut){
	init();
	strcpy(layerName,Name);
}
Layer::Layer(FILE *modelFile, const char *Name){
	init();
	strcpy(layerName, Name);
	loadModel(modelFile);
}
Layer::Layer(const char * fileName, const char *Name){
	init();
	strcpy(layerName, Name);
	FILE * fp =fopen(fileName, "rb");
	if(fp==NULL){
		printf("can not open file : %s\n", fileName);
		exit(1);
	}
	loadModel(fp);
	fclose(fp);
}
Layer::Layer(int numIn, int numOut, double *w, double *b, const char *Name):numIn(numIn), numOut(numOut){
	init();
	strcpy(layerName,Name);
	memcpy(weight, w, numIn*numOut*sizeof(double));
	memcpy(bias, b, numOut*sizeof(double));
}
void Layer::init(){
	weight = new double[numIn*numOut];
	bias = new double[numOut];
	out =NULL;
	delta = NULL;
//	bI = NULL;
	bI = new double[maxUnit];
	for(int i=0; i<maxUnit; ++i){
		bI[i] = 1.0;	
	}
}
Layer::~Layer(){
	delete[] weight;
	delete[] bias;
	delete[] out;
	delete[] delta;
	delete[] bI;
}

/*
 *  forward and backpropagate
 */
void Layer::forward(){
	if(out == NULL){
		out = new double[batchSize*numOut];
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
				batchSize, numOut, numIn, 1.0, 
				in, numIn, weight, numOut, 
				0, out, numOut);
	cblas_dger(CblasRowMajor, batchSize, numOut, 
			   1.0, bI, 1, bias, 1, 
			   out, numOut);
	activFunction();
}
void Layer::backpropagate(Layer *prevLayer){
	if(delta ==NULL){
		delta = new double[batchSize*numOut];
	}
	computeDelta(prevLayer);
	updateWeight();
	updateBias();
}

// comoute delta,  delta = (prevW*prevDelta) * f'
void Layer::computeDelta(Layer * prevLayer){
	double * prevLayerWeight = prevLayer->getWeight();
	double * prevLayerDelta = prevLayer->getDelta();
	int prevLayerNumOut = prevLayer->getOutputNumber();

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				batchSize, numOut, prevLayerNumOut, 1.0, 
				prevLayerDelta, prevLayerNumOut, prevLayerWeight, prevLayerNumOut, 
				0.0, temp1, numOut);

	activFunctionDerivate();
	for(int i=0; i<batchSize*numOut; ++i){
		delta[i] = temp1[i] * delta[i];
	}
	
}
//update ,w = w -lr*deltaw - 2*L2*lr*w
void Layer::updateWeight(){
	cblas_dscal(numIn*numOut, 1. -2.0*learningRate*L2Reg, weight, 1);
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
				numIn, numOut, batchSize, -1.0*learningRate/static_cast<double>(batchSize),
				in, numIn, delta, numOut, 
				1.0, weight, numOut);
}
void Layer::updateBias(){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
				1, numOut, batchSize, -1.0*learningRate/static_cast<double>(batchSize),
				bI, batchSize, delta, numOut,
				1.0, bias, numOut);
}

void Layer::saveModel(FILE *modelFile){
	fwrite(&numIn, sizeof(int), 1, modelFile);
	fwrite(&numOut, sizeof(int), 1, modelFile);
	fwrite(weight, sizeof(double), numIn*numOut, modelFile);
	fwrite(bias, sizeof(double), numOut, modelFile);
}
void Layer::loadModel(FILE *modelFile){
	fread(&numIn, sizeof(int), 1, modelFile);
	fread(&numOut, sizeof(int), 1, modelFile);
	fread(weight, sizeof(double), numIn*numOut, modelFile);
	fread(bias, sizeof(double), numOut, modelFile);
	printf("numIn: %d\tnumOut: %d\n", numIn, numOut);
}

/*
 *	Sigmoid layer
 */
SigmoidLayer::SigmoidLayer(int numIn, int numOut):Layer(numIn, numOut, "sigmoid"){
	initWeight();
	Layer::initBias();
}
SigmoidLayer::SigmoidLayer(FILE *modelFile):Layer(modelFile, "sigmoid"){}
SigmoidLayer::SigmoidLayer(int numIn, int numOut, double *w, double *b):
					 Layer(numIn, numOut, w, b, "sigmoid"){}
void SigmoidLayer::initWeight(){
	initWeightSigmoid(weight, numIn, numOut);
}
void SigmoidLayer::activFunction(){
	for(int i=0; i<batchSize*numOut; ++i){
		out[i] = sigmoid(out[i]);
	}
}
void SigmoidLayer::activFunctionDerivate(){
	for(int i=0; i<batchSize*numOut; ++i){
		delta[i] = get_sigmoid_derivate(out[i]);
	}
}

/*
 * ReLU layer
 */
ReLULayer::ReLULayer(int numIn, int numOut):Layer(numIn, numOut, "relu"){
	initWeight();
	Layer::initBias();
}
ReLULayer::ReLULayer(FILE *modelFile):Layer(modelFile, "relu"){}
ReLULayer::ReLULayer(int numIn, int numOut, double *w, double *b):
					 Layer(numIn, numOut, w, b, "relu"){}
void ReLULayer::initWeight(){
	initWeightSigmoid(weight, numIn, numOut);
}
void ReLULayer::activFunction(){
	for(int i=0; i<batchSize*numOut; ++i){
		out[i] = relu(out[i]);
	}
}
void ReLULayer::activFunctionDerivate(){
	for(int i=0; i<batchSize*numOut; ++i){
		delta[i] = get_sigmoid_derivate(out[i]);
	}
}

/*
 *	Tanh layer
 */
TanhLayer::TanhLayer(int numIn, int numOut):Layer(numIn, numOut, "tanh"){
	initWeight();
	Layer::initBias();
}
TanhLayer::TanhLayer(FILE *modelFile):Layer(modelFile, "tanh"){}
TanhLayer::TanhLayer(int numIn, int numOut, double *w, double *b):
					 Layer(numIn, numOut, w, b, "tanh"){}
void TanhLayer::initWeight(){
	initWeightTanh(weight, numIn, numOut);
}
void TanhLayer::activFunction(){
	for(int i=0; i<batchSize*numOut; ++i){
		out[i] = tanh(out[i]);
	}
}
void TanhLayer::activFunctionDerivate(){
	for(int i=0; i<batchSize*numOut; ++i){
		delta[i] = get_tanh_derivate(out[i]);
	}
}

/*
 * softmax layer
 */
SoftmaxLayer::SoftmaxLayer(int numIn, int numOut):Layer(numIn, numOut, "softmax"){
	initWeight();
	Layer::initBias();
}
SoftmaxLayer::SoftmaxLayer(const char * fileName):Layer(fileName, "softmax"){}
SoftmaxLayer::SoftmaxLayer(FILE *modelFile):Layer(modelFile, "softmax"){}
SoftmaxLayer::SoftmaxLayer(int numIn, int numOut, double *w, double *b):
					 Layer(numIn, numOut, w, b, "softmax"){}
void SoftmaxLayer::initWeight(){
	memset(weight, 0, numIn*numOut*sizeof(double));
}
void SoftmaxLayer::activFunction(){
	for(int i=0; i<batchSize; i++){
		softmax(out + i*numOut, numOut);
	}
}
