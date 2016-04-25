#include "EncoderLayer.h"
#include "mkl_cblas.h"

EncoderLayer::EncoderLayer(int numIn, int numOut){
	this->numIn = numIn;
	this->numOut = numOut;
	init();
	initWeight();
}

EncoderLayer::EncoderLayer(int numIn, int numOut, double *w, double *b, double *c){
	this->numIn = numIn;
	this->numOut = numOut;
	init();
	memcpy(this->w, w, numIn*numOut*sizeof(double));
	memcpy(this->b, b, numOut*sizeof(double));
	memcpy(this->c, c, numIn*sizeof(double));
}

EncoderLayer::EncoderLayer(FILE *fp){
	init();
	loadModel(fp);
}

EncoderLayer::~EncoderLayer(){
	delete[] w;
	delete[] b;
	delete[] c;
	delete[] dh;
	delete[] dy;
	delete[] h;
	delete[] y;
}
void EncoderLayer::init(){
	y = h = dh = dy =NULL;
	binIn = true;
	binOut = true;
    bI = new double[maxUnit];
    for(int i=0; i<maxUnit; ++i){
        bI[i] = 1.0;                                                                    
    }
	w = new double[numIn*numOut];
	b = new double[numOut];
	c = new double[numIn];
}

void EncoderLayer::initWeight(){
	initWeightSigmoid(w, numIn, numOut);
	memset(b, 0, numOut*sizeof(double));
	memset(c, 0, numIn*sizeof(double));
}
void EncoderLayer::mallocMemory(){
	if(h==NULL)  h = new double[batchSize*numOut];
	if(y==NULL)  y = new double[batchSize*numIn];
	if(dh==NULL)  dh = new double[batchSize*numOut];
	if(dy==NULL)  dy = new double[batchSize*numIn];
}

void EncoderLayer::saveModel(FILE *fp){
	fwrite(&numIn, sizeof(int), 1, fp);
	fwrite(&numOut, sizeof(int), 1, fp);
	fwrite(w, sizeof(double), numIn*numOut, fp);
	fwrite(b, sizeof(double), numOut, fp);
	fwrite(c, sizeof(double), numIn, fp);
}

void EncoderLayer::loadModel(FILE *fp){
	fread(&numIn, sizeof(int), 1, fp);
	fread(&numOut, sizeof(int), 1, fp);
	fread(w, sizeof(double), numIn*numOut, fp);
	fread(b, sizeof(double), numOut, fp);
	fread(c, sizeof(double), numIn, fp);
	printf("numIn: %d\tnumOut: %d\n", numIn, numOut);
}

void EncoderLayer::getHFromX(double *x, double *h){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				batchSize, numOut, numIn, 1.0,
				x, numIn, w, numOut, 
				0, h, numOut);
	cblas_dger(CblasRowMajor, batchSize, numOut, 1.0,
				bI, 1, b, 1, h, numOut);
	if(binOut){
		for(int i=0; i<numOut*batchSize; i++){
			h[i] = sigmoid(h[i]);
		}
	}
}

void EncoderLayer::getYFromH(double *h, double *y){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				batchSize, numIn, numOut, 1.0,
				h, numOut, w, numOut, 
				0, y, numIn);
	cblas_dger(CblasRowMajor, batchSize, numIn, 1.0,
				bI, 1, c, 1, y, numIn);
	if(binIn){
		for(int i=0; i<numIn*batchSize; i++){
			y[i] = sigmoid(y[i]);
		}
	}
}

void EncoderLayer::getDeltaH(EncoderLayer * prevLayer){
	if(prevLayer == NULL ){
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					batchSize, numOut, numIn, 1.0,
					dy, numIn, w, numOut, 
					0, dh, numOut);
		if(binOut){
			for(int i=0; i<numOut * batchSize; i++){
				dh[i] = get_sigmoid_derivate(h[i]) * dh[i];
			}
		}
	}
	else{
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
					batchSize, prevLayer->numIn, prevLayer->numOut, 1.0,
					prevLayer->dh, prevLayer->numOut, prevLayer->w, prevLayer->numOut,
					0, dh, prevLayer->numIn);
		if(binOut){
			for(int i=0; i<numOut *batchSize; i++){
				dh[i] = get_sigmoid_derivate(h[i]) * dh[i];
			}
		}
	}
}

void EncoderLayer::getDeltaY(EncoderLayer * prevLayer){
	if(prevLayer == NULL ){
		if(binOut){
			cblas_dcopy(batchSize*numIn, y, 1, dy, 1);
			cblas_daxpy(batchSize*numIn, -1.0, x, 1, dy, 1);
		}
		else{
			for(int i=0; i<numIn*batchSize; i++){
				dy[i] = (y[i] - x[i]) / ((y[i] + 1e-10) * (1- y[i] + 1e-10));
			}
		}
	}
	else{
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					batchSize, prevLayer->numOut, prevLayer->numIn, 1.0,
					prevLayer->dy, prevLayer->numIn, prevLayer->w, prevLayer->numOut,
					0, dy, prevLayer->numOut);
		if(binOut){
			for(int i=0; i<numIn *batchSize; i++){
				dy[i] = get_sigmoid_derivate(y[i]) * dy[i];
			}
		}
	}
}

void EncoderLayer::updateWeight(double * prevH){
// update weight
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
				numIn, numOut, batchSize, -1.0*lr/static_cast<double>(batchSize),
				dy, numIn, prevH, numOut,
				1, w, numOut);
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				numIn, numOut, batchSize, -1.0*lr/static_cast<double>(batchSize),
				x, numIn, dh, numOut,
				1, w, numOut);
//update bias
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				1, numIn, batchSize, -1.0*lr/static_cast<double>(batchSize), 
				bI, batchSize, dy, numIn,
				1, c, numIn);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				1, numOut, batchSize, -1.0*lr/static_cast<double>(batchSize),
				bI, batchSize, dh, numOut, 
				1, b, numOut);
}
