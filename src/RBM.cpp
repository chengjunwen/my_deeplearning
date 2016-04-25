#include "RBM.h"
#include "mkl_cblas.h"

static double temp[maxUnit*maxUnit];

RBM::RBM(int numIn, int numOut){
	numVis = numIn;
	numHid = numOut;
	init();
	initWeight();
}
RBM::RBM(const char *fileName){
	init();
	FILE *fp=fopen(fileName,"rb");
	if(fp==NULL){
		printf("the file can not open!\n");
		exit(1);
	}
	loadModel(fp);
}
RBM::RBM(FILE *fp){
	init();
	loadModel(fp);
}
RBM::~RBM(){
	delete[] w;
	delete[] bv;
	delete[] bh;
	
	delete[] v2;
	delete[] pv;
	delete[] ph1;
	delete[] ph2;
	delete[] h1;
	delete[] h2;
}

void RBM::init(){
	ph1 = ph2 = h1 = h2 = v2 = pv = NULL;
	chainStart = NULL;
	binVis = true;
	binHid = true;
	persist = false;
	step = 1;
	bI = new double[maxUnit];
	for(int i=0; i<maxUnit; ++i){
		bI[i] = 1.0;                                                                    
	}
	xi = 0;
    w = new double[numVis*numHid];
	bv = new double[numVis];
	bh = new double[numHid];

	AMDelta = NULL;
}
void RBM::initWeight(){
	initWeightSigmoid(w, numVis, numHid);
	memset(bv, 0, sizeof(double)*numVis);
	memset(bh, 0, sizeof(double)*numHid);
}

void RBM::trainBatch(){
	mallocMemory();
	runChain();
	updateWeight();
	updateBias();
}
void RBM::runBatch(){
	mallocMemory();
	runChain();
}

/*
 *	分配内存
 *
 */
void RBM::mallocMemory(){
	if(ph1==NULL) ph1=new double[batchSize*numHid];
	if(ph2==NULL) ph2=new double[batchSize*numHid];
	if(h1==NULL) h1=new double[batchSize*numHid];
	if(h2==NULL) h2=new double[batchSize*numHid];
	if(pv==NULL) pv=new double[batchSize*numVis];
	if(v2==NULL) v2=new double[batchSize*numVis];
	if(AMDelta==NULL) AMDelta=new double[numVis];
}

/*
 *	链式前向
 *
 */
void RBM::runChain(){
	getProbH(v, ph1);
	if(binHid)
		getSampleH(ph1, h1);
	else{
		cblas_dcopy(batchSize*numHid, ph1, 1, h1, 1);
	}
	if(persist){
		if(chainStart==NULL){
			memset(h2, 0, sizeof(double)*batchSize*numHid);
			chainStart = h2;
		}
	}
	else
		chainStart = h1;
	gibbs_hvh(chainStart, h2, ph2, v2, pv);
}

void RBM::getProbH(double *v, double *ph){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				batchSize, numHid, numVis, 1.0,
				v, numVis, w, numHid,
				0, ph, numHid);
	cblas_dger(CblasRowMajor, batchSize, numHid,
				1.0, bI, 1, bh, 1, ph, numHid);
	if(binHid){
		for(int i=0; i<batchSize*numHid; i++){
			ph[i] = sigmoid(ph[i]);
		}
	}
}
void RBM::getProbV(double *h, double *pv){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				batchSize, numVis, numHid, 1.0,
				h, numHid, w, numHid, 
				0, pv, numVis);
	cblas_dger(CblasRowMajor, batchSize, numVis,
				1.0, bI, 1, bv, 1, pv, numVis);
    if(binVis){
        for(int i=0; i<batchSize*numVis; i++){
            pv[i] = sigmoid(pv[i]);
        }
    }

}
void RBM::getSampleH(double *ph, double *h){
	for(int i=0; i<batchSize*numHid; i++){
		h[i] = random_double(0,1) < ph[i] ? 1:0;
	}
}
void RBM::getSampleV(double *pv, double *v){
	for(int i=0; i<batchSize*numVis; i++){
		v[i] = random_double(0,1) < pv[i] ? 1:0;
	}
}

/*
 *gibbs 采样
 *
 */
void RBM::gibbs_hvh(double *hstart, double *h, double *ph, double *v, double *pv){

	cblas_dcopy(batchSize*numHid, hstart, 1, h, 1);

	for(int k=0; k<step ; k++){
		getProbV(h, pv);
        if(binVis)
        	getSampleV(pv, v);
        else
        	cblas_dcopy(batchSize*numVis, pv, 1, v, 1);
		
		getProbH(v, ph);
        if(binHid)
        	getSampleH(ph, h);
        else
            cblas_dcopy(batchSize*numHid, ph, 1, h, 1);
	}
}

/*
 * 更新权重
 *
 */
void RBM::updateWeight(){
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
				numVis, numHid, batchSize, lr/static_cast<double>(batchSize), 
				v, numVis, h1, numHid, 
				1.0, w, numHid);
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
				numVis, numHid, batchSize, -1.0*lr/static_cast<double>(batchSize), 
				v2, numVis, ph2, numHid, 
				1.0, w, numHid);
}
void RBM::updateBias(){
	cblas_dcopy(batchSize*numVis, v, 1, temp, 1);
	cblas_daxpy(batchSize*numVis, -1.0, v2, 1, temp, 1);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
				1, numVis, batchSize, lr/static_cast<double>(batchSize),
				bI, batchSize, temp, numVis,
				1.0, bv, numVis);

	cblas_dcopy(batchSize*numHid, h1, 1, temp, 1);
	cblas_daxpy(batchSize*numHid, -1.0, ph2, 1, temp, 1);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
				1, numHid, batchSize, lr/static_cast<double>(batchSize),
				bI, batchSize, temp, numHid,
				1.0, bh, numHid);
}

double RBM::getTrainingCost(){
	if(persist)
		return getPseudoCost();
	else
		return getReconstructCost();
}
double RBM::getReconstructCost(){
	 double cost = 0;
	 for(int k =0; k<batchSize*numVis; k++){
	 	cost = cost - v[k] * log(pv[k] + 1e-5) - (1-v[k]) * log(1-pv[k] +1e-5);
//	 	cost = cost - v[k] * log(pv[k] + 1e-5);
	}
	 cost = cost / batchSize;
	 return cost;
}
double RBM::getPseudoCost(){
	double *v_flip, *FE, *FE_flip;
	v_flip = new double[batchSize*numVis];
	FE = new double[batchSize];
	FE_flip = new double[batchSize];

	cblas_dcopy(batchSize*numVis, v, 1, v_flip, 1);
	for(int k=0; k<batchSize; k++){
		v_flip[k*numVis+xi] = 1- v[k*numVis+xi];
	}
	getFreeEnergy(v, FE);
	getFreeEnergy(v_flip, FE_flip);

	double cost =0;
	for(int k=0; k<batchSize; k++){
		cost += log(sigmoid(FE_flip[k] - FE[k]));
	}
	cost =cost *numVis/batchSize;
	xi = (xi+1) % numVis;

	delete[] v_flip;
	delete[] FE;
	delete[] FE_flip;
	return cost;
}
void RBM::getFreeEnergy(double *v, double *fe){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
				batchSize, numHid, numVis, 1,
				v, numVis, w, numHid, 
				0, temp, numHid);
	cblas_dger(CblasRowMajor, batchSize, numHid ,
				1.0, bI, 1, bh, 1, temp, numHid);
	for(int k=0; k<batchSize*numHid; ++k){
		temp[k] = log(1+exp(temp[k]));
	}

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				1, batchSize, numHid, 1,
				bI, numHid, temp, numHid,
				0, fe, batchSize);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				batchSize, 1, numVis, -1.0,
				v, numVis, bv, 1,
				-1.0, fe, 1);
}

void RBM::saveModel(FILE *fp){
	 fwrite(&numVis, sizeof(int), 1, fp);
	 fwrite(&numHid, sizeof(int), 1, fp);
	 fwrite(w, sizeof(double), numVis*numHid, fp);
	 fwrite(bv, sizeof(double), numVis, fp);
	 fwrite(bh, sizeof(double), numHid, fp);
}

void RBM::loadModel(FILE *fp){
	fread(&numVis, sizeof(int), 1, fp);
	fread(&numHid, sizeof(int), 1, fp);
	fread(w, sizeof(double), numVis*numHid, fp);
	fread(bv, sizeof(double), numVis, fp);
	fread(bh, sizeof(double), numHid, fp);
	printf("numIn: %d\tnumOut: %d\n", numVis, numHid);
}

/*
 * Activition Maximization
 * lastAMDelta 为NULL时， 表示最大激励化该层的节点
 *
 */

void RBM::getAMDelta(int idx, double *lastAMDelta){
	if(lastAMDelta == NULL){
		for(int i=0; i< numHid; i++){
			if(i == idx){
				if(binHid)
					temp[i] = get_sigmoid_derivate(ph1[i]);
				else
					temp[i] = 1.0;
			}
			else{
				temp[i] = 0.0;
			}
		}
	}
	else{
		for(int i=0; i<numHid; i++){
			if(binHid)
				temp[i] = lastAMDelta[i] * get_sigmoid_derivate(ph1[i]);
			else
				temp[i] = lastAMDelta[i];
		}
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				numVis, 1, numHid, 1, 
				w, numHid, temp, 1,
				0, AMDelta, 1);
}
