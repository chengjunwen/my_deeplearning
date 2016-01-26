#ifndef _LAYER_H_
#define _LAYER_H_
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "Utility.h"

class MLP;
typedef double LayerBuff[maxUnit * maxUnit];	//缓存，存储计算过程中的各种中间值
/*
 *  基类Layer ，实现Layer的各种基本参数和方法
 *  sigmoid，tanh等layer继承基类，实现各自的不同的计算和更新方式
 */
class Layer {
	public :
		Layer(int numIn, int numOut, const char *Name);
		Layer(FILE * modelFile, const char *Name);
		Layer(const char *fileName,const char *Name);
		Layer(int numIn, int numOut, double *w, double *b, const char *Name);
		void setLearningRate(double lr){ learningRate=lr; }
		void setBatchSize(int size){ batchSize=size; }
		void setInput(double * input){ in = input; }
		
		void forward();
		void backpropagate(Layer * prevLayer);
		void saveModel(FILE * modelFile);
		void loadModel(FILE * modelFile);
		double * getInput(){ return in;}
		double *getOutput(){ return out;}
		double *getWeight(){ return weight;}
		double *getBias(){ return bias;}
		double *getDelta(){ return delta;}
		int getInputNumber(){ return numIn;}
		int getOutputNumber(){ return numOut;}
		char * getLayerName(){ return layerName;}

		void initWeight(){}	//为了能在构造器中调用该函数，不能使用虚函数
		void initBias(){ memset(bias, 0, numOut*sizeof(double)); } 

//		virtual void setLabel(double *label);	//不一定会用到
		virtual ~Layer();	//虚析构函数，便于基类指针指向子类时执行子类析构函数

	protected :

		void updateWeight();
		void updateBias();
		void computeDelta(Layer * prevLayer);
		
		virtual void activFunction();
		virtual void activFunctionDerivate();

		void init();
		void loadModel(const char *fileName);

		int numIn, numOut;
		double *in, *out;
		double *weight, *bias, *delta, *bI; //bI存储单元向量用于计算
	    double learningRate;			
		int batchSize;
		char layerName[15];
};

class SigmoidLayer : public Layer {
	public :
		SigmoidLayer(int numIn, int numOut);
		SigmoidLayer(FILE * modelFile);
//		SigmoidLayer(const char * fileName);
		SigmoidLayer(int numIn, int numOut, double *w, double*b);
	private :
		void activFunction();
		void activFunctionDerivate();
		void initWeight();
};
class ReLULayer : public Layer {
	public :
		ReLULayer(int numIn, int numOut);
		ReLULayer(FILE * modelFile);
//		ReLULayer(const char * fileName);
		ReLULayer(int numIn, int numOut, double *w, double*b);
	private :
		void activFunction();
		void activFunctionDerivate();
		void initWeight();
};
class TanhLayer : public Layer {
	public :
		TanhLayer(int numIn, int numOut);
		TanhLayer(FILE * modelFile);
//		TanhLayer(const char * fileName);
		TanhLayer(int numIn, int numOut, double *w, double*b);
	private :
		void activFunction();
		void activFunctionDerivate();
		void initWeight();
};
class SoftmaxLayer : public Layer {
	public :
		SoftmaxLayer(int numIn, int numOut);
		SoftmaxLayer(FILE * modelFile);
		SoftmaxLayer(const char * fileName);
		SoftmaxLayer(int numIn, int numOut, double *w, double*b);
	private :
		void activFunction();
		void activFunctionDerivate(){}
		void initWeight();
};
#endif
