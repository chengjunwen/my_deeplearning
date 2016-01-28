#ifndef _IMODEL_H_
#define _IMODEL_H_

#include <cstdio>
#include <cstdlib>
#include <string.h>

enum ModelType { Unsupervise, Supervise };

/*
 * Model 定义，只用于接口调用，不会进行实例化
 */

class IModel {
	public :
		IModel(ModelType);
		virtual ~IModel();
		virtual void setLearningRate(double lr)=0;
		virtual void setBatchSize(int)=0;
	    virtual void trainBatch()=0;
		virtual void runBatch()=0;
		virtual void setInput(double *)=0;
		virtual void setLabel(double *)=0;
		ModelType getModelType(){ return modelType; }
		virtual int getInputNumber()=0;
		virtual int getOutputNumber()=0;
		virtual double * getOutput()=0;
		virtual double * getLabel()=0;
		virtual double getTrainingCost(){ return 0.0; }
			
		void setModelFile(const char * fileName);
		void saveModel();//用于外部调用该方法来存储模型，内部调用下面的存储函数
		virtual void saveModel(FILE *modelFile)=0;//用于子类实现并调用子类的方法
		
	protected :
		char * modelFileName;
		ModelType modelType;
};

class SuperviseModel : public IModel {
	public:
		SuperviseModel();
		virtual ~SuperviseModel(){}
};

class UnsuperviseModel : public IModel {
	public:
		UnsuperviseModel();
		virtual ~UnsuperviseModel(){}
};
#endif
