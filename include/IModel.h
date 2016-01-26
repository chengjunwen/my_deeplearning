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
		IModel(ModelType t);
		virtual ~IModel(){}
		virtual void setLearningRate(double lr);
		virtual void setBatchSize(int);
	    virtual void trainBatch();
		virtual void runBatch();
		virtual void setInput(double *);
		virtual void setLabel(double *);
		ModelType getTrainType(){ return modelType; }
		virtual int getInputNumber();
		virtual int getOutputNumber();
		virtual double * getOutput();
		virtual double * getLabel();
		virtual double getTrainingCost();
			
		void setModelFile(const char * fileName);
		void saveModel();//用于外部调用该方法来存储模型，内部调用下面的存储函数
		virtual void saveModel(FILE *modelFile);//用于子类实现并调用子类的方法

	protected :
		char modelFileName[20];
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
