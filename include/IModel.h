#ifndef _IMODEl_H_
#define _IMODEL_H_

#include <cstdio>
#include <cstdlib>

enum ModelType { Unsuperise, Supervise };

/*
 * Model 定义，只用于接口调用，不会进行实例化
 */
class IModel {
	public :
		IModel(ModelType t);
		virtual ~Imodel(){}
		virtual void setLearningRate();
		virtual void setBatchSize()
	    virtual void trainBatch();
		virtual void runBatch();
		virtual void setInput();
		virtual void setLabel();

		virtual int getInputNumber();
		virtual int getOutputNumber();
		virtual double * getOutput();
		virtual double getTrainingCost();
		
		void setModelFile(const char * fileName);
		void saveModel();//用于外部调用该方法来存储模型，内部调用下面的存储函数
		virtual void saveModel(FILE *modelFile);//用于子类实现并调用子类的方法

	protected :
		char modelFileName[20];
		ModelType modelType;
};
class SuperviseModel : public Imodel {
	SuperviseModel();
	virtual ~SuperviseModel(){}
};
class UnsuperviseModel : public Imodel {
	UnsuperviseModel();
	virtual ~UnsuperviseModel(){}
};
