#ifndef _IMODEL_H_
#define _IMODEL_H_

#include <cstdio>
#include <cstdlib>
#include <string.h>

// 枚举， 有监督 或者 无监督
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

/*
 * 有监督模型
 */
class SuperviseModel : public IModel {
	public:
		SuperviseModel();
		virtual ~SuperviseModel(){}
};

/*
 * 无监督模型
 */
class UnsuperviseModel : public IModel {
	public:
		UnsuperviseModel();
		virtual ~UnsuperviseModel(){}
		void setLabel(double *){}
		double * getLabel(){ return NULL; } //纯虚函数必须实现， 否则该类不能实例化，此处实现是便于各类无监督模型的实例化
};

/*
 *   层与层之间独立的模型， 例如 stackAutoEncoder， stackRBMs, 
 */

class LayerWiseModel : public IModel {
	public:
		LayerWiseModel();
		virtual ~LayerWiseModel(){}
		virtual int getNumLayer() = 0;
		virtual IModel * getLayerModel(int i)=0;

//实现IMolde的纯虚函数
        void setLearningRate(double lr){}
        void setBatchSize(int){}
        void trainBatch(){}
        void runBatch(){}
        void setInput(double *){}
        void setLabel(double *){}
        int getInputNumber(){ return 0; }
        int getOutputNumber(){ return 0; }
        double * getOutput(){ return NULL;}
        double * getLabel(){ return NULL;}
        double getTrainingCost(){ return 0; }

};
#endif
