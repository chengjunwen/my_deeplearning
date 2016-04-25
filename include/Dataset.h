#ifndef _DATASET_H_
#define _DATASET_H_

#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <cstdlib>
#include "Utility.h"
#include "IModelComponent.h"

class SubDataset;
class BatchIterator;

/*
 * 基类 Dataset
 */
class Dataset {
	public:
		Dataset();
		inline double * getTrainDataBatch(int offset){
			return trainData + offset*numFeature;
		}
		inline double * getTrainLabelBatch(int offset){
			return trainLabel + offset*numLabel;
		}
		inline double * getValidDataBatch(int offset){
			return validData + offset*numFeature;
		}
		inline double * getValidLabelBatch(int offset){
			return validLabel + offset*numLabel;
		}
		inline int getTrainNumber(){
			return numTrain;
		}
		inline int getValidNumber(){
			return numValid;
		}
		inline int getFeatureNumber(){
			return numFeature;
		}
		inline int getLabelNumber(){
			return numLabel;
		}
		virtual ~Dataset();
		virtual void loadDataset(const char *, const char *){}
		void dumpTrainData(const char * savefile);
		SubDataset getTrainDataset();
		SubDataset getValidDataset();
	protected:
		void dumpData(const char* savefile, int numData, double*data, double *label);
		int numTrain, numValid, numFeature, numLabel;
		double *trainData;
		double *trainLabel;
		double *validData;
		double *validLabel;
};

class MNISTDataset : public Dataset {
	public:
		MNISTDataset(){}
		void loadData(const char* DataFileName, const char * LabelFileName);
		~MNISTDataset(){}
};
class SVMDataset : public Dataset {
	public:
		SVMDataset(){}
		void loadData(const char* trainDataFileName, const char * validDataFileName);
		~SVMDataset(){}
};
class BinDataset : public Dataset {
	public:
		BinDataset(){}
		void loadData(const char* DataFileName, const char * LabelFileName);
		~BinDataset(){}
};

/*
 *  获取数据Dataset里的trainData或者validData
 */
class SubDataset {
	public:
		SubDataset(){};	
		SubDataset(int numSample, int numFeature, int numLabel, double *data, double *label):numSample(numSample),numFeature(numFeature),numLabel(numLabel),data(data),label(label){}
		~SubDataset(){}
	private:
		int numSample,numFeature,numLabel;
		double *data;
		double *label;

		friend class BatchIterator;
};

/*
 *  遍历subDataset里的batch
 */
class BatchIterator {
	public:
		BatchIterator(SubDataset * data, int batchSize):data(data), batchSize(batchSize),cur(0) {
			size = (data->numSample-1)/batchSize + 1;
		}
		inline void first() {cur=0; }
		inline void next() {cur++;}
		bool isDone();
		int getCurrentIndex() {return cur;}
		int getRealBatchSize();
		double * getCurrentDataBatch() {
			return data->data + data->numFeature * cur * batchSize;
		}
		double * getCurrentLabelBatch() {
			return data->label + data->numLabel * cur * batchSize;
		}
	private:
		SubDataset * data;
		int size, batchSize;
		int cur;
		
};

/*
 * 根据模型和输入数据， 计算该模型的输出
 */
class TransmissionDataset : public Dataset{
public:
	TransmissionDataset(Dataset * data, IModel *model);
	~TransmissionDataset(){}
};

/*
* 将多种cbind 融合
*/
class MergeDataset : public Dataset {
    public :
        MergeDataset(Dataset * originDatas[], int numSets);
        ~MergeDataset();
};
#endif
