#ifndef _DATASET_H_
#define _DATASET_H_

#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <cstdlib>

class SubDataset;
class BatchIterator;

class Dataset {
	publict:
		Dataset();
		inline double * getTrainingDataBatch(int offset){
			return trainData + offset*numFeature;
		}
		inline double * getTrainingLabelBatch(int offset){
			return trainLabel + offset*numLabel;
		}
		inline double * getValidateDataBatch(int offset){
			return validData + offset*numFeature;
		}
		inline double * getValidateLabelBatch(int offset){
			return validLabel + offset*numLabel;
		}
		inline int getTrainingNumber(){
			return numTrain;
		}
		inline int getValidateNumber(){
			return numValid;
		}
		inline int getFeatureNumber(){
			return numFeature;
		}
		inline int getLabelNumber(){
			return numLabel;
		}
		virtual ~Dataset();
		virtual loadDataset(const char *, const char *);
		void dumpTrainingData(const char * savefile);
		SubDataset getTrainDataset();
		SubDataset getValidDataset();
	protected:
		void dumpData(const char* savefile, int numData, double*data, double *label);
		int numTrain, numValid, numFeature, numLabel;
		double *tainData;
		double *trainLabel;
		double *validData;
		double *validLabel;
};

class MNISTDataset : public Dataset {
	public:
		MNISTDataset(){}
		loadData(const char* DataFileName, const char * LabelFileName);
		~MNISTDataset(){}
};
class SVMDataset : public Dataset {
	public:
		SVMDataset(){}
		loadData(const char* trainDataFileName, const char * validDataFileName);
		~SVMDataset(){}
};
class BinDataset : public Dataset {
	public:
		BinDataset(){}
		loadData(const char* DataFileName, const char * LabelFileName);
		~BinDataset(){}
};

/*
 *  获取数据Dataset里的trainData或者validData
 */
class SubDataset {
	ptchIteraror()ublic:
		SubDataset(int numSample, int numFeature, int numLabel, double *data, double label):numSample(numSample),numFeature(numFeature),numLabel(numLabel),data(data),label(label){}
	private:
		int numSample,numFeature,numLabel;
		double *data;
		double *label;
		~SubDataset(){}

		friend class BarchIterator;
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
		int getCurrentIndex() {return cur};
		int getRealBatchSize();
		double * getCurrentDataBatch() {
			return data->data + data->numFeature * cur * batchsize;
		}
		double * getCurrentLabelBatch() {
			return data->label + data->numLabel * cur * batchsize;
		}
	private:
		SubDataset * data;
		int size, batchSize;
		int cur;
		
};
