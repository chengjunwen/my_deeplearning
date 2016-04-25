#ifndef _MULTIMODALCOMPONENT_H_
#define _MULTIMODALCOMPONENT_H_

#include "Dataset.h"  
#include "Utility.h"
#include "MultiModal.h"
#include "LayerWiseRBMs.h"
#include "TrainModel.h"

class MultiModalModel{
	public:
		MultiModalModel(int n);
		virtual ~MultiModalModel(){}

		void setModelFile(const string modelFileNames[]);
		void setGaussVisible(bool gauss[]);
		int getNumModel(){ return numModel; }
		virtual void trainModel(Dataset *, double [], int, int []) =0 ;
		virtual Dataset *runModel(Dataset *) = 0;

	protected :
		int numModel;
		LayerWiseRBMs * multiModels[maxModel];

		
};

class MultiModalRBMs : public MultiModalModel{
	public:
		MultiModalRBMs();
		MultiModalRBMs(int numModel, const string modelFileNames[]);
		~MultiModalRBMs();
		void trainModel(Dataset *dataset, double lr[], int batchSize, int epoch[]);
		void addModel(LayerWiseRBMs *model) { multiModels[numModel++] = model; }
		Dataset * runModel(Dataset *dataset);
		
};

class SpiralMultiModal : public MultiModalModel {
    public:
        SpiralMultiModal();
        SpiralMultiModal(int numModel, const string modelFileNames[]);
        ~SpiralMultiModal();

            
        void addModel(LayerWiseRBMs * model){
            multiModels[numModel] = model;
            multiModels[numModel] ->setPersistent(false);
            numModel++;
        }
        void addModel(int numLayer, int unitSize[]);

		void trainModel(Dataset *dataset, double lr[], int batchSize, int epoch[]);
		void trainModel(Dataset *dataset, double lr, int batchSize, int epoch);
		Dataset * runModel(Dataset *dataset);
		Dataset * getHiddenOutput(int k){ return hiddenDataset[k-1]; } 

		void setDataset(Dataset *dataset, int n);


    private :
		Dataset *datas[];
        Dataset * hiddenDataset[maxModel];
		void trainOneModel(int, Dataset *, double lr, int batchSize, int epoch);
		Dataset *runOneModel(int , Dataset *);
};

#endif


