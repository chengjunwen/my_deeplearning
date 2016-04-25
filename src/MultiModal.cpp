#include "MultiModal.h"	
void logisticUp(Dataset * combinDataSet){
        Logistic logi(combinDataSet->getFeatureNumber(), combinDataSet->getLabelNumber());
        TrainModel logisticModel(&logi);
        logisticModel.train(combinDataSet,0.01,10,1000);

}
void mlpUp(Dataset * combinDataSet, int numHid, double lr){
        MLP mlp;
        Layer *firstLayer = new SigmoidLayer(combinDataSet->getFeatureNumber(), numHid);
        Logistic *secondLayer = new Logistic(numHid,combinDataSet->getLabelNumber());
        mlp.addLayer(firstLayer);
        mlp.addLayer(secondLayer);
        TrainModel mlpModel(&mlp);
        mlpModel.train(combinDataSet, lr,10,1000);
}
void DBNUp(Dataset * combinDataSet, int nLayers, int layerSizes[], double lr, int batchsize, int epochs){
// train common DBN model
//        int comLayers = 1;
//        int commonLayerSize[] = {combinDataSet->getFeatureNumber(),100};
        LayerWiseRBMs common_DBN(nLayers,layerSizes);    //unfied hidden layer
//        int epochs = 100;
        common_DBN.setModelFile("result/common_model.dat");
        common_DBN.setPersistent(false);
        LayerWiseTrainModel preTrainModel(&common_DBN);
        preTrainModel.train(combinDataSet, lr, batchsize, epochs);

        MLP mlp;
        common_DBN.toMLP(&mlp,combinDataSet->getLabelNumber());
        mlp.setModelFile("result/com_MSI_GMCNV_mlp.dat");
        TrainModel supervisedModel(&mlp);
        supervisedModel.train(combinDataSet, lr/10,batchsize,epochs*10);

}
void DeepAd(Dataset * combinDataSet, int nLayers, int layerSizes[],double lr, int batchsize, int epochs, string savefile){
	printf("lr: %f\nlayers: %d\nlayersize: %d\t%d\t%d\nbatchsize :%d\nepochs :%d\nsavefile: %s\n",
				lr, nLayers, layerSizes[0],layerSizes[1], layerSizes[2],
				batchsize, epochs, savefile.c_str());
	DeepAutoEncoder dad(2,layerSizes);
	dad.setModelFile("result/com_dad.dat");
	TrainModel model(&dad);
	model.train(combinDataSet,lr,batchsize,epochs);
	
	TransmissionDataset out(combinDataSet,&dad);
	out.dumpTrainData(savefile.c_str());
}
void StackRBM(Dataset * combinDataSet, int nLayers, int layerSizes[], double lr , int batchsize, int epochs, string savefile){
	printf("lr: %f\nlayers: %d\nlayersize: %d\t%d\t%d\nbatchsize :%d\nepochs :%d\nsavefile: %s\n",
				lr, nLayers, layerSizes[0],layerSizes[1], layerSizes[2],
				batchsize, epochs, savefile.c_str());
        LayerWiseRBMs common_DBN(nLayers,layerSizes);    //unfied hidden layer
        common_DBN.setModelFile("result/common_model.dat");
        common_DBN.setPersistent(false);
        LayerWiseTrainModel preTrainModel(&common_DBN);
        preTrainModel.train(combinDataSet, lr, batchsize, epochs);

        Dataset * curData = combinDataSet;
        for(int j=0; j<nLayers; j++){
	        Dataset * tmData = new TransmissionDataset(curData, common_DBN.getLayer(j) );
            if(j == nLayers-1)
            	tmData->dumpTrainData(savefile.c_str());
       		if(curData!=combinDataSet){
            	delete curData;
       		}
        	curData = tmData;
   		}
        if(curData!=combinDataSet)
                delete curData;
}












