#include "Utility.h"
#include "LayerWiseRBMs.h"
#include "MLP.h"
#include "TrainModel.h"
#include "Dataset.h"

void mnistTrain(){

    MNISTDataset mnist;
    mnist.loadData("./data/train-images-idx3-ubyte","./data/train-labels-idx1-ubyte");

	int layerUnits[] = { mnist.getFeatureNumber(), 500, 200};
	LayerWiseRBMs dbn( 2, layerUnits);
	dbn.setModelFile("./result/mnistDBNModel.bin");
	LayerWiseTrainModel dbnTrainModel(&dbn);
	dbnTrainModel.train(&mnist, 0.01, 10, 2);

	MLP mlp;
	dbn.toMLP(&mlp, mnist.getLabelNumber());
	mlp.setModelFile("./result/mnistMLP_pretrain.bin");
	TrainModel trainModel(&mlp);
	printf("train model------------\n");
	trainModel.train(&mnist, 0.01, 10, 1000);
}

int main(){
	mnistTrain();
}
