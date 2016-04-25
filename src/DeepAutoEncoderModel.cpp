#include "Utility.h"                                                                     
#include "DeepAutoEncoder.h"
#include "TrainModel.h"
#include "Dataset.h"

void mnistTrain(){
    MNISTDataset mnist;
   	mnist.loadData("./data/train-images-idx3-ubyte","./data/train-labels-idx1-ubyte");
    int sizes[] = {mnist.getFeatureNumber(), 500, 100};
	DeepAutoEncoder dae(2, sizes);
	TrainModel trainmodel(&dae);
    trainmodel.train(&mnist, 0.01, 20, 5);
}

int main(){
    mnistTrain();
}

