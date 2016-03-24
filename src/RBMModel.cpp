#include <iostream>                                                                      
#include "Utility.h"
#include "RBM.h"
#include "IModel.h"
#include "TrainModel.h"
#include "Dataset.h"

void mnistTrain(){
	MNISTDataset mnist;
	mnist.loadData("./data/train-images-idx3-ubyte","./data/train-labels-idx1-ubyte");
	RBM rbm(mnist.getFeatureNumber(), 500);
	rbm.setModelFile("./result/mnistRBMModel.bin");
//	rbm.setPersist(true);
	TrainModel trainModel(&rbm);
	trainModel.train(&mnist, 0.1, 500, 10);
}

int main(){
	srand(1234);
	mnistTrain();
}
