#include <iostream>
#include "../include/Utility.h"
#include "Logistic.h"
#include "TrainModel.h"

void mnistTrain(){
	MNISTDataset mnist();
	mnist.loadData("train-images-idx3-ubyte","train-labels-idx1-ubyte");
	Logistic log(mnist.getFeatureNumber(), mnist.getLabelNumber());
	log.setModelFile("./result/mnistLogisticModel.dat");
	TrainModel model(log);
	model.train(&mnist, 0.01, 100, 1000);
}

int main(){
	mnistTrain();
}
