#include <iostream>
#include "Utility.h"
#include "Logistic.h"
#include "IModel.h"
#include "TrainModel.h"
#include "Dataset.h"

void mnistTrain(){
	MNISTDataset mnist;
	mnist.loadData("./data/train-images-idx3-ubyte","./data/train-labels-idx1-ubyte");
	Logistic log(mnist.getFeatureNumber(), mnist.getLabelNumber());
	log.setModelFile("./result/mnistLogisticModel.dat");
	TrainModel trainmodel(&log);
	trainmodel.train(&mnist, 0.01, 10, 1000);
}

int main(){
	mnistTrain();
}
