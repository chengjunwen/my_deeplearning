#include "Utility.h"
#include "MLP.h"
#include "IModel.h"
#include "TrainModel.h"
#include "Dataset.h"
#include "Logistic.h"

void mnistTrain(){
        MNISTDataset mnist;
        mnist.loadData("./data/train-images-idx3-ubyte","./data/train-labels-idx1-ubyte");
	MLP mlp;
	Layer *layer1 = new SigmoidLayer(mnist.getFeatureNumber(), 500);
	Logistic *layer2 = new Logistic(500, mnist.getLabelNumber());
	mlp.addLayer(layer1);
	mlp.addLayer(layer2);
	TrainModel trainmodel(&mlp);
	trainmodel.train(&mnist, 0.03, 81, 1000);
}

int main(){
	mnistTrain();
}
