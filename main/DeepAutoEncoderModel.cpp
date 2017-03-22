#include "Utility.h"                                                                     
#include "DeepAutoEncoder.h"
#include "EncoderLayer.h"
#include "TrainModel.h"
#include "Dataset.h"

void mnistTrain(){
    MNISTDataset mnist;
   	mnist.loadData("./data/train-images-idx3-ubyte","./data/train-labels-idx1-ubyte");
    int sizes[] = {mnist.getFeatureNumber(), 100, 10};
	DeepAutoEncoder dae(2, sizes);
	dae.setModelFile("result/DAE_modelFile.dat");
	TrainModel trainmodel(&dae);
    trainmodel.train(&mnist, 0.01, 20, 500);

	string savefile="./result/mnist_dae_result.bin";
    TransmissionDataset out(&mnist,&dae);
	out.dumpTrainData(savefile.c_str());

}

int main(){
    mnistTrain();
}

