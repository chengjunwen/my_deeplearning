#include "MultiModal.h"
#include "Dataset.h"
#include "MultiModalComponent.h"
#include <string>


void multiModalMSI(int numModels, double lrt[], int numH, int epochs, string savefile1){
	BinDataset dataset[maxModel];
//	dataset[0].loadData("../../binFile/7msi_6828_5000_nonames.bin","../../binFile/7random_label_6828.bin");
//	dataset[1].loadData("../../binFile/7cnv_6828_1412_nonames.bin","../../binFile/7random_label_6828.bin");
//	dataset[2].loadData("../../binFile/7snv_6828_5026_nonames.bin","../../binFile/7random_label_6828.bin");
	dataset[0].loadData("../../binFile/7cnv_6828_1077_nonames.bin","../../binFile/7random_label_6828.bin");
	int specificLayers[] = {2,2,2,2};
	int specificLayerSize[][maxLayer] = { {dataset[0].getFeatureNumber(), numH*10, numH},
					       {dataset[1].getFeatureNumber(), numH*10, numH},
					       {dataset[2].getFeatureNumber(), numH*10, numH},
					       {dataset[3].getFeatureNumber(), numH*10, numH}};
	int numEpochs[] ={epochs,epochs,epochs,epochs};
//	string modelFileNames[4] = {savefile1+"msi.dat",savefile1+"cnv.dat",savefile1+"snv.dat",savefile1+"mrna.dat"};
	string modelFileNames[] = {savefile1+"cnv_1077.dat"};
//	string saveFileNames[] = {savefile1+"mrnaDAD.dat"};
	bool gauss[] = {0,0,0,1};
//	bool gauss[] = {1};
	for (int i =0; i<numModels; i++){
		printf("%s\t",modelFileNames[i].c_str());
	}
	printf("%d\n", numH);
	MultiModalRBMs multiModalModel;
	for(int i=0; i<numModels; i++){
		multiModalModel.addModel(new LayerWiseRBMs(specificLayers[i], specificLayerSize[i]));
	}
	multiModalModel.setModelFile(modelFileNames);
	multiModalModel.setGaussVisible(gauss);
	multiModalModel.trainModel(dataset, lrt, 10, numEpochs);
}

int main(int argc, char*argv[]){
	int numModels=1;
	if(argc<8){
		printf("error input!\n");
		exit(0);
	}
	double lr[5];
	for(int i=0; i<5; i++){
		lr[i]=atof(argv[i+1]);
		printf("%f\t",lr[i]);	
	}
	int numHidden=atoi(argv[6]);
	int epochs = atoi(argv[7]);
//	double lr[5] = {0.01, 0.01, 0.01, 0.01, 0.01 };
	multiModalMSI(numModels, lr,  numHidden, epochs, string(argv[8]));
//	multiModalGm_cnv(2);
//	multiModalTCGA_MSI(2);
//	multiModalTCGA(3);
}
