#ifndef MULTIMODAL_H
#define MULTIMODAL_H

#include <cfloat>
#include <cstdio>
#include <string>
#include "Utility.h"
#include "mkl_cblas.h"
#include "LayerWiseRBMs.h"
#include "DeepAutoEncoder.h"
#include "TrainModel.h"

void logisticUp(Dataset * combinDataSet);
void mlpUp(Dataset * combinDataSet, int numHid=128, double lr=0.01);
void DBNUp(Dataset * combinDataSet, int nLayers, int layerSizes[],double lr, int batchsize, int epochs);
void DeepAd(Dataset * combinDataSet, int nLayers, int layerSizes[], double lr, int batchsize, int epochs, string savefile);
void StackRBM(Dataset * combinDataSet, int nLayers, int layerSizes[], double lr, int batchsize, int epochs, string savefile);
#endif
