CPP=g++

INCLUDE=-Iinclude -I../include -I/opt/intel/mkl/include
CPPFLAGS=-c -g -pg -fopenmp -Wall ${INCLUDE}

MKLLDFLAGS_64=-L/opt/intel/mkl/lib/intel64 /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a \
           -Wl,--start-group \
           /opt/intel/mkl/lib/intel64/libmkl_intel_thread.a \
           /opt/intel/mkl/lib/intel64/libmkl_core.a \
           -Wl,--end-group \
           -L/opt/intel/compiler/lib/intel64 \
           -liomp5 -lpthread -ldl -lm
LDFLAGS=-lm -pg -lgfortran
LDFLAGS:=$(LDFLAGS) $(MKLLDFLAGS_64) #指定库文件的位置


OBJECTS= Dataset.o Layer.o Logistic.o Utility.o IModelComponent.o TrainModel.o MLP.o RBM.o LayerWiseRBMs.o EncoderLayer.o DeepAutoEncoder.o MultiModalComponent.o MultiModal.o
OBJECTS:=$(patsubst %.o, src/%.o,$(OBJECTS))

MODELS=LogisticModel MLPModel RBMModel DBNModel DeepAutoEncoderModel MultiModalModel

test:
	@echo $(CFLAGS)
$(MODELS): % : src/%.o $(OBJECTS)
	$(CPP) $^ $(LDFLAGS) -o $@
$(OBJECTS): %.o : %.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<
clean:
	rm -rf $(MODELS) $(OBJECTS) *.png *.txt src/*.o
	find . -name "*swp" | xargs rm -rf
