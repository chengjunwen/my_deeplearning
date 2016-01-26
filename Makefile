CPP=g++

INCLUDE=-Iinclude -I../include -I/opt/intel/composer_xe_2013.5.192/mkl/include
CPPFLAGS=-c -g -pg -fopenmp -Wall ${INCLUDE}

MKLLDFLAGS_64=-L/opt/intel/composer_xe_2013.5.192/mkl/lib/intel64 /opt/intel/composer_xe_2013.5.192/mkl/lib/intel64/libmkl_intel_lp64.a \
           -Wl,--start-group \
           /opt/intel/composer_xe_2013.5.192/mkl/lib/intel64/libmkl_intel_thread.a \
           /opt/intel/composer_xe_2013.5.192/mkl/lib/intel64/libmkl_core.a \
           -Wl,--end-group \
           -L/opt/intel/composer_xe_2013.5.192/compiler/lib/intel64 \
           -liomp5 -lpthread -ldl -lm
LDFLAGS=-lm -pg -lgfortran
LDFLAGS:=$(LDFLAGS) $(MKLLDFLAGS_64) #指定库文件的位置


OBJECTS= Dataset.o Layer.o Logistic.o Utility.o IModel.o TrainModel.o
OBJECTS:=$(patsubst %.o, src/%.o,$(OBJECTS))

MODELS=LogisticModel

test:
	@echo $(CFLAGS)
$(MODELS): % : src/%.o $(OBJECTS)
	$(CPP) $^ $(LDFLAGS) -o $@
$(OBJECTS): %.o : %.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<
