CXX = g++
CFLAGS = -std=c++11 -O2 `pkg-config --cflags opencv`
LIBS = -larmadillo `pkg-config --libs opencv`
OBJS = fmincg.o highgui.o imgproc.o dataset.o

all: neuralnet_train

neuralnet_train: neuralnet_train.o $(OBJS)
	$(CXX) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<

clean:
	rm -f *.o neuralnet_train
