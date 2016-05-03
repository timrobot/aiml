CXX = g++
CFLAGS = -std=c++11 -O2 `pkg-config --cflags opencv`
LIBS = -larmadillo `pkg-config --libs opencv`
OBJS = fmincg.o highgui.o imgproc.o dataset.o

all: perceptron_train logistic_train

perceptron_train: perceptron_train.o $(OBJS)
	$(CXX) $(CFLAGS) -o $@ $^ $(LIBS)

logistic_train: logistic_train.o $(OBJS)
	$(CXX) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<

clean:
	rm -f *.o perceptron_train logistic_train
