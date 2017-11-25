CC = g++
CFLAGS  = -g -Wall -O2 -std=c++11
OBJECTS = Node.o Math.o HelperFunctions.o MatrixMath.o Optimizers.o Activations.o CostFunctions.o MNISTLoad.o Regularization.o TextLoad.o Rubix.o main.o


all: MLLib

MLLib: $(OBJECTS)
	$(CC) $(CFLAGS) -o MLLibCpp $(OBJECTS)

Node.o: Node.cpp Node.hpp MapVals.hpp
	$(CC) $(CFLAGS) -c Node.cpp

Math.o: Math.cpp Math.hpp Node.hpp MapVals.hpp
	$(CC) $(CFLAGS) -c Math.cpp

HelperFunctions.o: HelperFunctions.cpp HelperFunctions.hpp Node.hpp
	$(CC) $(CFLAGS) -c HelperFunctions.cpp

MatrixMath.o: MatrixMath.cpp MatrixMath.hpp Node.hpp MapVals.hpp
	$(CC) $(CFLAGS) -c MatrixMath.cpp

Optimizers.o: Optimizers.cpp Optimizers.hpp Node.hpp HelperFunctions.hpp MapVals.hpp
	$(CC) $(CFLAGS) -c Optimizers.cpp

Activations.o: Activations.cpp Activations.hpp Node.hpp MapVals.hpp HelperFunctions.hpp
	$(CC) $(CFLAGS) -c Activations.cpp

CostFunctions.o: CostFunctions.cpp CostFunctions.hpp Node.hpp MapVals.hpp
	$(CC) $(CFLAGS) -c CostFunctions.cpp

MNISTLoad.o: MNISTLoad.cpp MNISTLoad.hpp Node.hpp
	$(CC) $(CFLAGS) -c MNISTLoad.cpp

Regularization.o: Regularization.cpp Regularization.hpp Node.hpp MapVals.hpp
	$(CC) $(CFLAGS) -c Regularization.cpp

TextLoad.o: TextLoad.cpp TextLoad.hpp Node.hpp
	$(CC) $(CFLAGS) -c TextLoad.cpp

Rubix.o: Rubix.cpp Rubix.hpp Node.hpp
	$(CC) $(CFLAGS) -c Rubix.cpp

main.o: main.cpp Node.hpp Math.hpp MapVals.hpp HelperFunctions.hpp MatrixMath.hpp Optimizers.hpp Activations.hpp CostFunctions.hpp Regularization.hpp TextLoad.hpp Rubix.hpp
	$(CC) $(CFLAGS) -c main.cpp

clean:
	$(RM) MLLib $(OBJECTS)