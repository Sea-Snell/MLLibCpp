CC = g++
CFLAGS  = -g -Wall -O2 -std=c++11
OBJECTS = Node.o Math.o HelperFunctions.o MatrixMath.o Optimizers.o CostFunctions.o Activations.o Regularization.o MNISTLoad.o TextLoad.o main.o


all: MLLib

MLLib: $(OBJECTS)
	$(CC) $(CFLAGS) -o MLLibCpp $(OBJECTS)

Node.o: Node.cpp Node.hpp
	$(CC) $(CFLAGS) -c Node.cpp

Math.o: Math.cpp Math.hpp Node.hpp
	$(CC) $(CFLAGS) -c Math.cpp

HelperFunctions.o: HelperFunctions.cpp HelperFunctions.hpp Node.hpp
	$(CC) $(CFLAGS) -c HelperFunctions.cpp

MatrixMath.o: MatrixMath.cpp MatrixMath.hpp Node.hpp
	$(CC) $(CFLAGS) -c MatrixMath.cpp

Optimizers.o: Optimizers.cpp Optimizers.hpp Node.hpp HelperFunctions.hpp
	$(CC) $(CFLAGS) -c Optimizers.cpp

CostFunctions.o: CostFunctions.cpp CostFunctions.hpp Node.hpp
	$(CC) $(CFLAGS) -c CostFunctions.cpp

Activations.o: Activations.cpp Activations.hpp Node.hpp HelperFunctions.hpp
	$(CC) $(CFLAGS) -c Activations.cpp

Regularization.o: Regularization.cpp Regularization.hpp Node.hpp
	$(CC) $(CFLAGS) -c Regularization.cpp

MNISTLoad.o: MNISTLoad.cpp MNISTLoad.hpp Node.hpp
	$(CC) $(CFLAGS) -c MNISTLoad.cpp

TextLoad.o: TextLoad.cpp TextLoad.hpp Node.hpp
	$(CC) $(CFLAGS) -c TextLoad.cpp

main.o: main.cpp Node.hpp Math.hpp HelperFunctions.hpp MatrixMath.hpp Optimizers.hpp CostFunctions.hpp Activations.hpp Regularization.hpp MNISTLoad.hpp TextLoad.hpp
	$(CC) $(CFLAGS) -c main.cpp

clean:
	$(RM) MLLib $(OBJECTS)