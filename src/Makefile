CC = g++
CFLAGS  = -g -Wall -O2 -std=c++11
OBJECTS = Node.o Math.o HelperFunctions.o MatrixMath.o Optimizers.o CostFunctions.o Activations.o MNISTLoad.o TextLoad.o Regularization.o main.o


all: MLLib

MLLib: $(OBJECTS)
	$(CC) $(CFLAGS) -framework OpenCL -o MLLibCpp $(OBJECTS)

Node.o: Node.cpp Node.hpp openCL.hpp
	$(CC) $(CFLAGS) -c Node.cpp

Math.o: Math.cpp Math.hpp Node.hpp openCL.hpp
	$(CC) $(CFLAGS) -c Math.cpp

HelperFunctions.o: HelperFunctions.cpp HelperFunctions.hpp Node.hpp openCL.hpp
	$(CC) $(CFLAGS) -c HelperFunctions.cpp

MatrixMath.o: MatrixMath.cpp MatrixMath.hpp Node.hpp openCL.hpp
	$(CC) $(CFLAGS) -c MatrixMath.cpp

Optimizers.o: Optimizers.cpp Optimizers.hpp Node.hpp openCL.hpp
	$(CC) $(CFLAGS) -c Optimizers.cpp

CostFunctions.o: CostFunctions.cpp CostFunctions.hpp Node.hpp openCL.hpp
	$(CC) $(CFLAGS) -c CostFunctions.cpp

Activations.o: Activations.cpp Activations.hpp Node.hpp openCL.hpp
	$(CC) $(CFLAGS) -c Activations.cpp

MNISTLoad.o: MNISTLoad.cpp MNISTLoad.hpp Node.hpp
	$(CC) $(CFLAGS) -c MNISTLoad.cpp

TextLoad.o: TextLoad.cpp TextLoad.hpp Node.hpp
	$(CC) $(CFLAGS) -c TextLoad.cpp

Regularization.o: Regularization.cpp Regularization.hpp Node.hpp openCL.hpp
	$(CC) $(CFLAGS) -c Regularization.cpp

main.o: main.cpp Node.hpp Math.hpp HelperFunctions.hpp openCL.hpp MatrixMath.hpp Optimizers.hpp CostFunctions.hpp Activations.hpp MNISTLoad.hpp TextLoad.hpp Regularization.hpp
	$(CC) $(CFLAGS) -c main.cpp

clean:
	$(RM) MLLib $(OBJECTS)