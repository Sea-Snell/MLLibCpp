#define GROUP_SIZE 64


void kernel zeroBuffer(global float* A){
	A[get_global_id(0)] = 0.0;
}

void kernel reduceSum(constant const int* ADim, global const float* A, constant const int* BDim, global float* B){
	float total = 0.0;
	for (int i = 0; i < (ADim[0] / BDim[0]); i++){
		total += A[get_global_id(0) + i * get_global_size(0)];
	}
	B[get_global_id(0)] += total;
}

void kernel explodeUp(constant const int* ADim, global const float* A, constant const int* BDim, global float* B){
	B[get_global_id(0)] += A[get_global_id(0) % ADim[0]];
}

void kernel gradientDescentStep(constant const int* seedDim, global const float* seed, global float* var, const float learningRate){
	var[get_global_id(0)] -= seed[get_global_id(0) % seedDim[0]] * learningRate;
}








void kernel add(constant const int* ADim, global const float* A, constant const int* BDim, global const float* B, global float* C){
	C[get_global_id(0)] = A[get_global_id(0) % ADim[0]] + B[get_global_id(0) % BDim[0]];
}

void kernel subtract(constant const int* ADim, global const float* A, constant const int* BDim, global const float* B, global float* C){
	C[get_global_id(0)] = A[get_global_id(0) % ADim[0]] - B[get_global_id(0) % BDim[0]];
}

void kernel multiply(constant const int* ADim, global const float* A, constant const int* BDim, global const float* B, global float* C){
	C[get_global_id(0)] = A[get_global_id(0) % ADim[0]] * B[get_global_id(0) % BDim[0]];
}

void kernel divide(constant const int* ADim, global const float* A, constant const int* BDim, global const float* B, global float* C){
	C[get_global_id(0)] = A[get_global_id(0) % ADim[0]] / B[get_global_id(0) % BDim[0]];
}

void kernel pow_(constant const int* ADim, global const float* A, constant const int* BDim, global const float* B, global float* C){
	C[get_global_id(0)] = pow(A[get_global_id(0) % ADim[0]], B[get_global_id(0) % BDim[0]]);
}

void kernel ln(global const float* A, global float* B){
	B[get_global_id(0)] = log(A[get_global_id(0)]);
}

void kernel exp_(global const float* A, global float* B){
	B[get_global_id(0)] = exp(A[get_global_id(0)]);
}

void kernel log_(global const float* A, global float* B, const float baseLN){
	B[get_global_id(0)] = log(A[get_global_id(0)]) / baseLN;
}

void kernel sin_(global const float* A, global float* B){
	B[get_global_id(0)] = sin(A[get_global_id(0)]);
}

void kernel cos_(global const float* A, global float* B){
	B[get_global_id(0)] = cos(A[get_global_id(0)]);
}

void kernel tan_(global const float* A, global float* B){
	B[get_global_id(0)] = tan(A[get_global_id(0)]);
}

void kernel asin_(global const float* A, global float* B){
	B[get_global_id(0)] = asin(A[get_global_id(0)]);
}

void kernel acos_(global const float* A, global float* B){
	B[get_global_id(0)] = acos(A[get_global_id(0)]);
}

void kernel atan_(global const float* A, global float* B){
	B[get_global_id(0)] = atan(A[get_global_id(0)]);
}

void kernel matMul2x2(constant const int* ADim, global const float* A, constant const int* BDim, global const float* B, global float* C){
	float total = 0.0;
	for (int i = 0; i < ADim[3]; i++){
		total += A[(get_global_id(0) / BDim[3]) * ADim[3] + i] * B[i * BDim[3] + (get_global_id(0) % BDim[3])];
	}
	C[get_global_id(0)] = total;
}

void kernel matMul2x1(constant const int* ADim, global const float* A, constant const int* BDim, global const float* B, global float* C){
	local float BLocal [GROUP_SIZE];

	int globalId = get_global_id(0);
	int localId = get_local_id(0);
	int localSize = get_local_size(0);
	int bSize = BDim[2];
	int split = bSize / localSize;
	float total = 0.0;

	for (int i = 0; i < split; i++){
		BLocal[localId] = B[localSize * i + localId];
		
		barrier(CLK_LOCAL_MEM_FENCE);

		if (globalId < ADim[2]){
			for (int x = 0; x < localSize; x++){
				total += A[globalId * bSize + i * localSize + x] * BLocal[x];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localSize * split + localId < bSize){
		BLocal[localId] = B[localSize * split + localId];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (globalId < ADim[2]){
		for (int x = 0; x < (bSize % localSize); x++){
			total += A[globalId * bSize + split * localSize + x] * BLocal[x];
		}
		C[globalId] = total;
	}
}

void kernel matMul1x2(constant const int* ADim, global const float* A, constant const int* BDim, global const float* B, global float* C){
	float total = 0.0;
	for (int i = 0; i < ADim[2]; i++){
		total += A[i] * B[i * BDim[3] + get_global_id(0)];
	}
	C[get_global_id(0)] = total;
}

void kernel matMul1x1(constant const int* ADim, global const float* A, constant const int* BDim, global const float* B, global float* C){
	float total = 0.0;
	for (int i = 0; i < ADim[2]; i++){
		total += A[i] * B[i];
	}
	C[get_global_id(0)] = total;
}

void kernel sum_(global const float* A, global float* B, const int dimentionSize, const int preSum){
	float total = 0.0;
	for (int i = 0; i < dimentionSize; i++){
		total += A[(get_global_id(0) / preSum) * preSum * dimentionSize + i * preSum + (get_global_id(0) % preSum)];
	}
	B[get_global_id(0)] = total;
}

void kernel mean_(global const float* A, global float* B, const int dimentionSize, const int preSum){
	float total = 0.0;
	for (int i = 0; i < dimentionSize; i++){
		total += A[(get_global_id(0) / preSum) * preSum * dimentionSize + i * preSum + (get_global_id(0) % preSum)];
	}
	B[get_global_id(0)] = total / ((float)dimentionSize);
}

void kernel trans(constant const int* ADim, global const float* A, global float* B){
	B[get_global_id(0)] = A[(get_global_id(0) % ADim[2]) * ADim[3] + (get_global_id(0) / ADim[2])];
}

void kernel max_(global const float* A, global float* B, global int* Idx, const int dimentionSize, const int preSum){
	float highVal = FLT_MIN;
	float currentVal = 0.0;
	int highIdx = 0;
	for (int i = 0; i < dimentionSize; i++){
		Idx[(get_global_id(0) / preSum) * preSum * dimentionSize + i * preSum + (get_global_id(0) % preSum)] = 0;
		currentVal = A[(get_global_id(0) / preSum) * preSum * dimentionSize + i * preSum + (get_global_id(0) % preSum)];
		if (currentVal > highVal){
			highVal = currentVal;
			highIdx = i;
		}
	}
	Idx[(get_global_id(0) / preSum) * preSum * dimentionSize + highIdx * preSum + (get_global_id(0) % preSum)] = 1;
	B[get_global_id(0)] = highVal;
}

void kernel min_(global const float* A, global float* B, global int* Idx, const int dimentionSize, const int preSum){
	float lowVal = FLT_MAX;
	float currentVal = 0.0;
	int lowIdx = 0;
	for (int i = 0; i < dimentionSize; i++){
		Idx[(get_global_id(0) / preSum) * preSum * dimentionSize + i * preSum + (get_global_id(0) % preSum)] = 0;
		currentVal = A[(get_global_id(0) / preSum) * preSum * dimentionSize + i * preSum + (get_global_id(0) % preSum)];
		if (currentVal < lowVal){
			lowVal = currentVal;
			lowIdx = i;
		}
	}
	Idx[(get_global_id(0) / preSum) * preSum * dimentionSize + lowIdx * preSum + (get_global_id(0) % preSum)] = 1;
	B[get_global_id(0)] = lowVal;
}

void kernel meanSquared(global const float* hypothesis, global const float* y, global float* differenceMemo, global float* diffSquared){
	float diff = hypothesis[get_global_id(0)] - y[get_global_id(0)];
	differenceMemo[get_global_id(0)] = diff;
	diffSquared[get_global_id(0)] = diff * diff;
}









void kernel addDerivative(constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = seed[get_global_id(0) % seedDim[0]];
}

void kernel subtractDerivative1(constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = -1.0 * seed[get_global_id(0) % seedDim[0]];
}

void kernel multiplyDerivative(constant const int* ABDim, global const float* AB, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = AB[get_global_id(0) % ABDim[0]] * seed[get_global_id(0) % seedDim[0]];
}

void kernel divideDerivative0(constant const int* BDim, global const float* B, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = (1.0 / B[get_global_id(0) % BDim[0]]) * seed[get_global_id(0) % seedDim[0]];
}

void kernel divideDerivative1(constant const int* ADim, global const float* A, constant const int* BDim, global const float* B, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = (-A[get_global_id(0) % ADim[0]] / pow(B[get_global_id(0) % BDim[0]], (float)2.0)) * seed[get_global_id(0) % seedDim[0]];
}

void kernel powDerivative0(constant const int* ADim, global const float* A, constant const int* BDim, global const float* B, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = B[get_global_id(0) % BDim[0]] * pow(A[get_global_id(0) % ADim[0]], (float)(B[get_global_id(0) % BDim[0]] - 1.0)) * seed[get_global_id(0) % seedDim[0]];
}

void kernel powDerivative1(constant const int* ADim, global const float* A, constant const int* CDim, global const float* C, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = log(A[get_global_id(0) % ADim[0]]) * C[get_global_id(0) % CDim[0]] * seed[get_global_id(0) % seedDim[0]];
}

void kernel logDerivative(constant const int* ADim, global const float* A, const float baseLN, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = (1.0 / (baseLN * A[get_global_id(0) % ADim[0]])) * seed[get_global_id(0) % seedDim[0]];
}

void kernel sinDerivative(constant const int* ADim, global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = cos(A[get_global_id(0) % ADim[0]]) * seed[get_global_id(0) % seedDim[0]];
}

void kernel cosDerivative(constant const int* ADim, global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = -sin(A[get_global_id(0) % ADim[0]]) * seed[get_global_id(0) % seedDim[0]];
}

void kernel tanDerivative(constant const int* ADim, global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = pow((float)(1.0 / cos(A[get_global_id(0) % ADim[0]])), (float)(2.0)) * seed[get_global_id(0) % seedDim[0]];
}

void kernel asinDerivative(constant const int* ADim, global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = (1.0 / sqrt((float)1.0 - pow(A[get_global_id(0) % ADim[0]], (float)2.0))) * seed[get_global_id(0) % seedDim[0]];
}

void kernel acosDerivative(constant const int* ADim, global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = (-1.0 / sqrt((float)1.0 - pow(A[get_global_id(0) % ADim[0]], (float)2.0))) * seed[get_global_id(0) % seedDim[0]];
}

void kernel atanDerivative(constant const int* ADim, global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = (1.0 / (1.0 + pow(A[get_global_id(0) % ADim[0]], (float)2.0))) * seed[get_global_id(0) % seedDim[0]];
}

void kernel matMul2x2Derivative0(constant const int* BDim, global const float* B, constant const int* seedDim, global const float* seed, global float* out){
	float total = 0.0;
	for (int i = 0; i < BDim[3]; i++){
		total += seed[((get_global_id(0) / BDim[2]) * BDim[3] + i) % seedDim[0]] * B[(get_global_id(0) % BDim[2]) * BDim[3] + i];
	}
	out[get_global_id(0)] = total;
}

void kernel matMul2x2Derivative1(constant const int* ADim, global const float* A, constant const int* seedDim, global const float* seed, constant const int* outDim, global float* out){
	float total = 0.0;
	for (int i = 0; i < ADim[2]; i++){
		total += A[i * ADim[3] + (get_global_id(0) / outDim[3])] * seed[(i * outDim[3] + (get_global_id(0) % outDim[3])) % seedDim[0]];
	}
	out[get_global_id(0)] = total;
}

void kernel matMul2x1Derivative0(constant const int* BDim, global const float* B, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = seed[(get_global_id(0) / BDim[2]) % seedDim[0]] * B[get_global_id(0) % BDim[2]];
}

void kernel matMul2x1Derivative1(constant const int* ADim, global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	local float seedLocal [GROUP_SIZE];

	int globalId = get_global_id(0);
	int ASize2 = ADim[3];
	int localId = get_local_id(0);
	int localSize = get_local_size(0);
	int seedSize = seedDim[2];
	int split = seedSize / localSize;
	float total = 0.0;

	for (int i = 0; i < split; i++){
		seedLocal[localId] = seed[localSize * i + localId];
		
		barrier(CLK_LOCAL_MEM_FENCE);

		if (globalId < ASize2){
			for (int x = 0; x < localSize; x++){
				total += A[(i * localSize + x) * ASize2 + globalId] * seedLocal[x];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localSize * split + localId < seedSize){
		seedLocal[localId] = seed[localSize * split + localId];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (globalId < ASize2){
		for (int x = 0; x < (seedSize % localSize); x++){
			total += A[(split * localSize + x) * ASize2 + globalId] * seedLocal[x];
		}
		out[globalId] = total;
	}
}


void kernel matMul1x2Derivative0(constant const int* BDim, global const float* B, constant const int* seedDim, global const float* seed, global float* out){
	float total = 0.0;
	for (int i = 0; i < BDim[3]; i++){
		total += seed[i % seedDim[0]] * B[get_global_id(0) * BDim[3] + i];
	}
	out[get_global_id(0)] = total;
}

void kernel matMul1x2Derivative1(global const float* A, constant const int* seedDim, global const float* seed, constant const int* outDim, global float* out){
	out[get_global_id(0)] = A[get_global_id(0) / outDim[3]] * seed[(get_global_id(0) % outDim[3]) % seedDim[0]];
}


void kernel matMul1x1Derivative0(global const float* B, global const float* seed, global float* out){
	out[get_global_id(0)] = B[get_global_id(0)] * seed[0];
}

void kernel matMul1x1Derivative1(global const float* A, global const float* seed, global float* out){
	out[get_global_id(0)] = A[get_global_id(0)] * seed[0];
}

void kernel sumDerivative(constant const int* seedDim, global const float* seed, global float* out, const int dimentionSize, const int preSum){
	out[get_global_id(0)] = seed[((get_global_id(0) % preSum) + (get_global_id(0) / (preSum * dimentionSize)) * preSum) % seedDim[0]];
}

void kernel meanDerivative(constant const int* seedDim, global const float* seed, global float* out, const int dimentionSize, const int preSum){
	out[get_global_id(0)] = seed[((get_global_id(0) % preSum) + (get_global_id(0) / (preSum * dimentionSize)) * preSum) % seedDim[0]] / ((float)dimentionSize);
}

void kernel meanDerivativeSmallSeed(global const float* seed, global float* out, const int dimentionSize){
	out[get_global_id(0)] = seed[get_global_id(0)] / ((float)dimentionSize);
}

void kernel transDerivative1(global const float* seed, constant const int* outDim, global float* out){
	out[get_global_id(0)] = seed[get_global_id(0) / outDim[3]];
}

void kernel transDerivative2(constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = seed[(get_global_id(0) % seedDim[2]) * seedDim[3] + (get_global_id(0) / seedDim[2])];
}

void kernel maxDerivative(constant const int* seedDim, global const float* seed, global float* out, global const int* Idx, const int dimentionSize, const int preSum){
	out[get_global_id(0)] = seed[((get_global_id(0) % preSum) + (get_global_id(0) / (preSum * dimentionSize)) * preSum) % seedDim[0]] * Idx[get_global_id(0)];
}

void kernel maxDerivativeSmallSeed(constant const int* seedDim, global const float* seed, global float* out, global const int* Idx){
	out[get_global_id(0)] = seed[get_global_id(0) % seedDim[0]] * Idx[get_global_id(0)];
}

void kernel meanSquaredDerivative(constant const int* seedDim, global const float* seed, global const float* differenceMemo, global float* out, const int dimentionSize, const int preSum){
	out[get_global_id(0)] = seed[((get_global_id(0) % preSum) + (get_global_id(0) / (preSum * dimentionSize)) * preSum) % seedDim[0]] * differenceMemo[get_global_id(0)] * (2.0 / ((float)dimentionSize));
}

void kernel meanSquaredDerivativeSmallSeed(constant const int* seedDim, global const float* seed, global const float* differenceMemo, global float* out, const int dimentionSize){
	out[get_global_id(0)] = seed[get_global_id(0) % seedDim[0]] * differenceMemo[get_global_id(0)] * (2.0 / ((float)dimentionSize));
}



