void kernel zeroBuffer(global float* A){
	A[get_global_id(0)] = 0.0;
}

void kernel reduceSum(constant const int* ADim, constant const float* A, constant const int* BDim, global float* B){
	float total = 0.0;
	for (int i = 0; i < (ADim[0] / BDim[0]); i++){
		total += A[get_global_id(0) + i * get_global_size(0)];
	}
	B[get_global_id(0)] += total;
}

void kernel explodeUp(constant const int* ADim, constant const float* A, constant const int* BDim, global float* B){
	B[get_global_id(0)] += A[get_global_id(0) % ADim[0]];
}

void kernel add(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	C[get_global_id(0)] = A[get_global_id(0) % ADim[0]] + B[get_global_id(0) % BDim[0]];
}

void kernel subtract(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	C[get_global_id(0)] = A[get_global_id(0) % ADim[0]] - B[get_global_id(0) % BDim[0]];
}

void kernel multiply(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	C[get_global_id(0)] = A[get_global_id(0) % ADim[0]] * B[get_global_id(0) % BDim[0]];
}

void kernel divide(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	C[get_global_id(0)] = A[get_global_id(0) % ADim[0]] / B[get_global_id(0) % BDim[0]];
}

void kernel pow_(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	C[get_global_id(0)] = pow(A[get_global_id(0) % ADim[0]], B[get_global_id(0) % BDim[0]]);
}

void kernel ln(constant const float* A, global float* B){
	B[get_global_id(0)] = log(A[get_global_id(0)]);
}

void kernel exp_(constant const float* A, global float* B){
	B[get_global_id(0)] = exp(A[get_global_id(0)]);
}

void kernel log_(constant const float* A, global float* B, float baseLN){
	B[get_global_id(0)] = log(A[get_global_id(0)]) / baseLN;
}

void kernel addDerivative(constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = seed[get_global_id(0) % seedDim[0]];
}

void kernel subtractDerivative1(constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = -1.0 * seed[get_global_id(0) % seedDim[0]];
}

void kernel multiplyDerivative(constant const int* ABDim, constant const float* AB, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = AB[get_global_id(0) % ABDim[0]] * seed[get_global_id(0) % seedDim[0]];
}

void kernel divideDerivative0(constant const int* BDim, constant const float* B, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = (1.0 / B[get_global_id(0) % BDim[0]]) * seed[get_global_id(0) % seedDim[0]];
}

void kernel divideDerivative1(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = (-A[get_global_id(0) % ADim[0]] / pow(B[get_global_id(0) % BDim[0]], (float)2.0)) * seed[get_global_id(0) % seedDim[0]];
}

void kernel powDerivative0(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = B[get_global_id(0) % BDim[0]] * pow(A[get_global_id(0) % ADim[0]], (float)(B[get_global_id(0) % BDim[0]] - 1.0)) * seed[get_global_id(0) % seedDim[0]];
}

void kernel powDerivative1(constant const int* ADim, constant const float* A, constant const int* CDim, constant const float* C, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = log(A[get_global_id(0) % ADim[0]]) * C[get_global_id(0) % CDim[0]] * seed[get_global_id(0) % seedDim[0]];
}





