
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

void kernel matMul2x2(constant const int* ADim, global const float* A, constant const int* BDim, global const float* B, global float* C, const int blocksWide){
	local float ALocal [1024];
	local float BLocal [1024];

	const int blockSide = 32;
	const int workPerBlockSide = 4;
	const int workPerBlockSideSquared = 16;
	const int threadsPerSide = 8;
	const int threadsPerSideSquared = 64;
	const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int aSize1 = ADim[2];
	const int aSize2 = ADim[3];
	const int bSize2 = BDim[3];
	int split = aSize2 / blockSide;
	if(split * blockSide != aSize2){
		split += 1;
	}

	float aReg = 0.0;
	float bReg[4] = {};
	float total[16] = {};

	for (int i = 0; i < split; i++){
		for (int x = 0; x < workPerBlockSideSquared; x++){
			int pt1 = (localId / threadsPerSide) * workPerBlockSide + (x / workPerBlockSide);
			int pt2 = (localId % threadsPerSide) * workPerBlockSide + (x % workPerBlockSide);
			int newLocalId = pt1 * blockSide + pt2;
			int aRow = (globalId / (threadsPerSideSquared * blocksWide)) * blockSide + pt1;
			int aCol = i * blockSide + pt2;
			if (aRow < aSize1 && aCol < aSize2){
				ALocal[newLocalId] = A[aRow * aSize2 + aCol];
			}
			else{
				ALocal[newLocalId] = 0.0;
			}

			int bRow = i * blockSide + pt1;
			int bCol = ((globalId / threadsPerSideSquared) % blocksWide) * blockSide + pt2;
			if (bRow < aSize2 && bCol < bSize2){
				BLocal[newLocalId] = B[bRow * bSize2 +  bCol];
			}
			else{
				BLocal[newLocalId] = 0.0;
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int x = 0; x < blockSide; x++){
			for (int z = 0; z < workPerBlockSide; z++){
				bReg[z] = BLocal[x * blockSide + (localId % threadsPerSide) * workPerBlockSide + z];
			}

			for (int n = 0; n < workPerBlockSide; n++){
				aReg = ALocal[((localId / threadsPerSide) * workPerBlockSide + n) * blockSide + x];
				for (int m = 0; m < workPerBlockSide; m++){
					total[n * workPerBlockSide + m] += aReg * bReg[m];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}


	for(int x = 0; x < workPerBlockSideSquared; x++){
		int aRow = (globalId / (threadsPerSideSquared * blocksWide)) * blockSide + (localId / threadsPerSide) * workPerBlockSide + (x / workPerBlockSide);
		int bCol = ((globalId / threadsPerSideSquared) % blocksWide) * blockSide + (localId % threadsPerSide) * workPerBlockSide + (x % workPerBlockSide);
		if (aRow < aSize1 && bCol < bSize2){
			C[aRow * bSize2 + bCol] = total[x];
		}
	}
}

void kernel matMul2x1(constant const int* ADim, global const float* A, global const float* B, global float* C){
	local float BLocal [16];

	const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int localSize = get_local_size(0);
	const int aSize1 = ADim[2];
	const int aSize2 = ADim[3];
	int split = aSize2 / localSize;
	if (split * localSize != aSize2){
		split += 1;
	}

	float total = 0.0;

	for (int i = 0; i < split; i++){
		int bItem = localSize * i + localId;
		if (bItem < aSize2){
			BLocal[localId] = B[bItem];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);

		if (globalId < aSize1){
			if ((i + 1) * localSize - 1 < aSize2){
				for (int x = 0; x < localSize; x++){
					total += A[globalId * aSize2 + i * localSize + x] * BLocal[x];
				}
			}
			else{
				for (int x = 0; x < (aSize2 % localSize); x++){
					total += A[globalId * aSize2 + i * localSize + x] * BLocal[x];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (globalId < aSize1){
		C[globalId] = total;
	}
}

void kernel matMul1x2(global const float* A, constant const int* BDim, global const float* B, global float* C){
	local float ALocal [16];

	const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int localSize = get_local_size(0);
	const int bSize1 = BDim[2];
	const int bSize2 = BDim[3];
	int split = bSize1 / localSize;
	if (split * localSize != bSize1){
		split += 1;
	}

	float total = 0.0;

	for (int i = 0; i < split; i++){
		int aItem = localSize * i + localId;
		if (aItem < bSize1){
			ALocal[localId] = A[aItem];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);

		if (globalId < bSize2){
			if ((i + 1) * localSize - 1 < bSize1){
				for (int x = 0; x < localSize; x++){
					total += ALocal[x] * B[(i * localSize + x) * bSize2 + globalId];
				}
			}
			else{
				for (int x = 0; x < (bSize1 % localSize); x++){
					total += ALocal[x] * B[(i * localSize + x) * bSize2 + globalId];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (globalId < bSize2){
		C[globalId] = total;
	}
}

void kernel matMul1x1(constant const int* ADim, global const float* A, global const float* B, global float* C){
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
	float highVal = -FLT_MAX;
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

void kernel meanSquared(constant const int* hypothesisDims, global const float* hypothesis, global const float* y, global float* differenceMemo, global float* diffSquaredResedue, global float* result, const int dimentionSize){
	local float diffSquared [128];

	const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int localSize = get_local_size(0);
	const int groupId = get_group_id(0);
	const int numGroups = get_num_groups(0);
	const int totalSize = hypothesisDims[0];

	if (globalId < totalSize){
		float diff = hypothesis[globalId] - y[globalId];
		differenceMemo[globalId] = diff;
		diffSquared[localId] = diff * diff;
	}
	else{
		diffSquared[localId] = 0.0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int currentSize = localSize / 2;
	while(currentSize > 0){
		if (localId < currentSize){
			diffSquared[localId] += diffSquared[localId + currentSize];
		}
		currentSize /= 2;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localId == 0){
		diffSquaredResedue[groupId] = diffSquared[0];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (globalId == 0){
		float total = diffSquaredResedue[0];
		for (int i = 1; i < numGroups; i++){
			total += diffSquaredResedue[i];
		}
		result[0] = total / dimentionSize;
	}
}

void kernel crossEntropy(constant const int* hypothesisDims, global const float* hypothesis, global const float* y, global float* crossResultResedue, global float* result, const int dimentionSize){
	local float crossResult [128];

	const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int localSize = get_local_size(0);
	const int groupId = get_group_id(0);
	const int numGroups = get_num_groups(0);
	const int totalSize = hypothesisDims[0];

	if (globalId < totalSize){
		float hypVal = hypothesis[globalId];
		float yVal = y[globalId];
		crossResult[localId] = -(yVal * log(hypVal) + ((float)(1.0) - yVal) * log((float)(1.0) - hypVal));
	}
	else{
		crossResult[localId] = 0.0;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	int currentSize = localSize / 2;
	while(currentSize > 0){
		if (localId < currentSize){
			crossResult[localId] += crossResult[localId + currentSize];
		}
		currentSize /= 2;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localId == 0){
		crossResultResedue[groupId] = crossResult[0];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (globalId == 0){
		float total = crossResultResedue[0];
		for (int i = 1; i < numGroups; i++){
			total += crossResultResedue[i];
		}
		result[0] = total / dimentionSize;
	}
}

void kernel crossEntropySoftmax(global const float* hypothesis, global const float* y, global float* resedue, global float* softmaxMemo, global float* result, const int dimentionSize, const int meanDimentionSize, const int preSum, const int blocksWide){
	local float localData [128];
	local float localResedue;

	const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int localSize = get_local_size(0);
	const int groupId = get_group_id(0);
	const int numGroups = get_num_groups(0);

	float maxVal;
	float hypothesisVal;

	const int postIdx = groupId / (blocksWide * preSum);
	const int dimIdx = (groupId % blocksWide) * localSize + localId;
	const int preIdx = (groupId / blocksWide) % preSum;
	const int finalIdx = postIdx * (preSum * dimentionSize) + dimIdx * preSum + preIdx;

	if (dimIdx < dimentionSize){
		hypothesisVal = hypothesis[finalIdx];
		localData[localId] = hypothesisVal;
	}
	else{
		hypothesisVal = 0.0;
		localData[localId] = -FLT_MAX;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	int currentSize = localSize / 2;
	while(currentSize > 0){
		if (localId < currentSize){
			localData[localId] = max(localData[localId], localData[localId + currentSize]);
		}
		currentSize /= 2;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localId == 0){
		resedue[groupId] = localData[0];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (dimIdx == 0){
		float dimMaxVal = resedue[groupId];
		for (int i = 1; i < blocksWide; i++){
			dimMaxVal = max(dimMaxVal, resedue[groupId + i]);
		}
		resedue[groupId] = dimMaxVal;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (localId == 0){
		localResedue = resedue[(groupId / blocksWide) * blocksWide];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	maxVal = localResedue;

	if (dimIdx < dimentionSize){
		localData[localId] = exp(hypothesisVal - maxVal);
	}
	else{
		localData[localId] = 0.0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	currentSize = localSize / 2;
	while(currentSize > 0){
		if (localId < currentSize){
			localData[localId] += localData[localId + currentSize];
		}
		currentSize /= 2;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localId == 0){
		resedue[groupId] = localData[0];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (dimIdx == 0){
		float dimTotalVal = resedue[groupId];
		for (int i = 1; i < blocksWide; i++){
			dimTotalVal += resedue[groupId + i];
		}
		resedue[groupId] = maxVal + log(dimTotalVal);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (localId == 0){
		localResedue = resedue[(groupId / blocksWide) * blocksWide];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (dimIdx < dimentionSize){
		float diff = hypothesisVal - localResedue;
		softmaxMemo[finalIdx] = exp(diff);
		localData[localId] = diff * y[finalIdx];
	}
	else{
		localData[localId] = 0.0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	currentSize = localSize / 2;
	while(currentSize > 0){
		if (localId < currentSize){
			localData[localId] += localData[localId + currentSize];
		}
		currentSize /= 2;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localId == 0){
		resedue[groupId] = localData[0];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (globalId == 0){
		float total = resedue[0];
		for (int i = 1; i < numGroups; i++){
			total += resedue[i];
		}
		result[0] = -total / meanDimentionSize;
	}
}

void kernel sigmoid(global const float* A, global float* B){
	B[get_global_id(0)] = ((float)(1.0) / ((float)(1.0) + exp(-A[get_global_id(0)])));
}

void kernel reLU(global const float* A, global float* B){
	float AVal = A[get_global_id(0)];
	if (AVal >= 0.0){
		B[get_global_id(0)] = AVal;
	}
	else{
		B[get_global_id(0)] = 0.0;
	}
}

void kernel leakyReLU(global const float* A, global float* B){
	float AVal = A[get_global_id(0)];
	if (AVal >= 0.0){
		B[get_global_id(0)] = AVal;
	}
	else{
		B[get_global_id(0)] = 0.01 * AVal;
	}
}

void kernel gaussian(global const float* A, global float* B){
	float AVal = A[get_global_id(0)];
	B[get_global_id(0)] = exp(-AVal * AVal);
}

void kernel tanH(global const float* A, global float* B){
	B[get_global_id(0)] = ((float)(2.0) / ((float)(1.0) + exp((float)(-2.0) * A[get_global_id(0)]))) - (float)(1.0);
}

void kernel softsign(global const float* A, global float* B){
	float AVal = A[get_global_id(0)];
	B[get_global_id(0)] = AVal / (1 + fabs(AVal));
}

void kernel softmax(global const float* A, global float* resedue, global float* B, const int dimentionSize, const int preSum, const int blocksWide){
	local float localData [128];
	local float localResedue;

	const int localId = get_local_id(0);
	const int localSize = get_local_size(0);
	const int groupId = get_group_id(0);

	float maxVal;
	float AVal;

	const int postIdx = groupId / (blocksWide * preSum);
	const int dimIdx = (groupId % blocksWide) * localSize + localId;
	const int preIdx = (groupId / blocksWide) % preSum;
	const int finalIdx = postIdx * (preSum * dimentionSize) + dimIdx * preSum + preIdx;

	if (dimIdx < dimentionSize){
		AVal = A[finalIdx];
		localData[localId] = AVal;
	}
	else{
		AVal = 0.0;
		localData[localId] = -FLT_MAX;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int currentSize = localSize / 2;
	while(currentSize > 0){
		if (localId < currentSize){
			localData[localId] = max(localData[localId], localData[localId + currentSize]);
		}
		currentSize /= 2;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localId == 0){
		resedue[groupId] = localData[0];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (dimIdx == 0){
		float dimMaxVal = resedue[groupId];
		for (int i = 1; i < blocksWide; i++){
			dimMaxVal = max(dimMaxVal, resedue[groupId + i]);
		}
		resedue[groupId] = dimMaxVal;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (localId == 0){
		localResedue = resedue[(groupId / blocksWide) * blocksWide];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	maxVal = localResedue;

	if (dimIdx < dimentionSize){
		localData[localId] = exp(AVal - maxVal);
	}
	else{
		localData[localId] = 0.0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	currentSize = localSize / 2;
	while(currentSize > 0){
		if (localId < currentSize){
			localData[localId] += localData[localId + currentSize];
		}
		currentSize /= 2;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localId == 0){
		resedue[groupId] = localData[0];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (dimIdx == 0){
		float dimTotalVal = resedue[groupId];
		for (int i = 1; i < blocksWide; i++){
			dimTotalVal += resedue[groupId + i];
		}
		resedue[groupId] = maxVal + log(dimTotalVal);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (localId == 0){
		localResedue = resedue[(groupId / blocksWide) * blocksWide];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (dimIdx < dimentionSize){
		B[finalIdx] = exp(AVal - localResedue);
	}
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

void kernel logDerivative(global const float* A, const float baseLN, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = (1.0 / (baseLN * A[get_global_id(0)])) * seed[get_global_id(0) % seedDim[0]];
}

void kernel sinDerivative(global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = cos(A[get_global_id(0)]) * seed[get_global_id(0) % seedDim[0]];
}

void kernel cosDerivative(global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = -sin(A[get_global_id(0)]) * seed[get_global_id(0) % seedDim[0]];
}

void kernel tanDerivative(global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = pow((float)(1.0 / cos(A[get_global_id(0)])), (float)(2.0)) * seed[get_global_id(0) % seedDim[0]];
}

void kernel asinDerivative(global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = (1.0 / sqrt((float)1.0 - pow(A[get_global_id(0)], (float)2.0))) * seed[get_global_id(0) % seedDim[0]];
}

void kernel acosDerivative(global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = (-1.0 / sqrt((float)1.0 - pow(A[get_global_id(0)], (float)2.0))) * seed[get_global_id(0) % seedDim[0]];
}

void kernel atanDerivative(global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = (1.0 / (1.0 + pow(A[get_global_id(0)], (float)2.0))) * seed[get_global_id(0) % seedDim[0]];
}

void kernel matMul2x2Derivative0(constant const int* BDim, global const float* B, constant const int* seedDim, global const float* seed, constant const int* outDim, global float* out, const int blocksWide){
	local float seedLocal [1024];
	local float BLocal [1024];

	const int blockSide = 32;
	const int workPerBlockSide = 4;
	const int workPerBlockSideSquared = 16;
	const int threadsPerSide = 8;
	const int threadsPerSideSquared = 64;
	const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int seedSize1 = outDim[2];
	const int bSize1 = BDim[2];
	const int bSize2 = BDim[3];
	int split = bSize2 / blockSide;
	if(split * blockSide != bSize2){
		split += 1;
	}

	float seedReg = 0.0;
	float bReg[4] = {};
	float total[16] = {};

	for (int i = 0; i < split; i++){
		for (int x = 0; x < workPerBlockSideSquared; x++){
			int pt1 = (localId / threadsPerSide) * workPerBlockSide + (x / workPerBlockSide);
			int pt2 = (localId % threadsPerSide) * workPerBlockSide + (x % workPerBlockSide);
			int newLocalId = pt1 * blockSide + pt2;
			int seedRow = (globalId / (threadsPerSideSquared * blocksWide)) * blockSide + pt1;
			int seedCol = i * blockSide + pt2;
			if (seedRow < seedSize1 && seedCol < bSize2){
				seedLocal[newLocalId] = seed[(seedRow * bSize2 + seedCol) % seedDim[0]];
			}
			else{
				seedLocal[newLocalId] = 0.0;
			}

			int bRow = i * blockSide + pt1;
			int bCol = ((globalId / threadsPerSideSquared) % blocksWide) * blockSide + pt2;
			if (bRow < bSize2 && bCol < bSize1){
				BLocal[newLocalId] = B[bCol * bSize2 + bRow];
			}
			else{
				BLocal[newLocalId] = 0.0;
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int x = 0; x < blockSide; x++){
			for (int z = 0; z < workPerBlockSide; z++){
				bReg[z] = BLocal[x * blockSide + (localId % threadsPerSide) * workPerBlockSide + z];
			}

			for (int n = 0; n < workPerBlockSide; n++){
				seedReg = seedLocal[((localId / threadsPerSide) * workPerBlockSide + n) * blockSide + x];
				for (int m = 0; m < workPerBlockSide; m++){
					total[n * workPerBlockSide + m] += seedReg * bReg[m];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}


	for(int x = 0; x < workPerBlockSideSquared; x++){
		int seedRow = (globalId / (threadsPerSideSquared * blocksWide)) * blockSide + (localId / threadsPerSide) * workPerBlockSide + (x / workPerBlockSide);
		int bCol = ((globalId / threadsPerSideSquared) % blocksWide) * blockSide + (localId % threadsPerSide) * workPerBlockSide + (x % workPerBlockSide);
		if (seedRow < seedSize1 && bCol < bSize1){
			out[seedRow * bSize1 + bCol] = total[x];
		}
	}
}

void kernel matMul2x2Derivative1(constant const int* ADim, global const float* A, constant const int* seedDim, global const float* seed, constant const int* outDim, global float* out, const int blocksWide){
	local float ALocal [1024];
	local float seedLocal [1024];

	const int blockSide = 32;
	const int workPerBlockSide = 4;
	const int workPerBlockSideSquared = 16;
	const int threadsPerSide = 8;
	const int threadsPerSideSquared = 64;
	const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int aSize1 = ADim[2];
	const int aSize2 = ADim[3];
	const int seedSize2 = outDim[3];
	int split = aSize1 / blockSide;
	if(split * blockSide != aSize1){
		split += 1;
	}

	float aReg = 0.0;
	float seedReg[4] = {};
	float total[16] = {};

	for (int i = 0; i < split; i++){
		for (int x = 0; x < workPerBlockSideSquared; x++){
			int pt1 = (localId / threadsPerSide) * workPerBlockSide + (x / workPerBlockSide);
			int pt2 = (localId % threadsPerSide) * workPerBlockSide + (x % workPerBlockSide);
			int newLocalId = pt1 * blockSide + pt2;
			int aRow = (globalId / (threadsPerSideSquared * blocksWide)) * blockSide + pt1;
			int aCol = i * blockSide + pt2;
			if (aRow < aSize2 && aCol < aSize1){
				ALocal[newLocalId] = A[aCol * aSize2 + aRow];
			}
			else{
				ALocal[newLocalId] = 0.0;
			}

			int seedRow = i * blockSide + pt1;
			int seedCol = ((globalId / threadsPerSideSquared) % blocksWide) * blockSide + pt2;
			if (seedRow < aSize1 && seedCol < seedSize2){
				seedLocal[newLocalId] = seed[(seedRow * seedSize2 + seedCol) % seedDim[0]];
			}
			else{
				seedLocal[newLocalId] = 0.0;
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int x = 0; x < blockSide; x++){
			for (int z = 0; z < workPerBlockSide; z++){
				seedReg[z] = seedLocal[x * blockSide + (localId % threadsPerSide) * workPerBlockSide + z];
			}

			for (int n = 0; n < workPerBlockSide; n++){
				aReg = ALocal[((localId / threadsPerSide) * workPerBlockSide + n) * blockSide + x];
				for (int m = 0; m < workPerBlockSide; m++){
					total[n * workPerBlockSide + m] += aReg * seedReg[m];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}


	for(int x = 0; x < workPerBlockSideSquared; x++){
		int aRow = (globalId / (threadsPerSideSquared * blocksWide)) * blockSide + (localId / threadsPerSide) * workPerBlockSide + (x / workPerBlockSide);
		int seedCol = ((globalId / threadsPerSideSquared) % blocksWide) * blockSide + (localId % threadsPerSide) * workPerBlockSide + (x % workPerBlockSide);
		if (aRow < aSize2 && seedCol < seedSize2){
			out[aRow * seedSize2 + seedCol] = total[x];
		}
	}
}

void kernel matMul2x1Derivative0(global const float* B, constant const int* seedDim, global const float* seed, constant const int* outDim, global float* out, const int blocksWide){
	local float BLocal [32];
	local float seedLocal [32];

	const int blockSide = 32;
	const int blockSideSquared = 1024;
	const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int outSize1 = outDim[2];
	const int outSize2 = outDim[3];

	int pt1 = localId / blockSide;
	int pt2 = localId % blockSide;
	int seedItem = (globalId / (blockSideSquared * blocksWide)) * blockSide + pt1;
	if (pt2 == 0 && seedItem < outSize1){
		seedLocal[pt1] = seed[seedItem % seedDim[0]];
	}

	int bItem = ((globalId / blockSideSquared) % blocksWide) * blockSide + pt2;
	if (pt1 == 0 && bItem < outSize2){
		BLocal[pt2] = B[bItem];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (seedItem < outSize1 && bItem < outSize2){
		out[seedItem * outSize2 + bItem] = seedLocal[pt1] * BLocal[pt2];
	}
}

void kernel matMul2x1Derivative1(constant const int* ADim, global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	local float seedLocal [16];

	const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int localSize = get_local_size(0);
	const int aSize1 = ADim[2];
	const int aSize2 = ADim[3];
	int split = aSize1 / localSize;
	if(split * localSize != aSize1){
		split += 1;
	}

	float total = 0.0;

	for (int i = 0; i < split; i++){
		int seedItem = localSize * i + localId;
		if (seedItem < aSize1){
			seedLocal[localId] = seed[seedItem % seedDim[0]];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);

		if (globalId < aSize2){
			if ((i + 1) * localSize - 1 < aSize1){
				for (int x = 0; x < localSize; x++){
					total += A[(i * localSize + x) * aSize2 + globalId] * seedLocal[x];
				}
			}
			else{
				for (int x = 0; x < (aSize1 % localSize); x++){
					total += A[(i * localSize + x) * aSize2 + globalId] * seedLocal[x];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (globalId < aSize2){
		out[globalId] = total;
	}
}


void kernel matMul1x2Derivative0(constant const int* BDim, global const float* B, constant const int* seedDim, global const float* seed, global float* out){
	local float seedLocal [16];

	const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int localSize = get_local_size(0);
	const int bSize1 = BDim[2];
	const int bSize2 = BDim[3];
	int split = bSize2 / localSize;
	if (split * localSize != bSize2){
		split += 1;
	}

	float total = 0.0;

	for (int i = 0; i < split; i++){
		int seedItem = localSize * i + localId;
		if (seedItem < bSize2){
			seedLocal[localId] = seed[seedItem % seedDim[0]];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);

		if (globalId < bSize1){
			if ((i + 1) * localSize - 1 < bSize2){
				for (int x = 0; x < localSize; x++){
					total += B[globalId * bSize2 + i * localSize + x] * seedLocal[x];
				}
			}
			else{
				for (int x = 0; x < (bSize2 % localSize); x++){
					total += B[globalId * bSize2 + i * localSize + x] * seedLocal[x];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (globalId < bSize1){
		out[globalId] = total;
	}
}

void kernel matMul1x2Derivative1(global const float* A, constant const int* seedDim, global const float* seed, constant const int* outDim, global float* out, const int blocksWide){
	local float ALocal [32];
	local float seedLocal [32];

	const int blockSide = 32;
	const int blockSideSquared = 1024;
	const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int outSize1 = outDim[2];
	const int outSize2 = outDim[3];

	int pt1 = localId / blockSide;
	int pt2 = localId % blockSide;
	int seedItem = ((globalId / blockSideSquared) % blocksWide) * blockSide + pt2;
	if (pt1 == 0 && seedItem < outSize2){
		seedLocal[pt2] = seed[seedItem % seedDim[0]];
	}

	int aItem = (globalId / (blockSideSquared * blocksWide)) * blockSide + pt1;
	if (pt2 == 0 && aItem < outSize1){
		ALocal[pt1] = A[aItem];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (seedItem < outSize2 && aItem < outSize1){
		out[aItem * outSize2 + seedItem] = ALocal[pt1] * seedLocal[pt2];
	}
}

void kernel matMul1x1Derivative0(global const float* B, constant const float* seed, global float* out){
	out[get_global_id(0)] = B[get_global_id(0)] * seed[0];
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

void kernel meanSquaredDerivative(constant const float* seed, global const float* differenceMemo, global float* out, const int dimentionSize){
	out[get_global_id(0)] = seed[0] * differenceMemo[get_global_id(0)] * (2.0 / ((float)dimentionSize));
}

void kernel crossEntropyDerivative(constant const float* seed, global const float* hypothesis, global const float* y, global float* out, const int dimentionSize){
	float hypVal = hypothesis[get_global_id(0)];
	out[get_global_id(0)] = seed[0] * ((float)(-1.0) / ((float)dimentionSize)) * ((y[get_global_id(0)] - hypVal) / (hypVal * ((float)(1.0) - hypVal)));
}

void kernel crossEntropySoftmaxDerivative(constant const float* seed, global const float* softmaxMemo, global const float* y, global float* out, const int meanDimentionSize){
	out[get_global_id(0)] = seed[0] * ((float)(1.0) / ((float)meanDimentionSize)) * (softmaxMemo[get_global_id(0)] - y[get_global_id(0)]);
}

void kernel sigmoidDerivative(global const float* C, constant const int* seedDim, global const float* seed, global float* out){
	float CVal = C[get_global_id(0)];
	out[get_global_id(0)] = (CVal * ((float)(1.0) - CVal)) * seed[get_global_id(0) % seedDim[0]];
}

void kernel reLUDerivative(global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	float AVal = A[get_global_id(0)];
	if (AVal >= 0.0){
		out[get_global_id(0)] = seed[get_global_id(0) % seedDim[0]];
	}
	else{
		out[get_global_id(0)] = 0.0;
	}
}

void kernel leakyReLUDerivative(global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	float AVal = A[get_global_id(0)];
	if (AVal >= 0.0){
		out[get_global_id(0)] = seed[get_global_id(0) % seedDim[0]];
	}
	else{
		out[get_global_id(0)] = 0.01 * seed[get_global_id(0) % seedDim[0]];
	}
}

void kernel gaussianDerivative(global const float* A, global const float* C, constant const int* seedDim, global const float* seed, global float* out){
	out[get_global_id(0)] = -2.0 * A[get_global_id(0)] * C[get_global_id(0)] * seed[get_global_id(0) % seedDim[0]];
}

void kernel tanHDerivative(global const float* C, constant const int* seedDim, global const float* seed, global float* out){
	float CVal = C[get_global_id(0)];
	out[get_global_id(0)] = (1.0 - CVal * CVal) * seed[get_global_id(0) % seedDim[0]];
}

void kernel softsignDerivative(global const float* A, constant const int* seedDim, global const float* seed, global float* out){
	int pt1 = 1.0 + fabs(A[get_global_id(0)]);
	out[get_global_id(0)] = (1.0 / (pt1 * pt1)) * seed[get_global_id(0) % seedDim[0]];
}

void kernel softmaxDerivative(global const float* C, constant const int* seedDim, global const float* seed, global float* resedue, global float* out, const int dimentionSize, const int preSum, const int blocksWide){
	local float localData [128];
	local float localResedue;

	const int localId = get_local_id(0);
	const int localSize = get_local_size(0);
	const int groupId = get_group_id(0);

	float CVal;
	float seedVal;

	const int postIdx = groupId / (blocksWide * preSum);
	const int dimIdx = (groupId % blocksWide) * localSize + localId;
	const int preIdx = (groupId / blocksWide) % preSum;
	const int finalIdx = postIdx * (preSum * dimentionSize) + dimIdx * preSum + preIdx;

	if (dimIdx < dimentionSize){
		CVal = C[finalIdx];
		seedVal = seed[finalIdx % seedDim[0]];
	}
	else{
		CVal = 0.0;
		seedVal = 0.0;
	}

	localData[localId] = CVal * seedVal;

	barrier(CLK_LOCAL_MEM_FENCE);

	int currentSize = localSize / 2;
	while(currentSize > 0){
		if (localId < currentSize){
			localData[localId] += localData[localId + currentSize];
		}
		currentSize /= 2;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localId == 0){
		resedue[groupId] = localData[0];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (dimIdx == 0){
		float dimTotalVal = resedue[groupId];
		for (int i = 1; i < blocksWide; i++){
			dimTotalVal += resedue[groupId + i];
		}
		resedue[groupId] = dimTotalVal;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (localId == 0){
		localResedue = resedue[(groupId / blocksWide) * blocksWide];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (dimIdx < dimentionSize){
		out[finalIdx] = CVal * (seedVal - localResedue);
	}
}













