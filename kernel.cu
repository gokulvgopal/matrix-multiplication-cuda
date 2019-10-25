
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include<string>
#include <stdio.h>
#include <time.h>
#include<cuda.h>
#include <math.h>
#include "cublas_v2.h"


using std::ifstream;
using std::ofstream;
using namespace std;
#define _USE_MATH_DEFINES
#define TILE_WIDTH 32

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << "in" << file << " at line " << line;
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ));

//structure to store matrix values, row and column
struct Matrix
{
	int row;
	int col;
	float* val;
};

//function declaration
void printMatrix(Matrix* mat);
Matrix* changeMatrixColWise(Matrix* mat);
Matrix* changeMatrixRowWise(Matrix* mat);
//
//
//
//
//
//

// GPU function to find matrix multiplication using shared memory
// matrix A of size rowA x colA
// matrix B of size colA x colB
// matrix C of size rowA x colB
__global__  void gpuMultiplicationSharedMem(float* matrixC, float* matrixA, float* matrixB, int rowA, int colA, int colB) {
	float sum = 0;
	int blockx = blockIdx.x, blocky = blockIdx.y;
	int threadx = threadIdx.x, thready = threadIdx.y;
	int i = blocky * TILE_WIDTH + thready;
	int j = blockx * TILE_WIDTH + threadx;
	__shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

	for (int k = 0; k < (colA - 1) / TILE_WIDTH + 1; ++k) {
		if (i < rowA && k*TILE_WIDTH + threadx < colA) {
			sharedA[thready][threadx] = matrixA[i*colA + k * TILE_WIDTH + threadx];
		}
		else {
			sharedA[thready][threadx] = 0;
		}

		if (j < colB && k*TILE_WIDTH + thready < colA) {
			sharedB[thready][threadx] = matrixB[(k*TILE_WIDTH + thready)*colB + j];
		}
		else {
			sharedB[thready][threadx] = 0;
		}

		__syncthreads();

		for (int m = 0; m < TILE_WIDTH; ++m) {
			sum += sharedA[thready][m] * sharedB[m][threadx];
		}
		__syncthreads();
	}
	if (i < rowA && j < colB) {
		matrixC[i*colB + j] = sum;
	}
}

// GPU function to find matrix multiplication using cublas library
// matrix A of size rowA x colA
// matrix B of size colA x colB
// matrix C of size rowA x colB
void gpuMultiplicationUsingCublas(const float* matrixA, const float* matrixB, float* matrixC, int rowA, int colA, int colB) {
	int lda = rowA, ldb = colA, ldc = rowA;
	const float alf = 1.0;
	const float bet = 0.0;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rowA, colB, colA, alpha, matrixA, lda, matrixB, ldb, beta, matrixC, ldc);

	cublasDestroy(handle);
}

// GPU function to find matrix multiplication 
// matrix A of size rowA x colA
// matrix B of size colA x colB
// matrix C of size rowA x colB
__global__  void gpuMultiplication(float* matrixC, float* matrixA, float* matrixB, int rowA, int colA, int colB) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= rowA || j >= colB)
		return;

	float sum = 0;
	for (int k = 0; k < colA; k++) {
		sum += matrixA[i*colA + k] * matrixB[k*colB + j];
	}
	matrixC[i * colB + j] = sum;

}

// CPU function to invoke GPU function that do matrix multiplication using share memory
// matrix A of size rowA x colA
// matrix B of size colA x colB
// matrix C of size rowA x colB
Matrix* invokeGpuMultiplicationSharedMem(Matrix* matrixA, Matrix* matrixB) {
	clock_t start_timer = clock();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	Matrix* matrixC = (Matrix*)malloc(sizeof(Matrix));
	matrixC->row = matrixA->row;
	matrixC->col = matrixB->col;
	cout << "size of C: " << matrixC->row << " x " << matrixC->col << endl;
	matrixC->val = (float*)malloc(matrixA->row * matrixB->col * sizeof(float));
	

	float* gpuValA;
	float* gpuValB;
	float* gpuValC;
	if (matrixA->col != matrixB->row) {
		cout << "returning null";
		return NULL;
	}
	else {
		HANDLE_ERROR(cudaMalloc(&gpuValA, matrixA->row * matrixA->col * sizeof(float)));
		HANDLE_ERROR(cudaMalloc(&gpuValB, matrixB->row * matrixB->col * sizeof(float)));
		HANDLE_ERROR(cudaMalloc(&gpuValC, matrixC->row * matrixC->col * sizeof(float)));


		HANDLE_ERROR(cudaMemcpy(gpuValA, matrixA->val, matrixA->row * matrixA->col * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpuValB, matrixB->val, matrixB->row * matrixB->col * sizeof(float), cudaMemcpyHostToDevice));

		cudaDeviceProp prop;
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
		/*dim3 threads(sqrt(prop.maxThreadsPerBlock), sqrt(prop.maxThreadsPerBlock));
		dim3 blocks((matrixA->row/threads.x),(matrixB->col/threads.y));*/
		dim3 threads(32, 32);
		dim3 blocks((matrixA->row / threads.x) + 1, (matrixB->col / threads.y) + 1);
		//printf((matrixA->row / threads.x) + 1);
		cudaEventRecord(start);
		gpuMultiplicationSharedMem <<< blocks, threads >>> (gpuValC, gpuValA, gpuValB, matrixA->row, matrixA->col, matrixB->col);
		cudaEventRecord(stop);
		HANDLE_ERROR(cudaMemcpy(matrixC->val, gpuValC, matrixC->row * matrixC->col * sizeof(float), cudaMemcpyDeviceToHost));

		HANDLE_ERROR(cudaFree(gpuValA));
		HANDLE_ERROR(cudaFree(gpuValB));
		HANDLE_ERROR(cudaFree(gpuValC));
	}

	clock_t end_timer = clock();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	/*profile convolution time here*/
	double elapsed_time = (end_timer - start_timer) / (double)CLOCKS_PER_SEC;
	cout << endl << endl << "     ******************     GPU Shared Mem PROFILING RESULTS      ******************      " << endl;
	printf("Convolution time on GPU shared mem is %lf fractional seconds\n", elapsed_time);
	cout << "Convolution time on GPU shared mem is fractional seconds: " << elapsed_time << endl;
	cout << "Time taken by kernel: " << milliseconds << endl;
	cout << "FLOPS - GPU shared mem: " << 2*matrixA->row*matrixA->col*matrixA->col/(milliseconds*1000) << endl;
	cout << "B/S - GPU shared mem: " << 2 * 4 * matrixA->row*matrixA->col*matrixA->col / (milliseconds * 1000) << endl;
	return matrixC;
}

// CPU function to invoke GPU function that do matrix multiplication using cublas
// matrix A of size rowA x colA
// matrix B of size colA x colB
// matrix C of size rowA x colB
Matrix* invokeGpuMultiplicationCublas(Matrix* matrixA, Matrix* matrixB) {
	//cout << "it is alteast calling it";
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	clock_t start_timer = clock();
	cout << " calling cublas" << endl;
	Matrix* matrixC = (Matrix*)malloc(sizeof(Matrix));
	matrixC->row = matrixA->row;
	matrixC->col = matrixB->col;
	cout << "size of C: " << matrixC->row << " x " << matrixC->col;
	matrixC->val = (float*)malloc(matrixC->row * matrixC->col * sizeof(float));

	//cout << "Matrix A" << endl;
	//printMatrix(matrixA);
	//cout << "Col wise matrix A" << endl;
	Matrix* matrixAColWise = (Matrix*)malloc(sizeof(Matrix));
	matrixAColWise = changeMatrixColWise(matrixA);
	//printMatrix(matrixAColWise);
	//cout << "Matrix B" << endl;
	//printMatrix(matrixB);
	//cout << " matrix B col wise" << endl;
	Matrix* matrixBColWise = (Matrix*)malloc(sizeof(Matrix));
	matrixBColWise = changeMatrixColWise(matrixB);
	//printMatrix(matrixBColWise);

	float* gpuValA;
	float* gpuValB;
	float* gpuValC;
	if (matrixA->col != matrixB->row) {
		cout << "returning null";
		return NULL;
	}
	else {
		HANDLE_ERROR(cudaMalloc(&gpuValA, matrixA->row * matrixA->col * sizeof(float)));
		HANDLE_ERROR(cudaMalloc(&gpuValB, matrixB->row * matrixB->col * sizeof(float)));
		HANDLE_ERROR(cudaMalloc(&gpuValC, matrixC->row * matrixC->col * sizeof(float)));

		HANDLE_ERROR(cudaMemcpy(gpuValA, matrixAColWise->val, matrixA->row * matrixA->col * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpuValB, matrixBColWise->val, matrixB->row * matrixB->col * sizeof(float), cudaMemcpyHostToDevice));
		cudaEventRecord(start);
		gpuMultiplicationUsingCublas(gpuValA, gpuValB, gpuValC, matrixA->row, matrixA->col, matrixB->col);
		cudaEventRecord(stop);
		HANDLE_ERROR(cudaMemcpy(matrixC->val, gpuValC, matrixC->row * matrixC->col * sizeof(float), cudaMemcpyDeviceToHost));

		cudaFree(gpuValA);
		cudaFree(gpuValB);
		cudaFree(gpuValC);
	}
	// converting a row major matrix to a column major matrix
	Matrix* matrixCRowWise = (Matrix*)malloc(sizeof(Matrix));
	matrixCRowWise = changeMatrixRowWise(matrixC);
	matrixCRowWise = changeMatrixRowWise(matrixCRowWise);
	matrixCRowWise = changeMatrixRowWise(matrixCRowWise);
	clock_t end_timer = clock();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	/*profile convolution time here*/
	double elapsed_time = (end_timer - start_timer) / (double)CLOCKS_PER_SEC;
	cout << endl << endl << "     ******************     GPU CUBLAS PROFILING RESULTS      ******************      " << endl;
	printf("Convolution time on GPU Cublas is %lf fractional seconds\n", elapsed_time);
	cout << "Convolution time on GPU Cublas is fractional seconds: " << elapsed_time << endl;
	cout << "Time taken by kernel: " << milliseconds << endl;
	cout << "FLOPS - GPU Cublas kernel: " << 2 * matrixA->row*matrixA->col*matrixA->col / (milliseconds * 1000) << endl;
	cout << "B/S - GPU Cublas kernel: " << 2 * 4 * matrixA->row*matrixA->col*matrixA->col / (milliseconds * 1000) << endl;
	
	return matrixCRowWise;
}

// CPU function to invoke GPU function that do matrix multiplication
// matrix A of size rowA x colA
// matrix B of size colA x colB
// matrix C of size rowA x colB
Matrix* invokeGpuMultiplication(Matrix* matrixA, Matrix* matrixB) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	clock_t start_timer = clock();

	Matrix* matrixC = (Matrix*)malloc(sizeof(Matrix));
	matrixC->row = matrixA->row;
	matrixC->col = matrixB->col;
	cout << "size of C: " << matrixC->row << " x " << matrixC->col << endl;
	matrixC->val = (float*)malloc(matrixA->row * matrixB->col * sizeof(float));

	float* gpuValA;
	float* gpuValB;
	float* gpuValC;
	if (matrixA->col != matrixB->row) {
		cout << "returning null";
		return NULL;
	}
	else {
		HANDLE_ERROR(cudaMalloc(&gpuValA, matrixA->row * matrixA->col * sizeof(float)));
		HANDLE_ERROR(cudaMalloc(&gpuValB, matrixB->row * matrixB->col * sizeof(float)));
		HANDLE_ERROR(cudaMalloc(&gpuValC, matrixC->row * matrixC->col * sizeof(float)));

		HANDLE_ERROR(cudaMemcpy(gpuValA, matrixA->val, matrixA->row * matrixA->col * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpuValB, matrixB->val, matrixB->row * matrixB->col * sizeof(float), cudaMemcpyHostToDevice));

		cudaDeviceProp prop;
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
		/*dim3 threads(sqrt(prop.maxThreadsPerBlock), sqrt(prop.maxThreadsPerBlock));
		dim3 blocks((matrixA->row/threads.x),(matrixB->col/threads.y));*/
		dim3 threads(32, 32);
		dim3 blocks((matrixC->row / threads.x) + 1, (matrixC->col / threads.y) + 1);

		//printf((matrixA->row / threads.x) + 1);
		cudaEventRecord(start);
		gpuMultiplication <<< blocks, threads >>> (gpuValC, gpuValA, gpuValB, matrixA->row, matrixA->col, matrixB->col);
		cudaEventRecord(stop);
		HANDLE_ERROR(cudaMemcpy(matrixC->val, gpuValC, matrixA->row * matrixB->col * sizeof(float), cudaMemcpyDeviceToHost));
	
		HANDLE_ERROR(cudaFree(gpuValA));
		HANDLE_ERROR(cudaFree(gpuValB));
		HANDLE_ERROR(cudaFree(gpuValC));
	}
	clock_t end_timer = clock();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
//exit(1);
	/*profile convolution time here*/
	double elapsed_time = (end_timer - start_timer) / (double)CLOCKS_PER_SEC;
	cout << endl << endl << "     ******************     GPU PROFILING RESULTS      ******************      " << endl;
	printf("Convolution time on GPU is %lf fractional seconds\n", elapsed_time);
	cout << "Convolution time on GPU is fractional seconds: " << elapsed_time << endl;
	cout << "Time taken by kernel: " << milliseconds << endl;
	cout << "FLOPS - GPU kernel: " << 2 * matrixA->row*matrixA->col*matrixA->col / (milliseconds * 1000) << endl;
	cout << "B/S - GPU kernel: " << 2 * 4 * matrixA->row*matrixA->col*matrixA->col / (milliseconds * 1000) << endl;
	return matrixC;
}

// CPU function to convert column-wise read input to row-wise read matrix
Matrix* changeMatrixColWise(Matrix* mat) {
	Matrix* colMat = (Matrix*)malloc(sizeof(Matrix));
	colMat->col = mat->col;
	colMat->row = mat->row;
	colMat->val = (float*)malloc(colMat->col * colMat->row * sizeof(float));
	int k = 0;
	for (int i = 0; i < mat->col; i++) {
		for (int j = 0; j < mat->row; j++) {
			colMat->val[k] = mat->val[j*mat->col + i];
			k++;
		}
	}
	cout << endl;
	return colMat;
}

//CPU Function to convert row-wise created output to col-wise output
Matrix* changeMatrixRowWise(Matrix* mat) {
	Matrix* rowMat = (Matrix*)malloc(sizeof(Matrix));
	rowMat->col = mat->col;
	rowMat->row = mat->row;
	rowMat->val = (float*)malloc(rowMat->col * rowMat->row * sizeof(float));
	int k = 0;
	for (int i = 0; i < mat->col; i++) {
		for (int j = 0; j < mat->row; j++) {
			rowMat->val[k] = mat->val[j*mat->col + i];
			k++;
		}
	}
	cout << endl;
	return rowMat;
}

//A function print contents of the matrix
void printMatrix(Matrix* mat) {
	cout << "row: " << mat->row << endl;
	cout << "col: " << mat->col << endl;
	cout << "matrix" << endl;
	for (int i = 0; i < mat->row * mat->col; i = i + mat->col) {
		for (int j = 0; j < mat->col; j++) {
			cout << mat->val[i + j] << "  ";
		}
		cout << endl;
	}
	cout << endl;
}

// Finding RMS Error between Shared Matrix Multiplication and Cublas Multiplication
void verifyMultiplication(Matrix* matrixShared, Matrix* matrixCublas){
	if(matrixShared->row != matrixCublas->row || matrixShared->col != matrixCublas->col)
		cout <<"Shared Matrix Multiplication and cublas multiplication is not of same size" <<endl;
	double sum = 0;
	for(int i = 0; i < matrixShared->row; i++){
		for(int j = 0; j<matrixShared->col; j++){
			sum += pow((matrixShared->val[i*matrixShared->col + j] - matrixCublas->val[i*matrixShared->col + j]), 2);
		}
	}
	//cout << sum << endl;
	sum = sqrt(sum / (matrixShared->col * matrixShared->row));
	cout << "RMS error between Shared Memory Multplication and Cublas Mutliplication is " << sum << endl;
	
}

//cpu multiplication
Matrix* cpuMultiplication(Matrix* matrixA, Matrix* matrixB) {
	Matrix* matrixC = (Matrix*)malloc(sizeof(Matrix));
	matrixC->row = matrixA->row;
	matrixC->col = matrixB->col;
	matrixC->val = (float*)malloc(matrixC->row * matrixC->col * sizeof(float));
	if (matrixA->col != matrixB->row) {
		return NULL;
	}
	else {
		for (int i = 0; i < matrixA->row; i++) {
			float sum = 0;
			for (int j = 0; j < matrixB->col; j++) {
				for (int k = 0; k < matrixA->col; k++) {
					//cout << i << " X " << k << " , " << k << " x " << j;
					sum += matrixA->val[i * matrixA->col + k] * matrixB->val[k * matrixB->col + j];
				}
				//cout << endl;
				matrixC->val[i * matrixC->col + j] = sum;
				sum = 0;
			}
		}
	}
	return matrixC;
}

//function to read matrix from mtx file
Matrix* readMatrix(const char* filePath) {
	Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
	//float* val;
	FILE *fp;
	fp = fopen(filePath, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filePath);
		exit(1);
	}

	if (fread(&mat->row, sizeof(int), 1, fp) == 0)
	{
		cout << "Error reading row value";
	}
	if (fread(&mat->col, sizeof(int), 1, fp) == 0)
	{
		cout << "Error reading column value";
	}

	mat->val = (float*)malloc(mat->row * mat->col * sizeof(float));

	for (int i = 0; i < mat->row * mat->col; i++) {
		if (fread(&mat->val[i], sizeof(float), 1, fp) == 0)
		{
			cout << "Error reading value";
		}
	}
	return mat;
}

//function to write back the results
int writeMatrix(Matrix* mat, const char* filename) {
	FILE* file = fopen(filename, "wb+");

	if (file == NULL)
		return 0;
	fwrite(&mat->row, sizeof(int), 1, file);
	fwrite(&mat->col, sizeof(int), 1, file);
	for (int i = 0; i < mat->row; i++) {
		for(int j= 0; j< mat->col; j++){
			fwrite(&mat->val[i*mat->col + j], sizeof(float), 1, file);
		}
	}
	fclose(file);
	return 1;
}

// function to calculate number of floating point operation
void calculateGFLOPS(){
	cout << endl <<" ***************** Counting Number of FLOPS *****************" << endl;
	int numGLOPS = 720 / 4;
	cout << " num of GLOPS: "<< numGLOPS << endl;
	float perf = (numGLOPS * 100 / 5304);
	cout << " Performance: "<< perf << endl;
}


//main method that invokes all other functions
int main(int argc, char* argv[]) {
	cout << "hello";
	if (argc != 4)
		return 1;
	std::string filename(argv[1]);

	cout << "argv[1]" << argv[1] << endl;

	Matrix* matrixA = (Matrix*)malloc(sizeof(Matrix));
	if (!matrixA) {
		fprintf(stderr, "Unable to locate memory.\n");
		exit(1);
	}

	Matrix* matrixB = (Matrix*)malloc(sizeof(Matrix));;
	if (!matrixB) {
		fprintf(stderr, "Unable to locate memory.\n");
		exit(1);
	}

	Matrix* matrixC = (Matrix*)malloc(sizeof(Matrix));;
	if (!matrixC) {
		fprintf(stderr, "Unable to locate memory.\n");
		exit(1);
	}
	//exit(1);
	cout << " reading matrix a" << endl;
	matrixA = readMatrix(argv[1]);

	cout << " reading matrix b " << endl;
	matrixB = readMatrix(argv[2]);

	printMatrix(matrixA);
	printMatrix(matrixB);

	cout << "***** cpu mutliplication *****" << endl;
	matrixC = cpuMultiplication(matrixA, matrixB);
	printMatrix(matrixC);

	Matrix* gpuMatrixC = (Matrix*)malloc(sizeof(Matrix));;
	if (!gpuMatrixC) {
		fprintf(stderr, "Unable to locate memory.\n");
		exit(1);
	}
	cout << "***** GPU multiplication *****" << endl;
	gpuMatrixC = invokeGpuMultiplication(matrixA, matrixB);
	//printMatrix(gpuMatrixC);

	cout << "***** multiplication using cublas *****" << endl;
	Matrix* gpuMatrixCublas = (Matrix*)malloc(sizeof(Matrix));
	if (!gpuMatrixCublas) {
		fprintf(stderr, "Unable to locate memory.\n");
		exit(1);
	}

	//gpuMatrixCublas = ;
	gpuMatrixCublas = invokeGpuMultiplicationCublas(matrixA, matrixB);
	//printMatrix(gpuMatrixCublas);

	cout << "***** multiplication using shared Mem *****" << endl;
	Matrix* gpuMatrixSharedMem = (Matrix*)malloc(sizeof(Matrix));
	if (!gpuMatrixSharedMem) {
		fprintf(stderr, "Unable to locate memory.\n");
		exit(1);
	}

	//gpuMatrixCublas = ;
	gpuMatrixSharedMem = invokeGpuMultiplicationSharedMem(matrixA, matrixB);
	//printMatrix(gpuMatrixSharedMem);
	
	//writing back the file
	writeMatrix(gpuMatrixSharedMem, argv[3]);
	//checking if file was created correctly.

	Matrix* matrixD = (Matrix*)malloc(sizeof(Matrix));
	if (!matrixD) {
		fprintf(stderr, "Unable to locate memory.\n");
		exit(1);
	}
	
	cout <<"***** Reading the output to verify the output *****" <<endl;
	matrixD = readMatrix(argv[3]);
	//printMatrix(matrixD);
	
	cout << endl << endl << "     ******************     Calculating RMS Error      ******************      " << endl;
	verifyMultiplication(gpuMatrixC, gpuMatrixCublas);
	//calculateGFLOPS();
	return 0;
}

 