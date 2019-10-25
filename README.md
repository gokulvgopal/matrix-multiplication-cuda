# matrix-multiplication-cuda
Implementing Matrix Multiplication in CPU and GPU (in three different techniques)

## Description
This project is the implementation of General Matrix Multiplication(gemm) in CPU and GPU. Input to the program is two matrix files that contains two 32-bit unsigned integers representing the row and column size of the matrix, followed by the matrix elements as 32-bit floating points in column-major order.
#### Command Line to run the program: 
`
kernel.cu a.mtx b.mtx c.mtx
where a.mtx and b.mtx are input matrix
      c.mtx is the output matrix
`

This project contains the following:
##### Matrix struct - Data structure used to store row size, column size and elements of the matrix.
##### 1. Read .mtx format, display and write back the result in .mtx format
Functions implemented: 
readMatrix() - Takes in File location as input, reads .mtx file to extract row size, column size and elements.
printMatrix() - Takes matrix object as input and displays row and column size followed by matrix elements.
writeMatrix() - Takes in matrix object and file name, writes row and column size followed by elements.

##### 2. Implementation in CPU
cpuMultiplication() - Function to calculate matrix multipication in CPU. Takes two matrix as input and return output matrix.

##### 3. Implementation in GPU using CUDA - Default Method
invokeGpuMultiplication() - CPU function to invoke device function. It takes care of declaring, copying to and from and deleting device variables. It also calculates number of blocks and threads per block and invokes gpuMultiplication() function.
gpuMultiplication() - The device function that uses devices variables to find multiplication of two matrix. It calculates by using thread and block index ID to locate the elements to be multiplied and added.

##### 4. Implementation in GPU using CUDA - using shared memory
invokeGpuMultiplicationSharedMem() - CPU function to invoke device function. It takes care of declaring, copying to and from and deleting device variables. It also calculates number of blocks and threads per block and invokes gpuMultiplicationSharedMem() function.
gpuMultiplicationSharedMem() - The device function that uses devices variables to find multiplication of two matrix. It also takes care of defining the shared memory and uses that memory to copy necessary elements. It calculates by using thread and block index ID to locate the elements in the shared memory to be multiplied and added.

##### 5. Implementation in GPU using CUDA - Cublas library
invokeGpuMultiplicationCublas() - CPU function to invoke device function. It takes care of declaring, copying to and from and deleting device variables. It also calculates number of blocks and threads per block and invokes gpuMultiplicationUsingCublas() function.
gpuMultiplicationUsingCublas() - The device function that uses devices variables to find multiplication of two matrix. It calculates by invoking cublasSgemm() function.

##### 6. Measure the performance of each GPU implementation in terms of Convolution time, Total time taken by kernel, FLOPS, and B/S
Each of these parameters are implemented in each of the host functions that invoke device functions. The host functions are listed below. Time is calculated in milliseconds using cudaEvent_t to records start and end time. This time is used to calculate FLOPS and B/S.
a. invokeGpuMultiplication()
b. invokeGpuMultiplicationSharedMem()
c. invokeGpuMultiplicationCublas()

##### 7. Verify the results of each GPU implementation with CPU implementation by calculating Mean Square Error (MSE)
verifyMultiplication() - Output matrix from CPU multiplication and GPU multiplication is taken as input and mean square error is printed.

###### 8. Helper functions
These functions are required to read the matrix in column wise and row wise. These also serve the purpose of changing column major function to row major matrix. Since the input is column major matrix and cublas require row major matrix as input, these functionalities are required to convert the input matrix to row major and output matrix back to column major.
a. changeMatrixColWise()
b. changeMatrixRowWise() 
