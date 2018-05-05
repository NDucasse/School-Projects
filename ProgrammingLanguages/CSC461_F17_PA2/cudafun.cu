/********************************************//**
 * File: cudafun.cu
 * Author: Nathan Ducasse
 * Description: This file contains all of the
 *   functions that call anything CUDA related,
 *   aside from the main function and system specs.
 ***********************************************/
 
#include "edge.cuh"

__device__ byte getMagnitude1D(byte *image, int i, int j, int width);
__global__ void CUDAEdgeDetector(byte *image, byte *gradient, unsigned int width, unsigned int height);

/********************************************//**
 * Author: Nathan Ducasse
 * Description: Benchmark function for the CUDA
 *   implementation of the Sobel edge detector.
 *   Times the function call and outputs the png
 *   image, and returns the time taken.
 ***********************************************/
double doCUDATest(byte *image, string filename, unsigned int width, unsigned int height) {
    byte *CUDAImage;
    
    // allocate the memory for the image
    cudaMallocManaged(&CUDAImage, width*height*sizeof(unsigned char));
    
    // find the number of blocks and threads per block dynamically
    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );
    int cores = devProp.multiProcessorCount;
    int blockdim = sqrt(double(cores));
    dim3 block(blockdim, blockdim);
    dim3 grid(((height)+blockdim-1)/blockdim, ((width)+blockdim-1)/blockdim);
    
    // time the benchmark, then synchronize the system.
    auto begin = std::chrono::high_resolution_clock::now();
    CUDAEdgeDetector<<<grid, block>>>(image, CUDAImage, width, height);
    auto end = chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    
    // write grayscale PNG file
    auto error =  lodepng_encode_file( (filename + "_gpu.png").c_str(), CUDAImage, width, height, LCT_GREY, 8 );
    if ( error )
    {
        printf( "encoder error while writing file %s\n", "image_omp.png" );
        printf( "error code %u: %s\n", error, lodepng_error_text( error ) );
        cudaFree(CUDAImage);
        return -1;
    }

    // free memory
    cudaFree(CUDAImage);
    // fix time value and return
    return chrono::duration_cast<chrono::nanoseconds>(end-begin).count() / double(1000000);
}


/********************************************//**
 * Author: Nathan Ducasse
 * Description: CUDA implementation for the
 *   Sobel edge detector. The function parallelizes
 *   each pixel operation and uses the block
 *   dimensions and thread count to find "i" and "j"
 *   positions of each pixel.
 ***********************************************/
__global__ void CUDAEdgeDetector(byte *image, byte *gradient, unsigned int width, unsigned int height) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(i > 0 && j > 0 && i < height-1 && j < width-1) {
        gradient[i*width+j] = getMagnitude1D(image, i, j, width);
    } else if((i == 0 && j == 0) // make sure ONLY edge cases are worked on
           || (i == 0 && j == (width-1)) 
           || (i == (height-1) && j == 0)
           || (i == (height-1) && j == (width-1))) { 
        // fix edge
        gradient[i*width+j] = 0;
    }
}


/********************************************//**
 * Author: Nathan Ducasse
 * Description: CUDA version of the getMagnitude()
 *   function. It does the same things but uses
 *   the __global__/__device__ function sqrtf()
 *   instead of math sqrt.
 ***********************************************/
__device__ byte getMagnitude1D(byte *image, int i, int j, int width) {
    // calculates x mask
    int Gx = image[(i+1)*width + (j-1)] 
            + 2*image[(i+1)*width + j] 
            + image[(i+1)*width + (j+1)] 
            - image[(i-1)*width + (j-1)] 
            - 2*image[(i-1)*width + j] 
            - image[(i-1)*width + (j+1)];
    // calculates y mask
    int Gy = image[(i+1)*width + (j+1)] 
            + 2*image[i*width + (j+1)] 
            + image[(i+1)*width + (j+1)] 
            - image[(i-1)*width + (j-1)] 
            - 2*image[i*width + (j-1)] 
            - image[(i+1)*width + (j-1)];
    
    // get and bound the pixel's magnitude
    int mag = sqrtf(Gx*Gx + Gy*Gy);
    if(mag > 255) {
        mag = 255;
    }
    return mag;
}
