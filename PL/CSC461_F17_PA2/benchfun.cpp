/********************************************//**
 * File: benchfun.cpp
 * Author: Nathan Ducasse
 * Description: This file contains all of the
 *   functions which have to do with the CPU and
 *   OMP benchmarks.
 ***********************************************/

#include "edge.cuh"

double doCPUTest(byte *image, string filename, unsigned int width, unsigned int height);
double doOMPTest(byte *image, string filename, unsigned int width, unsigned int height);
byte *CPUEdgeDetector(byte *image, unsigned int width, unsigned int height);
byte *OMPEdgeDetector(byte *image, unsigned int width, unsigned int height);
byte getMagnitude(byte *image, int i, int j, int width);

/********************************************//**
 * Author: Nathan Ducasse
 * Description: Core function for the program.
 *   calls the functions to be benchmarked for
 *   each of the Sobel edge detector 
 *   implementations. This function also prints
 *   the time benchmarks for each of the function
 *   calls and prints the differences in benchmark
 *   times.
 ***********************************************/
int doTests(byte *image, string filename, unsigned int width, unsigned int height) {
    double cpux, ompx, cudax;
    
    cpux = doCPUTest(image, filename, width, height);
    
    if(cpux < 0) {
        return -3;
    }
    
    cout << "CPU execution time = " << fixed << setprecision(2) << cpux << " msec" << endl;

    // time the benchmarking for the OpenMP implementation
    ompx = doOMPTest(image, filename, width, height);
    
    if(ompx < 0) {
        return -3;
    }
    
    cout << "OpenMP execution time = " << fixed << setprecision(2)  << ompx << " msec" << endl;

    
    // time the benchmarking for the CUDA implementation
    cudax = doCUDATest(image, filename, width, height);
    if(cudax < 0) {
        return -3;
    }
    
    cout << "CUDA execution time = " << fixed << setprecision(2) << cudax << " msec" << endl << endl;
    
    // Get the times speedup for each of the implementations
    cout << "CPU -> OMP Speedup: " << right << setw(7) << (cpux/ompx) << " X" << endl;
    cout << "OMP -> GPU Speedup: " << right << setw(7) << (ompx/cudax) << " X" << endl;
    cout << "CPU -> GPU Speedup: " << right << setw(7) << (cpux/cudax) << " X" << endl;
    
    return 0;
}


/********************************************//**
 * Author: Nathan Ducasse
 * Description: Benchmark function for the CPU
 *   implementation of the Sobel edge detector.
 *   Times the function call and outputs the png
 *   image, and returns the time taken.
 ***********************************************/
double doCPUTest(byte *image, string filename, unsigned int width, unsigned int height){
    byte *CPUImage;
    
    // allocate the memory for the image
    CPUImage = new byte[width*height];
    
    // time the benchmark
    auto begin = std::chrono::high_resolution_clock::now();
    CPUImage = CPUEdgeDetector(image, width, height);
    auto end = chrono::high_resolution_clock::now();
    
    // write grayscale PNG file
    auto error =  lodepng_encode_file( (filename + "_cpu.png").c_str(), CPUImage, width, height, LCT_GREY, 8 );
    if ( error )
    {
        printf( "encoder error while writing file %s\n", "image_cpu.png" );
        printf( "error code %u: %s\n", error, lodepng_error_text( error ) );
        delete [] CPUImage;
        return -1;
    }
    
    // free memory
    delete [] CPUImage;
    // fix value and return
    return chrono::duration_cast<chrono::nanoseconds>(end-begin).count() / double(1000000);
}


/********************************************//**
 * Author: Nathan Ducasse
 * Description: Benchmark function for the OMP
 *   implementation of the Sobel edge detector.
 *   Times the function call and outputs the png
 *   image, and returns the time taken.
 ***********************************************/
double doOMPTest(byte *image, string filename, unsigned int width, unsigned int height) {
    byte *OMPImage;
    
    // allocate the memory for the image
    OMPImage = new byte[width*height];
    
    // time the benchmark
    auto begin = std::chrono::high_resolution_clock::now();
    OMPImage = OMPEdgeDetector(image, width, height);
    auto end = chrono::high_resolution_clock::now();
    
    // write grayscale PNG file
    auto error =  lodepng_encode_file( (filename + "_omp.png").c_str(), OMPImage, width, height, LCT_GREY, 8 );
    if ( error )
    {
        printf( "encoder error while writing file %s\n", "image_omp.png" );
        printf( "error code %u: %s\n", error, lodepng_error_text( error ) );
        delete [] OMPImage;
        return -1;
    }
    
    // free memory
    delete [] OMPImage;
    // fix time value and return
    return chrono::duration_cast<chrono::nanoseconds>(end-begin).count() / double(1000000);
}


/********************************************//**
 * Author: Nathan Ducasse
 * Description: CPU implementation for the Sobel
 *   edge detector. This function contains the
 *   basic for loops that call the getMagnitude()
 *   function. Returns the edge-detected image.
 ***********************************************/
byte *CPUEdgeDetector(byte *image, unsigned int width, unsigned int height) {
    byte *gradient = new byte [width*height];    
    for(int i = 1; i<height-1; i++) {
        for(int j = 1; j<width-1; j++) {
            if(i != height-1 && j != width-1) {
                gradient[i*width+j] = getMagnitude(image, i, j, width);
            } else { // fix edge
                gradient[i*width+j] = 0;
            }
        }
    }
    return gradient;
}


/********************************************//**
 * Author: Nathan Ducasse
 * Description: OpenMP implementation for the
 *   Sobel edge detector. This function contains
 *   the omp parallel for loops that call the
 *   getMagnitude() function. Returns the edge-
 *   detected image.
 ***********************************************/
byte *OMPEdgeDetector(byte *image, unsigned int width, unsigned int height) {
    byte *gradient = new byte [width*height];
    // collapse(2) should parallelise the nested loop 
    // as well as the outer loop
    #pragma omp parallel for collapse(2)
    for(int i = 1; i<height-1; i++) {
        for(int j = 1; j<width-1; j++) {
            if(i != height-1 && j != width-1) {
                gradient[i*width+j] = getMagnitude(image, i, j, width);
            } else {
                // fix edge
                gradient[i*width+j] = 0;
            }
        }
    }
    return gradient;
}


/********************************************//**
 * Author: Nathan Ducasse
 * Description: The getMagnitude function does the
 *   base calculations for the Sobel edge detectors.
 ***********************************************/
byte getMagnitude(byte *image, int i, int j, int width) {
    // calculates the x mask
    int Gx = image[(i+1)*width + (j-1)] 
            + 2*image[(i+1)*width + j] 
            + image[(i+1)*width + (j+1)] 
            - image[(i-1)*width + (j-1)] 
            - 2*image[(i-1)*width + j] 
            - image[(i-1)*width + (j+1)];
    // calculates the y mask
    int Gy = image[(i+1)*width + (j+1)] 
            + 2*image[i*width + (j+1)] 
            + image[(i+1)*width + (j+1)] 
            - image[(i-1)*width + (j-1)] 
            - 2*image[i*width + (j-1)] 
            - image[(i+1)*width + (j-1)];
    
    // get and bound the pixel's magnitude
    int mag = sqrt(Gx*Gx + Gy*Gy);
    if(mag > 255) {
        mag = 255;
    }
    return mag;
}
