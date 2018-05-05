/********************************************//**
 * Author: Nathan Ducasse
 *
 * File: edge.cu
 *
 * Compilation: make
 *
 * Usage: ./edge file.png
 *
 * Program Description: This program benchmarks the
 *   difference between the Sobel edge detector
 *   written serially, parallelized on the CPU
 *   with OpenMP, and parallelized on the GPU
 *   with CUDA.
 *
 * File Description: This file contains the main
 *   method, the getImage function, which
 *   returns the grayscale image, and the
 *   printBenchmark function, which outputs the
 *   system specs.
 ***********************************************/

#include "edge.cuh"

void printBenchmark();
void getImage(byte *image, byte *pixels, int width, int height);

/********************************************//**
 * Author: Nathan Ducasse
 * Description: Main function. Reads in the given
 *  PNG image, calls the function to convert it
 *  to grayscale, and calls the doTest function
 *  to proceed with the benchmark.
 ***********************************************/
int main( int argc, char** argv ) {

    // check usage
    if ( argc < 2 ) {
        printf( "Usage: %s infile.png\n", argv[0] );
        return -1;
    }

    // print benchmark info
    printBenchmark();
    
    // read input PNG file
    byte* pixels;
    unsigned int width, height;
    unsigned error = lodepng_decode_file(&pixels, &width, &height, argv[1], LCT_RGBA, 8);
    if ( error ) {
        printf( "decoder error while reading file %s\n", argv[1] );
        printf( "error code %u: %s\n", error, lodepng_error_text( error ) );
        return -2;
    }
    printf( "Processing %s: %d rows x %d columns\n", argv[1], height, width );
    
    // get the grayscale image from the color image
    byte *image;
    int npixels = width * height;
    // I had to use CUDA malloc to get the CUDA to work.
    // It still works with the other functions regardless.
    cudaMallocManaged(&image, npixels*sizeof(unsigned char));
    getImage(image, pixels, width, height);

    // get the filename before the .png
    string filename = string(argv[1]).substr(0, string(argv[1]).length()-4);

    // do the benchmarking
    int ret = doTests(image, filename, width, height);
    
    // free memory
    cudaFree(image);
    return ret;
}


/********************************************//**
 * Author: Nathan Ducasse
 * Description: Converts pixels into grayscale,
 *   returns the grayscale in image.
 ***********************************************/
void getImage(byte *image, byte *pixels, int width, int height) {
    // copy 24-bit RGB data into 8-bit grayscale intensity array
    int npixels = width * height;
    byte* img = pixels;
    for ( int i = 0; i < npixels; ++i ) {
        int r = *img++;
        int g = *img++;
        int b = *img++;
        int a = *img++;     // alpha channel is not used
        image[i] = 0.3 * r + 0.6 * g + 0.1 * b + 0.5;
    }
    
    free( pixels );     // LodePNG uses malloc, not new
}


/********************************************//**
 * Author: Nathan Ducasse, John Weiss
 * Description: Prints the benchmark information
 *   for the computer: time, cores, GPU info.
 ***********************************************/
void printBenchmark() {
        // CUDA device properties
    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );
    int cores = devProp.multiProcessorCount;
    switch ( devProp.major ) {
        case 2: // Fermi
            if ( devProp.minor == 1 ) cores *= 48;
            else cores *= 32; break;
        case 3: // Kepler
            cores *= 192; break;
        case 5: // Maxwell
            cores *= 128; break;
        case 6: // Pascal
            if ( devProp.minor == 1 ) cores *= 128;
            else if ( devProp.minor == 0 ) cores *= 64;
            break;
    }
    // print header
    time_t currtime = time( 0 );
    printf( "edge map benchmarks: %s", ctime( &currtime ) );
    printf( "CPU: %d hardware threads\n", thread::hardware_concurrency() );
    printf( "GPGPU: %s, CUDA %d.%d, %d Mbytes global memory, %d CUDA cores\n",
            devProp.name, devProp.major, devProp.minor, devProp.totalGlobalMem / 1048576, cores );
    cout << endl;

}

