/********************************************//**
 * File: edge.cuh
 * Author: Nathan Ducasse
 * Description: This file is the header file for
 *   the entire program. It contains all of the
 *   includes and necessary file/type definitions.
 ***********************************************/
#ifndef EDGE_CUH_
#include <iostream>
#include <cmath>
#include <iomanip>
#include <thread>
#include "lodepng.h"

using namespace std;

typedef unsigned char byte;

void printBenchmark();
int doTests(byte *image, string filename, unsigned int width, unsigned int height);
double doCUDATest(byte *image, string filename, unsigned int width, unsigned int height);
#endif
