#ifndef NETIMAGE_H
#define NETIMAGE_H

#include <stdio.h>

typedef struct 
{
  unsigned char min;
  unsigned char max;
} minMax;

typedef enum{NOTHING,NEGATE,BRIGHTEN,SHARPEN,SMOOTH,GRAYSCALE,CONTRAST} op_t;

typedef enum{ BINARY,ASCII,DEFAULT } out_t;

typedef struct{
  unsigned char **r, **g, **b;
}rgbdata;

typedef struct{
  unsigned char **i;
}graydata;

typedef struct
{	
  char  magicNumber;      // type of net image
  char* comment;          // the comments
  int bpp;                // the number of bits per pixel
  int rows;               // number of rows
  int cols;               // number of cols
  union{                  // the union allows us to access
    rgbdata rgb;          // the image data as either rgb 
    graydata gray;        // or grayscale.  magicNumber tells which
  };                 // type of data it is.
}netImage;

netImage *readImage(FILE *inf);
void writeImage(FILE *inf, netImage *image);

void negateImage( netImage *image);
void brightenImage( netImage *image, int factor);
void sharpenImage( netImage *image);
void smoothImage( netImage *image);
minMax grayscaleImage( netImage *image);
void contrastImage( netImage *image);

#endif
