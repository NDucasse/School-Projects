#ifndef HELPERS_H
#define HELPERS_H

// saturate returns the number between 0 and 255 which is closest to v
unsigned char saturate(int v);

// a helper to allocate an array to hold pixel values
unsigned char **allocate_pixel_array(int rows, int cols);

// a helper to free an array of pixel values
void free_pixel_array(unsigned char **a, int rows);


#endif
