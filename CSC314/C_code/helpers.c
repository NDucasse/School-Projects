#include <helpers.h>
#include <stdlib.h>
#include <stdio.h>


// saturate returns the number between 0 and 255 which is closest to v
unsigned char saturate(int v)
{
  if(v<0)
    return 0;
  if(v>255)
    return 255;
  return v;
}

// a helper to allocate an array to hold pixel values
unsigned char **allocate_pixel_array(int rows, int cols)
{
  unsigned char **array;
  int i;
  if((array = malloc(sizeof(unsigned char *)*rows))==NULL)
    {
      fprintf(stderr,"Unable to allocate space for input image\n");
      exit(1);
    }
  for(i=0;i<rows;i++)
    if((array[i] = malloc(sizeof(unsigned char)*cols))==NULL)
      {
	fprintf(stderr,"Unable to allocate space for input image\n");
	exit(1);
      }
  return array;
}

// a helper to free an array of pixel values
void free_pixel_array(unsigned char **a, int rows)
{
  int i;
  for(i=0;i<rows;i++)
    free(a[i]);
  free(a);
}

