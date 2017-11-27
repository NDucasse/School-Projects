
#include <stdlib.h>
#include <stdio.h>
#include <netImage.h>
#include <helpers.h>

static void brighten_array(unsigned char **a,int rows,int cols,int factor)
{
  int i,j;
  for( i=0; i<rows; i++)
    for( j=0; j<cols; j++)
      a[i][j] = saturate(a[i][j] + factor);
}

/* brighten the image */
void brightenImage( netImage *image, int factor)
{
  switch(image->magicNumber)
    {
    case '3':
    case '6':
      brighten_array(image->rgb.r,image->rows,image->cols,factor);
      brighten_array(image->rgb.g,image->rows,image->cols,factor);
      brighten_array(image->rgb.b,image->rows,image->cols,factor);
      break;
    case '2':
    case '4':
      brighten_array(image->gray.i,image->rows,image->cols,factor);
      break;
    default:
      fprintf(stderr,"Error brightening unknown image type\n");
      exit(1);
    }

}
