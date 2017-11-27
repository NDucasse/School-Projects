#include <stdio.h>
#include <stdlib.h>
#include <netImage.h>
#include <helpers.h>

static void negate_array(unsigned char **a,int rows,int cols)
{
  int i,j;
  for( i=0; i<rows; i++)
    for( j=0; j<cols; j++)
      a[i][j] = saturate(255 - a[i][j]);
}

/* negate the image */
void negateImage( netImage *image)
{
  switch(image->magicNumber)
    {
    case '3':
    case '6':
      negate_array(image->rgb.r,image->rows,image->cols);
      negate_array(image->rgb.g,image->rows,image->cols);
      negate_array(image->rgb.b,image->rows,image->cols);
      break;
    case '2':
    case '4':
      negate_array(image->gray.i,image->rows,image->cols);
      break;
    default:
      fprintf(stderr,"Error negating unknown image type\n");
      exit(1);
    }
}
