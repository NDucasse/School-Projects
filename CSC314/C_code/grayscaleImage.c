#include <stdio.h>
#include <stdlib.h>
#include <netImage.h>
#include <helpers.h>

static minMax get_bounds(unsigned char **a,int rows,int cols)
{
  int i,j;
  minMax mm = {255,0};
  for(i=0;i<rows;i++)
    for(j=0;j<cols;j++)
      {
	if(a[i][j]<mm.min)
	  mm.min=a[i][j];
	if(a[i][j]>mm.max)
	  mm.max=a[i][j];
      }
  return mm;
}

/* Define a type to hold minmax and pointer */
typedef struct{
  minMax mm;
  unsigned char **grayimage;
}funky_return_value;

/* convert no grayscale, return min/max and pointer */
static funky_return_value gray_array(unsigned char **r,
				     unsigned char **g,
				     unsigned char **b,
				     int rows,
				     int cols)
{
  int i,j;
  funky_return_value vals = {{255, 0}, NULL};
  unsigned char **grayimage;

  grayimage = allocate_pixel_array(rows,cols);
  
  for( i=0; i<rows; i++)
    for( j=0; j<cols; j++)
      {
	grayimage[i][j] = saturate((54 * r[i][j] +
				    184* g[i][j] +
				    18 * b[i][j] + 
				    128)/256);
	if( grayimage[i][j] < vals.mm.min )
	  vals.mm.min = grayimage[i][j];
	if( grayimage[i][j] > vals.mm.max )
	  vals.mm.max = grayimage[i][j];
      }
  vals.grayimage = grayimage;
  return vals;
}



/* Convert image to grayscale. */
minMax grayscaleImage( netImage *image)
{
  funky_return_value vals;
  switch(image->magicNumber)
    {
    case '3':
    case '6':
      image->magicNumber--;
      vals = gray_array(image->rgb.r,image->rgb.g,image->rgb.b,
			image->rows,image->cols);
      free_pixel_array(image->rgb.r,image->rows);
      free_pixel_array(image->rgb.g,image->rows);
      free_pixel_array(image->rgb.b,image->rows);
      image->gray.i = vals.grayimage;
      break;
    case '2':
    case '4':
      /* Already grayscale.  Do nothing but get the bounds. */
      vals.mm = get_bounds(image->gray.i,image->rows,image->cols);
      break;
    default:
      fprintf(stderr,"Error graying unknown image type\n");
      exit(1);
    }
  return vals.mm;
}

  


  
