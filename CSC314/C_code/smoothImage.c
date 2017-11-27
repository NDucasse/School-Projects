#include <stdlib.h>
#include <stdio.h>
#include <netImage.h>
#include <helpers.h>



/* Apply the smooth mask to create a new array.  Free the original
   and return the new array */
static unsigned char **smooth_array( unsigned char **a, int rows,int cols)
{
  int i,j,sum;
  unsigned char **tmp;

  tmp = allocate_pixel_array(rows,cols);

  for( i=1; i<rows-1; i++)
    for( j=1; j<cols -1; j++ )
      {
	sum = 4*(int)a[i][j] +
	  2*((int)a[i-1][j]+a[i+1][j]+a[i][j-1]+a[i][j+1])+ 
	  ((int)a[i-1][j-1]+a[i+1][j-1]+a[i+1][j-1]+a[i+1][j+1]);
	
	tmp[i][j] = saturate(sum/16);
      }
  
  /* fix up the corners */
  tmp[0][0] = (4*(int)a[0][0] + 2*(a[1][0]+a[0][1]) + a[1][1])/9;

  tmp[0][cols-1] = (4*(int)a[0][cols-1] +
		    2*((int)a[1][cols-1]+a[0][cols-2]) + a[1][cols-2])/9;

  tmp[rows-1][0] = (4*(int)a[rows-1][0] +
		    2*((int)a[rows-2][0]+a[rows-1][1]) + a[rows-2][1])/9;

  tmp[rows-1][cols-1] = (4*(int)a[rows-1][cols-1] +
			 2*((int)a[rows-2][cols-1]+a[rows-1][cols-2]) +
			 a[rows-2][cols-2])/9;


  /* fix up the vertical borders */
  for(i=1;i<rows-1;i++)
    {    
      tmp[i][0] = (4*(int)a[i][0] +
		   2*((int)a[i-1][0] + a[i][1] + a[i+1][0])+
		   ((int)a[i-1][1] + a[i+1][1]))/12;
      
      tmp[i][cols-1] = (4*(int)a[i][cols-1] +
			2*((int)a[i-1][cols-1] + a[i][cols-2] + a[i+1][cols-1])+
			((int)a[i-1][cols-2] + a[i+1][cols-2]))/12;
    }
  
  /* fix up the horizontal borders */
  for(i=1;i<cols-1;i++)
    {    
      tmp[0][i] = (4*(int)a[0][i] +
		   2*((int)a[0][i-1] + a[1][i] + a[0][i+1])+
		   ((int)a[1][i-1] + a[1][i+1]))/12;
      
      tmp[rows-1][i] = (4*(int)a[rows-1][i] +
			2*((int)a[rows-1][i-1] + a[rows-2][i] + a[rows-1][i+1])+
			((int)a[rows-2][i-1] + a[rows-2][i+1]))/12;
    }
  
  free_pixel_array(a,rows);
  return tmp;

}


void smoothImage( netImage *image)
{
  switch(image->magicNumber)
    {
    case '3':
    case '6':
      image->rgb.r=smooth_array(image->rgb.r,image->rows,image->cols);
      image->rgb.g=smooth_array(image->rgb.g,image->rows,image->cols);
      image->rgb.b=smooth_array(image->rgb.b,image->rows,image->cols);
      break;
    case '2':
    case '4':
      image->gray.i=smooth_array(image->gray.i,image->rows,image->cols);
      break;
    default:
      fprintf(stderr,"Error smoothing unknown image type\n");
      exit(1);
    }
}

  
