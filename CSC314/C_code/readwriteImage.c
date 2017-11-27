#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <netImage.h>
#include <helpers.h>


/* Read a netpbm image.  This function can read color or grayscale,
   but not bitmaps. */
netImage *readImage(FILE *inf)
{
  netImage *image;
  int i, j;
  int tmp1, tmp2, tmp3;
  char nextchar;
  unsigned char tmpdat[3];
  
  if((image = malloc(sizeof(netImage)))==NULL)
    {
      fprintf(stderr,"Unable to allocate space for input image\n");
      exit(1);
    }
  image->comment = NULL;
  image->gray.i = NULL;
  image->rgb.r = NULL;
  image->rgb.g = NULL;
  image->rgb.b = NULL;
  image->rows = 0;
  image->cols = 0;
  image->bpp = 0;
  image->magicNumber='0';
  nextchar = fgetc(inf);
  if(nextchar != 'P')
    {
      fprintf(stderr,"Input file is not a netpbm image\n");
      exit(1);
    }
  nextchar = fgetc(inf);
  if(!strchr("2356",nextchar))
    {
      fprintf(stderr,"Input file is not a color or grayscale image file\n");
      exit(1);
    }
  image->magicNumber = nextchar;
  // find next non-whitespace
  while(isspace(nextchar = fgetc(inf)));
  // strip out comment lines
  if(nextchar == '#')
    {
      i = 0;
      image->comment = malloc(i+2);
      while(nextchar=='#')
	{
	  image->comment[i] = nextchar;
	  while((nextchar = fgetc(inf)) != '\n')
	    {
	      i++;
	      image->comment = realloc(image->comment, i+2);
	      image->comment[i]=nextchar;
	    }
	  i++;
	  image->comment = realloc(image->comment, i+2);
	  image->comment[i]=nextchar;
	  nextchar = fgetc(inf);
	  if(nextchar=='#')
	    {
	      i++;
	      image->comment = realloc(image->comment, i+2);
	      image->comment[i]=nextchar;
	    }
	}
      image->comment[i+1] = 0;
    }
  if(ungetc(nextchar, inf) == EOF)
    {
      fprintf(stderr,"Error reading input file header\n");
      exit(1);
    }
  if(fscanf(inf,"%d%d%d",&(image->cols),&(image->rows),&(image->bpp))!=3)
    {
      fprintf(stderr,"Error reading input file header\n");
      exit(1);
    }
  while((nextchar = fgetc(inf)) != '\n');
    
  switch(image->magicNumber)
    {
    case '3': /* ASCII color  */
      image->rgb.r = allocate_pixel_array(image->rows,image->cols);
      image->rgb.g = allocate_pixel_array(image->rows,image->cols);
      image->rgb.b = allocate_pixel_array(image->rows,image->cols);
      for(i=0;i<image->rows;i++)
	for( j=0; j<image->cols; j++)
	  {
	    if(fscanf(inf,"%d%d%d",&tmp1,&tmp2,&tmp3)!=3)
	      {
		fprintf(stderr,"Unable to read all of image data\n");
		exit(1);
	      }
	    image->rgb.r[i][j] = tmp1;
	    image->rgb.g[i][j] = tmp2;
	    image->rgb.b[i][j] = tmp3;
	  }
      break;
	  
    case '6': /* binary color */
      image->rgb.r = allocate_pixel_array(image->rows,image->cols);
      image->rgb.g = allocate_pixel_array(image->rows,image->cols);
      image->rgb.b = allocate_pixel_array(image->rows,image->cols);
      for(i=0;i<image->rows;i++)
	for( j=0; j<image->cols; j++)
	  {
	    if(fread(tmpdat,sizeof(unsigned char),3,inf)!=3)
	      {
		fprintf(stderr,"Unable to read all of image data\n");
		exit(1);
	      }
	    image->rgb.r[i][j] = tmpdat[0];
	    image->rgb.g[i][j] = tmpdat[1];
	    image->rgb.b[i][j] = tmpdat[2];
	  }
      break;
      
    case '2': /* ASCII grayscale  */
      image->gray.i = allocate_pixel_array(image->rows,image->cols);
      for(i=0;i<image->rows;i++)
	for( j=0; j<image->cols; j++)
	  {
	    if(fscanf(inf,"%d",&tmp1)!=1)
	      {
		fprintf(stderr,"Unable to read all of image data\n");
		exit(1);
	      }
	    image->gray.i[i][j] = tmp1;
	  }
      break;

    case '4': /* binary grayscale */
      image->gray.i = allocate_pixel_array(image->rows,image->cols);
      for(i=0;i<image->rows;i++)
	if(fread(image->gray.i[i],sizeof(unsigned char),image->cols,inf)!=
	   image->cols)
	  {
	    fprintf(stderr,"Unable to read all of image data\n");
	    exit(1);
	  }
      break;
    }
  return image;
}


/* Write a netpbm image.  This function uses the magic number to
   determine how the data should be written. */
void writeImage(FILE *outf, netImage *image)
{
  int i,j;
  unsigned char tmpdat[3];
  if(fprintf(outf,"P%c\n",
	     image->magicNumber) < 1)
    {
      fprintf(stderr,"Unable to write to output file.  ");
      perror(NULL);
      exit(1);
    }
  if(image->comment != NULL)
    if(fprintf(outf,"%s",image->comment) < 1)
    {
      fprintf(stderr,"Unable to write to output file.  ");
      perror(NULL);
      exit(1);
    }
  if(fprintf(outf,"%d %d\n%d\n",
	     image->cols,image->rows, image->bpp) < 3)
    {
      fprintf(stderr,"Unable to write to output file.  ");
      perror(NULL);
      exit(1);
    }
  switch(image->magicNumber)
    {
    case '2':  /* ASCII grayscale */
      for(i=0;i<image->rows;i++)
	for(j=0;j<image->cols-1;j++)
	  if(fprintf(outf,"%d\n",image->gray.i[i][j])<1)
	    {
	      fprintf(stderr,"Unable to write to output file.  ");
	      perror(NULL);
	      exit(1);
	    }
      break;
    case '3':  /* ASCII color */
      for(i=0;i<image->rows;i++)
	for(j=0;j<image->cols;j++)
	  if(fprintf(outf,"%d %d %d\n",
		     image->rgb.r[i][j],
		     image->rgb.g[i][j],
		     image->rgb.b[i][j])< 3)
	    {
	      fprintf(stderr,"Unable to write to output file.  ");
	      perror(NULL);
	      exit(1);
	    }
      break;
    case '5':  /* binary grayscale */
      for(i=0;i<image->rows;i++)
	if(fwrite(image->gray.i[i],sizeof(unsigned char),image->cols,outf)!=
	   image->cols)
	  {
	    fprintf(stderr,"Unable to write to output file.  ");
	    perror(NULL);
	    exit(1);
	  }
      break;
    case '6':  /* binary color */
      for(i=0;i<image->rows;i++)
	for(j=0;j<image->cols;j++)
	  {
	    tmpdat[0] = image->rgb.r[i][j];
	    tmpdat[1] = image->rgb.g[i][j];
	    tmpdat[2] = image->rgb.b[i][j];
	    if(fwrite(tmpdat,sizeof(unsigned char),3,outf)!=3)
	      {
		fprintf(stderr,"Unable to write to output file.  ");
		perror(NULL);
		exit(1);
	      }
	  }
      break;
    default: /* error !*/
      fprintf(stderr,"Error trying to write unknown image type.\n");
      exit(1);
      break;
    }
}


