#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include <netImage.h>

/* The C library has some functions to help parse the command line.
   To use these functions, we first define an array of 'struct option'
   data, which specifies the options that are supported by the
   program.
 */
struct option options[] = {
  {"negate", 0, NULL, 'n'},
  {"brighten", 1, NULL, 'b'},
  {"sharpen", 0,  NULL, 'p'},
  {"smooth", 0, NULL, 's'},
  {"grayscale", 0,  NULL, 'g'},
  {"contrast", 0,  NULL, 'c'},
  {"output", 1, NULL, 'o'},
  {"help", 0, NULL, 'h'},
  {0}
};

/* this is an array of strings used by the usage() function to
   describe the options listed above. */
char *description[] = {
  "    Create a negative image.",
  "    Brighten the image.  The required argument must be a\n"
  "    number between -128 and 127 specifying the amount by\n"
  "    which the brightness should be changed.",
  "    Sharpen the image.",
  "    Smooth the image.",
  "    Convert to grayscale.",
  "    Maximize contrast. (converts image to grayscale)",
  "    Specify output in binary or ASCII. arg must be either\n"
  "    'a' or 'b'. The default is to output in same format as\n"
  "    the input.",
  "    Display this help screen.",
  "",
  ""
};



void usage(char *name)
{
  int i;
  printf("\nUSAGE: ");
  printf("%s <commands> input_file output_file\n",name);
  printf("  where a command is one of:\n\n");
  for(i=0;options[i].name!=NULL;i++)
    {
      if(options[i].has_arg==1)
	printf("  --%s <arg> or -%c <arg>\n",options[i].name, options[i].val );
      else
	printf("  --%s or -%c\n",options[i].name, options[i].val );
      printf("%s\n\n",description[i]);
    }
  printf("You can only use ONE of the following: -n, -b, -p, -s, -g, -c\n");
  exit(1);
}


/* main() parses the command line options, calls the function to
   perform the requested operation, then writes the new image to the
   output file.
*/

int main(int argc, char **argv)
{
  int c,i;
  op_t op = NOTHING;    // don't perform any operation by default
  out_t output_type = DEFAULT; // don't change output format by default
  FILE *ofile,*ifile;   // input and output files
  int factor;           // factor used with brightness
  netImage *image;      // the image to be processed
  int option_index = 0; // tracks the current argument while parsing

  if(argc < 3)     // we must have at least an input file and an output file
    usage(argv[0]);

  // process command line options, one at a time.  -b and -o require
  // arguments, so they are followed by ':' in the list.
  while((c = getopt_long (argc, argv, "nb:psgco:h", options,
  			  &option_index)) >= 0)
    {
      switch(c)
        {
        case 'n':
	  if(op != NOTHING)
	    usage(argv[0]);
	  op = NEGATE;
          break;
        case 'b':
	  if(op != NOTHING)
	    usage(argv[0]);
	  op = BRIGHTEN;
	  if(!isdigit(optarg[0]) && optarg[0]!='-')
	    {
	      fprintf(stderr,"-b or --brighten must "
		      "be followed by a number.\n");
	      exit(1);
	    }
	  for(i=1;i<strlen(optarg);i++)
	    if(!isdigit(optarg[i]))
	      {
		fprintf(stderr,"-b or --brighten must "
			"be followed by a number.\n");
		exit(1);
	      }
	  factor = atoi(optarg);
	  if(factor < -128 || factor > 127)
	      {
		fprintf(stderr,"Brightness factor must be between "
			"-128 and 127.\n");
		exit(1);
	      }
	  break;

	case 'p':
	  if(op != NOTHING)
	    usage(argv[0]);
	  op = SHARPEN;
          break;

        case 's':
	  if(op != NOTHING)
	    usage(argv[0]);
	  op = SMOOTH;
          break;

        case 'g':
	  if(op != NOTHING)
	    usage(argv[0]);
	  op = GRAYSCALE;
          break;

        case 'c':
	  if(op != NOTHING)
	    usage(argv[0]);
	  op = CONTRAST;
          break;

        case 'o':
	  if(!strcmp(optarg,"a"))
	    output_type = ASCII;
	  else if(!strcmp(optarg,"b"))
	    output_type = BINARY;
	  else
	    fprintf(stderr,"Unknown output format: -o%s\nUse -oa or -ob\n",
		    optarg);
	  break;
	  
        case '?':
        case 'h':
        default:
          usage(argv[0]);
          break;
        }
    }


  /* Try to open the input file */
  if((ifile = fopen(argv[argc-2],"r")) == NULL)
    {
      fprintf(stderr,"%s: Unable to open input file '%s'.  ",
	      argv[0],argv[argc-2]);
      perror(NULL);
      usage(argv[0]);
    }

  image = readImage(ifile);

  fclose(ifile);

     /* Try to open the output file */
  if((ofile = fopen(argv[argc-1],"w")) == NULL)
    {
      fprintf(stderr,"%s: Unable to open output file '%s'.  ",
	      argv[0],argv[argc-1]);
      perror(NULL);
      usage(argv[0]);
    }

  if(output_type == ASCII)
    {
      if(image->magicNumber == '6')
	image->magicNumber = '3';
      else
	if(image->magicNumber == '5')
	  image->magicNumber = '2';
    }
  else if(output_type == BINARY)
    {
      if(image->magicNumber == '3')
	image->magicNumber = '6';
      else
	if(image->magicNumber == '2')
	  image->magicNumber = '5';
    }
  
  switch(op)
    {
    case BRIGHTEN:
      brightenImage(image,factor);
      break;
    case NEGATE:
      negateImage(image);
      break;
    case SHARPEN:
      sharpenImage(image);
      break;
    case SMOOTH:
      smoothImage(image);
      break;
    case GRAYSCALE:
      grayscaleImage(image);
      break;
    case CONTRAST:contrastImage(image);
      break;
    default:
      break;
    }
  
  writeImage(ofile, image);
  
  fclose(ofile);

  return 0;
}

