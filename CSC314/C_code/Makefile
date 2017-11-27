# This makefile will try to build the listed object files (OBJS) and
# link them into an executable named 'imagetool'.  For each object
# file in the list, it will try to find a C source file.  If it finds
# one, then it will compile it and use the resulting object file.
# Otherwise, it will look for an assembly source file.  If one is
# found, then it will assemble it and use the resulting object file.



CC = gcc
OBJS = main.o readwriteImage.o helpers.o\
	negateImage.o brightenImage.o sharpenImage.o \
	smoothImage.o grayscaleImage.o contrastImage.o 
INCLUDES = -I.
LIBDIRS = 
LIBS = 
DEFINES = 
CFLAGS = -g -Wall 
LFLAGS = -g 
DEPENDFLAGS = -M 



imagetool: $(OBJS)
	$(CC) $(CFLAGS) $(LFLAGS) -o $@ $(OBJS) $(LIBDIRS) $(LIBS)      

.c.o:
	$(CC) -c $(DEFINES) $(CFLAGS) $(INCLUDES) $<

.s.o:
	$(CC) -c $(DEFINES) $(CFLAGS) $(INCLUDES) $<

.S.o:
	$(CC) -c $(DEFINES) $(CFLAGS) $(INCLUDES) $<

clean:
	rm -f *.o imagetool

realclean: clean
	rm -f *~ .depend

# make depend will create a file ".depend" with all the dependencies
depend:
	rm -f .depend
	$(CC) $(DEFINES) $(INCLUDES) $(DEPENDFLAGS) $(OBJS:.o=.c) > .depend

# if we have a .depend file, include it

ifeq (.depend,$(wildcard .depend))
include .depend
endif
