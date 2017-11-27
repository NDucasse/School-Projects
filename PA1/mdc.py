import sys
import csv

def Main(argc, argv):
    '''
    Main method, gets input data from files and prints output data
    to command line and output file. Uses input data to learn and
    extrapolate to classify new data.
    '''
    # Ensure correct number of command args
    if argc < 2 or argc > 2:
        return

    # Create variables for file manipulation
    out = open(argv[1].split('.')[0] + '.cv', 'w')
    filedata = GetInput(argv[1]);
    
    # Create variables for specific data
    info = filedata[0][1] # The name of the dataset
    classes = filedata[0][1:] # The different class names
    stypes = filedata[1] # The sample types
    samples = filedata[2:] # The list of samples

    # Get class names from the first line
    for i in range(0, len(classes)):
        classes[i] = [classes[i][:classes[i].find('=')], classes[i][classes[i].find('=')+1:]]
    
    # print(classes)
    # Send the data to the learning algorithm
    # Learn(classes, stypes, samples)
    
    CorrectData(samples[1:])
    
    # Figure out extrapolating later
    # Extrapolate(classes, stypes, samples)
    
    # Output information
    print(info)
    print(info, file=out)

    # Close file i/o
    out.close()



def GetInput(filename):
    filedata = []
    file = open(filename, 'r')
    read = csv.reader(file)
    
    # make csv into list of lines
    for line in read:
        filedata.append(line)  
   
    return filedata
    
    
    
def Learn(classes, stypes, samples):
    classdata = []
    centroids = [[]]
    for classinfo in classes:
        classdata.append([])

    CorrectData(samples[1:])
    
    for sample in samples:
        classdata[int(sample[1])].append(sample)
    print(classdata[0])
    
    
    
    pass

   
   
def CorrectData(sampleset):
    samples = []
    sampleset = list(zip(*sampleset))
    
    for i in range(int(sampleset[1][len(sampleset[1]-1))):
        samples[0].append([])
    
    for sample in sampleset:
        samples[
    print(sampleset)
    
    
def Extrapolate(classes, stypes, samples):
    pass


   
if __name__ == '__main__':
    Main(len(sys.argv), sys.argv)