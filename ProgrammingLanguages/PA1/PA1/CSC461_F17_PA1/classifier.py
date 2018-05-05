'''
Defines the MDC class. The MDC takes finds the centroids of the data and
can then predict what set a point belongs to based on the nearest centroid.

Project: MDC - a minimum distance classifier
Authors: Noah Brubaker, Nathan Ducasse
Instructor: Dr. Weiss

MDC is instantiated with a list of classes and list of features. Data points are passed in when calling the ``train`` method, which calculates the centroids.

The MDC predicts what classes a given set of datapoints belong to using Minimum Distance Classification, or distance to the nearest centroid.
``
>>> mdc = MDC( data['classes'], data['features'] )
``

'''
import math
from functools import reduce
class Classifier:
    '''
    Base class for MDC. Children classes must implement the predict method.
    '''
    def __init__( self, data, classes ):
        self.data = data
        self.classes = classes

        
    def predict(self):
        raise(NotImplementedError())

        
        
class MDC(Classifier):
    '''
    Class for using minimum distance classification to guess the class of a set of data points
    '''
    def __init__( self, classes, features ):
        self.classes = classes
        self.features = features
        

    def train(self, data):
        '''
        Trains the classifier by finding the centroids for each class in the data set.
        '''
        datasections = [ list() for c in self.classes ]
        centroids = []
        
        # Make a sub-array for each class
        for i in range(len(data)):
            datasections[data[i][1]].append(data[i][2:])

        # Find centroid for each class, append to centroids
        for datasection in datasections:
            avg = []
            
            if(len(datasection) == 0):
                avg = [float("Inf")]*len(self.features)
                
            # Find centroid for current class
            for datapoint in zip(*datasection):
            
                datasum = 0
                for point in datapoint:
                    datasum += point
                    
                avg.append(datasum/len(datapoint))

            centroids.append(avg)

        self.centroids = centroids
       

    def predict(self, testpoints):
        '''
        Predicts the class for each of the given data points based on the closest centroid.
        '''
        minlist = []
        for testpoint in testpoints:
        
            # Finds the distance to each centroid and appends it to dist
            dist = []
            for centroid in self.centroids:
                temp = []
                
                # Get distance to current centroid
                for point, test in zip(centroid, testpoint):
                    tempdist = point-test
                    temp.append(tempdist**2)
                dist.append(math.sqrt(sum(temp)))
                
            # minidx is the index of the minimum distance. Index is the same as class number in this context.
            minidx = 0
            for i in range(len(dist)):
                if dist[i] < dist[minidx]:
                    minidx = i
            minlist.append(minidx)
        
        return minlist
