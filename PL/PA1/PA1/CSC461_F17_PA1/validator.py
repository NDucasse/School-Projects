'''defines a class which does leave-out-one validation on a Classifier and stores the results internally

Project: MDC - a minimum distance classifier
Authors: Noah Brubaker, Nathan Ducasse
Instructor: Dr. Weiss

A validator is instantiated with the instance of the classifier to be validated. A call to the ``validate`` method will generate the results and store them internally, in addition to an optional return value.

The leave-out-one cross-validation method trains the classifier on all but one point in the test set and predicts the class of the point that was left out. 

This module demonstrates how python can be used to implement a general validation tool for a classifier, it is not intended for large-scale use.

``
>>> mdc = MDC( classes, features )
>>> loocv = LeaveOutOne( mdc )
>>> loocv.validate( data )
>>> loocv.overall
>>> loocv.accuracy
>>> loocv.results[0]
``

'''
class Validator:
    '''
    Base class for the validation of a classifier. Children must implement the validate method.
    '''
    def __init__(self):
        self.classes = None
        self.accuracy = None
        self.overall = None
        self.results = None
        
    def validate(self, data, * , train_args = {}, predict_args = {}, finish_all = True ):
        raise(NotImplementedError())
            
class LeaveOutOne(Validator):
    '''a leave out one validation object for a classifier
    
    Instantiated with a classifer. The validate method requires a 2D array ``data`` used to validate the classifier.
    A summary method is provide for convenience.
    ``
    >>> loocv.summary()
    >>> loocv.dump_results()
    ``
    '''
    def __init__(self, classifier ):
        self.results = None
        self.classifier = classifier
        
    def validate(self, data, * , train_args = {}, predict_args = {}, finish_all = True ):
        '''
        performs leave-out-one cross validation on the classifier instance using ``data``
        '''
        classes = self.classifier.classes
        nclasses = len(classes)
        ndata = len(data)
        self.results = [None] * ndata
        self.counts = dict( zip( classes.keys(), [0]*nclasses ) )
        self.correct = dict( zip( classes.keys(), [0]*nclasses ))
        
        # Leave out the ith element and train the classifier on the rest,
        # predict on element that has been removed, update results
        dcopy = data.copy()
        for i in range( ndata ):
            x = dcopy.pop(i)
            self.classifier.train(dcopy, **train_args )
            p = self.classifier.predict( [x[2:],], **predict_args )[0]
            c = x[1]
            self.results[i] = tuple( x[:2] + (p,) )
            self.counts[ c ] += 1
            self.correct[ c ] += 1 if c == p else 0
            dcopy.insert(i,x)
            
        # By default, train on all data so the classifier is in best possible state
        if finish_all: 
            self.classifier.train(data)
            
        self.accuracy = [ 100 * self.correct[k] / self.counts[k] if self.counts[k] > 0 else None for k in classes ]

        self.overall = len(data), 100*sum(self.correct.values())/len(data)
        return self.overall
        
    def summary(self):
        '''
        prints a summary of the results of the validation
        '''
        if self.results is None:
            raise( UserWarning('There are no results.') )
            return ''
            
        lines = list()
        for k,c in sorted(self.classifier.classes.items()):
            if self.counts[ k ] != 0:
                lines.append( 'class {} ({}): {} samples, {:.1f}% accuracy'.format(k, c, self.counts[ k ], self.accuracy[ k ]) )
        lines.append( 'overall: {} samples, {:.1f}% accuracy'.format( *self.overall ) )
        return '\n'.join(lines)
    
    def dump_results(self):
        '''
        dumps raw results of the validation
        '''
        if self.results is None:
            raise( UserWarning('There are no results.') )
            return ''
            
        head = 'Sample,Class,Predicted'
        lines = [ head, ]
        for r in self.results:
            lines.append( '{},{},{}'.format(*r) )
            lines[-1] += ' *' if r[1] != r[2] else ''
        return '\n'.join( lines )
        
