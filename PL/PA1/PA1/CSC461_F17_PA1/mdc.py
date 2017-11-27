'''
DOCS
'''
import sys, os
from csvio import reader
from preprocess import normalize
from classifier import MDC
from validator import LeaveOutOne

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = infile[:infile.rfind('.csv')] + '.cv'
    with open(infile) as f:
        data = reader(f)
    
    data['rows'] = normalize( data['rows'] )
    mdc = MDC( data['classes'], data['features'] )
    loocv = LeaveOutOne( mdc )
    loocv.validate(data['rows'])
    mdcoutput = 'MDC Parameters: nclasses = %d, nfeatures = %d, nsamples = %d' %(len(data['classes']), len(data['features']), len(data['rows']))
    print( data['title'] )
    print( mdcoutput )
    print( loocv.summary() )
    with open( outfile, 'w' ) as f:
        print( data['title'], file = f )
        print( mdcoutput, file = f )
        print( loocv.summary(), file = f )
        print(file = f)
        print( loocv.dump_results(), file = f )
