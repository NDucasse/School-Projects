'''
Has function reader( file f )
reads file format from a config 
returns
data = {
    title,
    classes,
    features,
    rows
}
'''

import csv
def reader(f):
    '''
    Takes in a file and parses it, formatting the data.
    '''
    data = {}
    read = list(csv.reader(f, delimiter=','))
    
    
    data['title'] = read[0][0].strip()\

    # Separates the classes into the class number and the name
    data['classes'] = {}
    for item in read[0][1:]:
        data['classes'][int(item[0:item.find('=')])] = item[item.find('=')+1:].strip()
        
    data['features'] = read[1][2:]
    # Converts data points from strings to ints or floats
    data['rows'] = [[(int(point) if '.' not in point else float(point)) for point in item] for item in read[2:]]
    
    return data
