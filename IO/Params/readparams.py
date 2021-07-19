import re
import numpy as np
from IO.Utils.print import print

class ParameterNotFound (Exception):
   pass

class NotBooleanParameter (Exception):
   pass

def typeArrayFromString (dtype):
    '''
    Convert a sequence of number on a string to a given numeric type dtype
    '''
    return lambda s: np.fromstring(s, dtype=dtype, sep=' ')

def getValueFromFile (value,file,type,rank=0):
    '''
    Search for a variable value inside the parameter string file and converts
    it to type
    '''
    # Case 0 it is an 3d array declared like +/- X[3]
    typing0 = r'\-?\d+\s+\-?\d+\s+\-?\d+|'
    # Case 1 it is an 3d array declared like +/- X.Y[3] or +/- X. for short
    typing1 = r'\-?\d+\.\d*\s+\-?\d+\.\d*\s+\-?\d+\.\d*|'
    # Case 2 it is an float declared like +/- X.Y or +/- X. for short
    typing2 = r'\-?\d+\.\d*|'
    # Case 3 it is an integer declared like +/- X
    typing3 = r'\-?\d+|'
    # Case 4 it is a string
    typing4 = r'.*?\S*'
    typings = typing0 + typing1 + typing2 + typing3 + typing4
    '(\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+|\d+\.+\s+\d+\.+\s+\d+\.|\d+\.\d+|\d+|\w+'
    matchedstring = re.search(r'^\s*'+value+'\s+('+typings+')',file,re.MULTILINE)
    if matchedstring == None:
       print("The parameter", value, "was not found",rank=rank)
       raise ParameterNotFound
    else:
       return type( matchedstring.group(1) )

def checkIfBoolExists (bool,file,rank=0):
    '''
    Check if a boolean parameter bool is content in the string file.
    '''
    matchedstring = re.search(r'^\s*'+bool, file, re.MULTILINE)

    if matchedstring == None:

       matchedstring = re.search(r'^\s*\%\s*|^\s*\#\s*'+bool, file, re.MULTILINE)

       if matchedstring == None:

           print("Boolean option", bool, "not found",rank=rank)
           return False

       else:

           print("Boolean option", bool, "found but commented.",rank=rank)
           return False

    else:

        sanitycheck = re.search(r'^\s*'+bool+'(.*)', file, re.MULTILINE)

        if sanitycheck == None:

            return True

        elif re.search(r'\s*\%|\s*\#|\s*',sanitycheck.group(1)) != None:

            return True

        else:

            print("Parameter", bool, "Found but could not be interpreted as boolean!",rank=rank)
            raise NotBooleanParameter
