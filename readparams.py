import re
import numpy as np

class ParameterNotFound (Exception):
   pass

class NotBooleanParameter (Exception):
   pass

def typeArrayFromString (dtype):
    '''
    Convert a sequence of number on a string to a given numeric type dtype
    '''
    return lambda s: np.fromstring(s, dtype=dtype, sep=' ')

def getValueFromFile (value,file,type):
    '''
    Search for a variable value inside the parameter string file and converts
    it to type
    '''
    matchedstring = re.search(r'^\s*'+value+'\s+(\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+|\d+\.+\s+\d+\.+\s+\d+\.|\d+\.\d+|\d+|\w+)',file,re.MULTILINE)
    if matchedstring == None:
       print("The parameter", value, "was not found")
       raise ParameterNotFound
    else:
       return type( matchedstring.group(1) )

def checkIfBoolExists (bool,file):
    '''
    Check if a boolean parameter bool is content in the string file.
    '''
    matchedstring = re.search(r'^\s*'+bool, file, re.MULTILINE)

    if matchedstring == None:

       matchedstring = re.search(r'^\s*\%\s*|^\s*\#\s*'+bool, file, re.MULTILINE)

       if matchedstring == None:

           print("Boolean option", bool, "not found")
           raise ParameterNotFound

       else:

           print("Boolean option", bool, "found but commented.")
           return False

    else:

        sanitycheck = re.search(r'^\s*'+bool+'(.*)', file, re.MULTILINE)

        if sanitycheck == None:

            return True

        elif re.search(r'\s*\%|\s*\#',sanitycheck.group(1)) != None:

            return True

        else:

            print("Parameter", bool, "Found but could not be interpreted as boolean!")
            raise NotBooleanParameter
