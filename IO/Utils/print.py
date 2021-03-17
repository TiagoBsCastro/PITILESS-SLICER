from datetime import datetime
# Adding time stamp to print
_print = print
def print (*args): 
      
     try: 
         _args = ["[", datetime.now(), "] "] + list(args) 
     except TypeError: 
         _args = ["[", datetime.now(), "] "] + [args] 
          
     _print(*_args)

