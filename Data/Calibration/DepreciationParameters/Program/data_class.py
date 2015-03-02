

import pandas as pd
import numpy as np


class pd_dfs:
    
    def __init__(self, *args):
        # The number of pd dfs to be stored.
        self.n = len(args)
        # Storing the pandas data frames.
        self.dfs = [pd.DataFrame]*len(args)
        for i in xrange(len(args)):
            self.dfs[i] = args[i]
    
    def append(self, *args):
        self.n += len(args)
        for i in xrange(len(args)):
            self.dfs.append(args[i])
    
    def delete(self, index=None): #code=None, name=None):
        del self.dfs[index]
        self.n -= 1


