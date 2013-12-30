#########################################################
# Set of non-linearities                                #
#########################################################

from numpy import *

class nonlin(object):
    def __init__(self, vec):
        return
    
    def __call__(self, vec):
        return self.activate(vec)
    
    def activate(self, vec):
        return vec
    
    def derivative(self, vec):
        return ones((vec.shape[0], vec.shape[1]))

class nntanh(nonlin):
    def __init__(self, vec):
        return activate(vec)
    
    def activate(self, vec):
        self.val = tanh(vec)
        return self.val
    
    def derivative():
        return
