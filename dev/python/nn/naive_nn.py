## [----- Apcera Omnis ------]
## initial test script for applying a 1-layer neural network on time series data

import sys
from net import *
from nntrainer import *
from streamer import *
import pylab
from pylab import *
from numpy import *
import matplotlib.pyplot as plt

# ----- 
lr = 1e-2   # learning rate
insz = 100  # input window size of the NN
nh1 = 40    # number of hidden units on layer 1
nh2 = 1     # number of hidden units on layer 2
nh3 = 1     # number of hidden units on layer 2

# ----- define neural network -----
l1 = layer(insz, nh1, 'tanh', 0.01*random.randn(insz*nh1+nh1))
l2 = layer(nh1, nh2, 'tanh', 0.01*random.randn(nh1*nh2+nh2))
l3 = layer(nh2, nh3, 'tanh', 0.01*random.randn(nh2*nh3+nh3))

nn  = net([l1, l2]); #nn.gradcheck(1e-7, 50)

# ----- define streamer -----
fn = '/omd/A5M.txt'
stre = streamer(fn)

# ----- define trainer -----
trainer = nntrainer(nn, stre, lr)

# ----- start simulation -----
print 'start training neural network'
trainer.nnlearnstream()

# ----- visualize learned weights -----
trainer.nn.plotweights(trainer.cur_params);
