## [----- Apcera Omnis ------]
##  initial test script for applying a 1-layer neural network on time series data

import sys
from net import *
from nntrainer import *
from streamer import *
from numpy import *

lr = 1e-3
insz = 100
nh1 = 20
nh2 = 1
l1 = layer(insz, nh1, 'tanh', 0.01*random.randn(insz*nh1+nh1))
l2 = layer(nh1, nh2, 'tanh', 0.01*random.randn(nh1*nh2+nh2))

nn  = net([l1, l2])

trainer = nntrainer(nn, lr)
fn = '/home/wzou/Desktop/Apcera/data/A5M.txt'

print 'start training neural network'
sys.stdout.flush()
stre = streamer(fn)
xt = stre.onestep()
count = 0
while xt != 'full':
    count += 1
    trainer.streamstep(xt)
    print 'sgd step '+ str(count)
    sys.stdout.flush()
    xt = stre.onestep()

trainer.nn.plotweights(trainer.cur_params);
