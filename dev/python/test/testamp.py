## initial test script for applying a multi-layer neural network on time series data
execfile('/omp/startup.py')

# ----- 
lr = 0.1   # learning rate
insz = 300  # input window size of the NN
nh1 = 40    # number of hidden units on layer 1
nh2 = 1    # number of hidden units on layer 2
nh3 = 1     # number of hidden units on layer 3
predhorizon = 100
plotint = 150
weightcost = 0.01
alpha = 1e2
beta = 1e3

# ----- define neural network -----
#l1 = amplayer(insz, nh1, nnsigmoid(), 0.01*randn(insz*(1+nh1)+nh1+1))
l1 = amplayermean(insz, nh1, nntanh(), 0.01*randn(insz*(nh1)+nh1))
#l1 = layer(insz, nh1, nntanh(), 0.01*randn(insz*(nh1)+nh1))
l2 = layer(nh1, nh2, nntanh(), 0.01*randn(nh1*nh2+nh2))
l3 = layer(nh2, nh3, nntanh(), 0.01*randn(nh2*nh3+nh3))

nn = net([l1, l2], 'mse'); #nn.gradcheck(1e-7, 1000)

# ----- define streamer -----
fn = '/omd/gen_sig_random.txt'
g = generator('default', 'default', 'default')
#sig = g.generate_signal(plotid='noplot', filename = fn)
stre = streamer(fn)

from nntrainer2 import *
# ----- define trainer -----
trainer = nntrainer2(nn, stre, lr, predhorizon, plotint, weightcost, alpha, beta)

# ----- start simulation -----
print 'start training neural network'
trainer.nnlearnstream('test')
