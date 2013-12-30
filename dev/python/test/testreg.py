## initial test script for applying a multi-layer neural network on time series data
execfile('startup.py')

## Directory containing test data
STREAMER_DATA_DIR = '../../../data'
if 'STREAMER_DATA_DIR' in os.environ:
    STREAMER_DATA_DIR = os.environ['STREAMER_DATA_DIR']

# ----- define signal generator -----
g = generator('default', 'default', 'default')

# ----- define streamers, one for each time series ------
stre = [streamer(os.path.join(STREAMER_DATA_DIR, 'A5M.txt'))]

# To define multi-dimensional time-series:
# stre = [streamer([os.path.realpath('../../../data/'), 'mult_test1.txt']), streamer([os.path.realpath('../../../data/'), 'mult_test2.txt']), streamer([os.path.realpath('../../../data/'), 'mult_test3.txt'])]

reglen = 10; # length of future time series the model tries to predict during training

# ----- model parameters -----
indim = len(stre)
contextlen = 500
insz = contextlen*indim  # input window size of the NN
nh1 = 100                # number of hidden units on layer 1
nh2 = 50       # number of hidden units on layer 2
nh3 = indim*reglen       # number of hidden units on layer 3
predhorizon = 50
plotint = 150
weightcost = 0.01
alpha = 1e2
beta = 1e3
quiet_steps = 5e2

# ----- define neural network -----
l1 = layer(insz, nh1, nntanh(), 0.01*randn(insz*nh1+nh1))
l2 = layer(nh1, nh2, nntanh(), 0.01*randn(nh1*nh2+nh2))
l3 = layer(nh2, nh3, nntanh(), 0.01*randn(nh2*nh3+nh3))

nn = net([l1, l2, l3], 'mse'); #nn.gradcheck(1e-7, 50)

# ----- define trainer -----
trainer = nnstreamtrainer(nn, stre, contextlen, reglen, predhorizon, plotint, weightcost, alpha, beta, quiet_steps)

# ----- start simulation -----
trainer.nnlearnstreamreg('plot')
