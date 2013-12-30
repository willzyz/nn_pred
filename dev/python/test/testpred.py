## [----- Apcera Omnis ------]
## initial test script for applying a multi-layer neural network on time series data
execfile('/omp/startup.py')

# ----- 
lr = 0.1   # learning rate
insz = 500  # input window size of the NN
nh1 = 40    # number of hidden units on layer 1
nh2 = 20     # number of hidden units on layer 2
nh3 = 1     # number of hidden units on layer 3
predhorizon = 150 
plotint = 500
weightcost = 0.01
amplitude_alpha = None
alpha = 1e2
beta = 1e3

# ----- define neural network -----
l1 = layer(insz, nh1, nntanh(), 0.01*randn(insz*nh1+nh1))
l2 = layer(nh1, nh2, nntanh(), 0.01*randn(nh1*nh2+nh2))
l3 = layer(nh2, nh3, nntanh(), 0.01*randn(nh2*nh3+nh3))

nn = net([l1, l2, l3], 'mse'); #nn.gradcheck(1e-7, 50)

# ----- define streamer -----
fn = 'gen_sig_'+str(amplitude_alpha)+'.txt'
g = generator()
g.gen_signal(amplitude_alpha, 'plot', fn)
stre = streamer(fn)

# ----- define trainer -----
trainer = nntrainer(nn, stre, lr, predhorizon, plotint, weightcost, alpha, beta)

# ----- start simulation -----
print 'start training neural network'
trainer.nnlearnstream('test')
