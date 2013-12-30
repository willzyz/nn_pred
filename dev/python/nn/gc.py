# gradcheck script
execfile('/omp/startup.py')

# ----- 
lr = 0.1   # learning rate
insz = 50  # input window size of the NN
nh1 = 20    # number of hidden units on layer 1
nh2 = 10     # number of hidden units on layer 2
nh3 = 1     # number of hidden units on layer 3
predhorizon = 150; 
plotint = 25; 
weightcost = 0.01

# ----- define neural network -----
l1 = layer(insz, nh1, nntanh(), 0.01*random.randn(insz*nh1+nh1))
l2 = layer(nh1, nh2, nntanh(), 0.01*random.randn(nh1*nh2+nh2))
l3 = layer(nh2, nh3, nntanh(), 0.01*random.randn(nh2*nh3+nh3))

nn = net([l1, l2, l3], 'mse'); 

#t = l1(random.randn(insz, 10))

d = randn(insz, 10)
t = nn(d)
G = nn.back_propagate(randn(1, 10), d)

#test fwpredict
d = randn(1, insz)
d = list(d[0])
pred = nn.fwpredict(d, 10)

[f, g] = nn.gradfunc(nn.getparams(), randn(insz, 10), randn(1, 10))

nn.gradcheck(1e-5, 50)
