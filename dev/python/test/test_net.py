## implementation tacticals: 
#  - finish net class
#  - 
#  - 
#  - 
#  - 
#  - 
#  - 

from net import *

# l = layer(150, 20, 'tanh', random.randn(150*20+20));

# h = l.act(matrix(random.randn(150, 10)));

# g = l.bkprop(matrix(random.rand(20, 10)), h[1], random.randn(150, 10));

# create a net class
# initialize the net class [mark 12:55]

#p = random.randn(20*10+10);

l1 = layer(20, 10, 'tanh', random.randn(20*10+10));
l2 = layer(10, 5, 'tanh', random.randn(10*5+5));

n  = net([l1, l2]);

#D = matrix(random.randn(20, 10));
D = matrix(random.randn(20, 100));

#res = n.act(D);	
#print res[1][1][0]
#print res[1][1][1]
#print 'stop'
#G = n.bkprop(ones((5, 10)), res[1], D);
# [mark 2:38]
# implementing gradient function, parameter functions
# [mark 6:36]
# check gradient function, do gradient check

#F = n.gradfunc(p, D, T);
#[mark 7:10]
#n.gradcheck(1e-5);

# [mark 1:02]
# use one layer for grad checks

# act func tested OK
# set params, get params tested OK
# 
    
n.gradcheck(1e-7, 1e4);
