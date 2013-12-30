from time import time
from numpy import *

nlist = [4, 5, 6, 7, 8];

start = time();
res = eval('tanh('+str(nlist)+')');
elapsed = time() - start;
print elapsed

start = time();
res = [eval('tanh('+str(i)+')') for i in nlist]
elapsed = time() - start;
print elapsed
