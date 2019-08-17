See Youtube video demo that does online learning and real-time time-series prediction with this system: 
https://www.youtube.com/watch?v=KJjRtnvNVXw 

[-------- ON-LINE PREDICTION & ANOMALY DETECTION SYSTEM ----------]

[Requirements]:
Python 2.7.3

*Numpy
*Scipy
*matplotlib
For Mac all contained in Superpack, simply run script:
https://raw.github.com/fonnesbeck/ScipySuperpack/master/install_superpack.sh

[dev/python/nn/ MODULES]:

net       #neural network class

streamer  #time-series streamer

nntrainer #online trainer with online predictor

nonlins   #neural network non-linearities

generator #signal(data) generator


[dev/python/test/ TEST SCRIPTS]:

testreg.py #run to test online-training + prediction + visualization
