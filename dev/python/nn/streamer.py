import sys
from numpy import *

#####################################
# Streamer Class:               
# currently simulates streams of data
# later streams time series from TSDB
#####################################

class streamer: 
    def __init__(self, fn):
        self.fn = fn                                # 
        self.id = fn                                # time series id
        self.series = []
        self.streamcount = 0
        print '----- open & read data file -----'
        self.file = open(fn, 'r')
        count = 1
        while 1:
            if mod(count, 1000)==0:
                sys.stdout.write('.')
                sys.stdout.flush()
            line = self.file.readline()
            if not line:
                break
            if mod(count, 1)==0:
                if not ':' in line:
                    a =  line.split('\r')[0]
                    if ',' in a:
                        a = a.split(', ')[1]
                    self.series.append(float(a))
            count += 1
        print '\n'
        self.series.pop()
        self.series = (array(self.series)/max(self.series) - 0.5)*2
        
    def __call__(self):
        for a in self.series:
            print a
            
    def get_element(self):
        if self.streamcount<len(self.series):
            xt = self.series[self.streamcount-1]         
            
            self.streamcount = self.streamcount+1
            return xt #(xt - self.meanest)/(self.stdest + 1e-10)
        else:
            return 'full'
