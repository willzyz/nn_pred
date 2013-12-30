from numpy import *
from scipy import *
import sys
from pylab import *
import time

class generator(object): 
    ## ----- class signal generator -----
    ## this class has several functions: 
    ## - generate a train of delta functions with
    ##   - defined amplitude shifts (function)
    ##   - defined phase shifts (function)
    ##   - defined width changes (function)
    ## - offer a selected set of motifs
    ## - convolve delta train and selection of motifs to produce signal data
    
    def __init__(self, phase_function, amplitude_function, ano_type):
        if phase_function == 'fun':
            self.phase_function = self.fun_phase_function
        else:
            self.phase_function = self.default_phase_function
        
        if amplitude_function == 'fun':
            self.amplitude_function = self.fun_amplitude_function
        else:
            self.amplitude_function = self.default_amplitude_function
        
        self.ano_type = ano_type
        
        return
    
    def delta_train(self): 
        ## ----- generate a delta train with amplitude and phase patterns -----
        M = 2e5
        x = zeros((1, M)); x = x[0] # initial signal
        x = self.add_phase_patterns(x)
        x = self.add_amplitude_patterns(x)
        x = self.add_anomalies(x)
        return x
    
    def generate_signal(self, plotid='plot', filename='test.txt', amplitude_number_changes=4, amplitude_alpha=5, phase_number_changes=3): 
        ## ---- specify the number of changes in trends ----
        self.phase_number_changes = phase_number_changes
        self.amplitude_number_changes = amplitude_number_changes
        self.amplitude_alpha = amplitude_alpha
        
        ## ----- convolve the generated delta train with motifs to generate a signal ----- 
        f = self.generate_motifs()
        x = self.delta_train()
        print len(x)
        sig = convolve(x, f, 'valid')
        sig = sig + 0.5*std(sig) ## add in standard deviation of signal as dc
        if plotid == 'plot':
            figure; plot(sig); show(); time.sleep(2); close()
        self.write_to_file(sig, filename)
        
        return sig
    
    def generate_motifs(self):
        xi = arange(-5, 5, 0.1)
        f = self.gaussian_function(xi, 1, 1)
        #f2 = self.delta_function(xi, max(f))
        #res = concatenate((f, f2))
        #plot(res); show()
        res = f
        return res
    
    def add_phase_patterns(self, x): 
        ## ----- generate a phase function -----
        patf = self.phase_function(len(x), self.phase_number_changes)
        ## ----- add in phase patterns -----
        old_count = 120
        count = 0
        for i in range(len(x)):
            count += 1
            if count == old_count + 1:
                x[i] = 1
                old_count += patf[i]
                count = 0
        return x
    
    def add_amplitude_patterns(self, x): 
        ## ----- add in amplitude patterns -----
        patf = self.amplitude_function(len(x), self.amplitude_alpha, self.amplitude_number_changes)
        x = multiply(x, patf)
        return x
    
    def add_anomalies(self, x):
        M = max(x)
        if self.ano_type == None:
            return x
        elif self.ano_type == 'default':
            ## ----- add in anomalies -----
            for i in range(len(x)): 
                for tempmod in range(2000, 2080):
                    if mod(i, tempmod) == 0 and x[i] != 0:
                        a = randn()
                        if a>0:
                            b = 1
                        else:
                            b = -1
                        x[i] = x[i] + 0.25*M*b
                        if x[i] <0:
                            x[i] = 0
        return x
    
    def fun_amplitude_function(self, L): 
        return
    
    def default_a(self, *arg, **kwarg): 
        return self.default_amplitude_function(*arg, **kwarg)
    
    def default_amplitude_function(self, L, alpha, number_changes):
        ## ----- alpha: tangent -----
        ## ----- num_changes: number of tendency changes -----
        print 'generating amplitude function .....'
        res = []
        period = int(L)/int(number_changes) # integer division
        print 'period ='+str(period)
        for i in range(L): 
            m = i/period
            if mod(m, 2) == 0:
                a = alpha*i - alpha*period*m
            else:
                a = alpha*period*(m+1) - alpha*i
            a += alpha*period
            res.append(a)
        return res
    
    def fun_phase_function(self, L):
        return
    
    def default_p(self, *arg, **kwarg): 
        return self.default_phase_function(*arg, **kwarg)
        
    def default_phase_function(self, L, number_changes): 
        res = []
        period = int(L)/int(number_changes)
        for i in range(L):
            m = i/period
            if mod(m, 2) == 0:
                a = 1
            else:
                a = -1
            res.append(a)
        return res
    
    def gaussian_function(self, vec, amplitude, sigma):
        return amplitude/sigma/sqrt(2*pi)*exp(-divide(square(vec), square(sigma)))
    
    def delta_function(self, vec, amplitude):
        res = []
        for v in vec:
            if v < 0:
                a = v + 1*amplitude
                if a > 0:
                    res.append(a)
                else:
                    res.append(0)
            else:
                a = 1*amplitude-v
                if a > 0:
                    res.append(a)
                else:
                    res.append(0)
        return array(res)
    
    def write_to_file(self, sig, filename):
        print 'writing to file: '+filename
        directory = '/omd/'
        f = open(directory + filename, 'w')
        for i in range(len(sig)):
            if mod(i, 1000) == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            f.write(str(sig[i])+'\n')
        f.close()
