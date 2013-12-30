from net import *
import pickle
from pylab import *
import copy as cp

class nnstreamtrainer(object):
    def __init__(self, nn, streamers, contextlen, reg_tail_len, predhorizon, plotint, weightcost, alpha, beta, run_quiet_steps): 
        self.nn = nn                                # store neural network
        self.streamers = streamers                  # store streamers
        self.context = None                         # series context store
        self.contextlen = contextlen                # series context length = nn input size
        self.context_state = 'init'
        self.reg_tail = None
        self.reg_tail_len = reg_tail_len
        self.reg_tail_state = 'init'
        self.cur_params = self.nn.getparams()       # current state parameters initialization
        self.init_params = self.nn.getparams()      # store initial params
        self.streamcount = 0                        # count of streaming steps from data series
        self.sgdstep = 0                            # count of sgd steps
        self.estf = 0                               # current estimation of objective value
        self.estimated_objective_list = []                          # full list of obj estimations
        self.running_objective_list = []                           # [mean/std est] running list of objective value        
        self.running_objective_length = 2000                          # [mean/std est] length of running list
        self.sgd_save_interval = 1000                      # how many sgd steps before saving
        self.predhorizon = predhorizon
        self.predlist = []
        self.predrunlist = []
        self.errlist = []
        self.predict_err_horizon = 1 #pred_err_horizon
        self.plotint = plotint
        self.weightcost = weightcost
        # alpha and beta are parameters for changing step-length during sgd: learning rate = alpha/(beta + iteration)
        self.alpha = alpha # 6e2
        self.beta = beta #6e3
        self.run_quiet_steps = run_quiet_steps
        self.series_dim = size(streamers)
        
    def multi_d_get_element(self): 
        # -- obtain a multi-dimensional stream element from list of streamers --
        series_end = False
        element = []
        for streamer in self.streamers:
            x = streamer.get_element()
            if x == 'full':
                series_end = True
            element.append(x)
        element = matrix(element).T
        return [element, series_end]
    
    def append_to_context(self, incontext, contextlen, element): 
        # --- initial None ---
        if incontext == None:
            context = element
            return [context, 'init']
        context = concatenate((incontext, element), axis=1)
        if context.shape[1]>contextlen:
            context = context[:, 1:context.shape[1]]
            return [context, 'full']
        return [context, 'filling']
        
    def gradient_descend(self, target): 
        [f, g] = self.nn.gradfunc(self.cur_params, self.context.reshape(1, []).T, target)
        # -- add regularization --
        f = f + 0.5*self.weightcost*sum(square(self.cur_params))
        g = g + self.weightcost*self.cur_params
        self.cur_params = self.cur_params - self.alpha/(self.beta + self.sgdstep)*g
        self.running_objective_list.append(f)
        if len(self.running_objective_list)>self.running_objective_length:
            self.running_objective_list.pop(0)
        estf = mean(self.running_objective_list)
        print 'sgd step '+ str(self.sgdstep) + '  objective value: ' + str(estf)
        self.estimated_objective_list.append(estf)
        self.sgdstep += 1
        
    def stream_step(self): 
        # -- gets element, gradient descend, append to context --
        [element, series_end] = self.multi_d_get_element()
        if series_end:
            return series_end
        if self.context_state == 'full':
            self.gradient_descend(element)
        [self.context, self.context_state] = self.append_to_context(self.context, self.contextlen, element)
        self.streamcount += 1
        return series_end
    
    def stream_step_reg(self):
        # -- update step: update context, update reg_tail, gradient_descend on condition -- 
        [element, series_end] = self.multi_d_get_element()
        if series_end:
            return series_end
        if self.context_state == 'full' and self.reg_tail_state == 'full':
            inter = cp.deepcopy(self.reg_tail[:, 0])
            [self.reg_tail, self.reg_tail_state] = self.append_to_context(self.reg_tail, self.reg_tail_len, element)
            [self.context, self.context_state] = self.append_to_context(self.context, self.contextlen, inter)
            self.gradient_descend(self.reg_tail.reshape(size(self.reg_tail), 1))
        elif self.context_state == 'full': # and reg_stail is half full
            # fill reg_tail
            [self.reg_tail, self.reg_tail_state] = self.append_to_context(self.reg_tail, self.reg_tail_len, element)
        else: #neither is full, reg_tail should be empty
            assert self.reg_tail == None
            [self.context, self.context_state] = self.append_to_context(self.context, self.contextlen, element)        
        self.streamcount += 1
        return series_end
    
    def forward_predict(self): 
        temp_context = cp.deepcopy(self.context)
        predictions = None
        #print temp_context
        for i in range(self.predhorizon):
            h = self.nn.activate(temp_context.reshape(1, []).T)
            h = h.reshape(self.series_dim, []) # reshape into sdim X reg_tail_len
            [temp_context, dummy] = self.append_to_context(temp_context, self.contextlen, h[:, 0])
            [predictions, dummy] = self.append_to_context(predictions, 1e5, h[:, 0])
        #print temp_context
        #print predictions
        #predictions = self.nn.activate(self.context.reshape(1, []).T).T
        return predictions
    
    def predict_and_measure(self):
        self.predlist = self.forward_predict()
        # store list of predictions
        #print self.predict_err_horizon-1
        #print self.predlist.shape
        #print self.predlist[:, self.predict_err_horizon-1]
        self.predrunlist.append(self.predlist[:, self.predict_err_horizon-1])
        if len(self.predrunlist) > self.predict_err_horizon+1:
            self.predrunlist.pop(0)
            # retract prediction compare with context
            temp_vec = self.context[:, len(self.context)-1] - self.predrunlist[len(self.predrunlist)-self.predict_err_horizon-1]
            temp_vec = abs(temp_vec)/2
            curerr = sum(temp_vec)/temp_vec.shape[1]
            self.errlist.append(curerr)


    #def nnlearnstream(self, plotid):
    #    print '----- start training neural network -----'
    #    self.plotid = plotid
    #    if plotid == 'plot':
    #        self.initialize_plots()
    #    self.sgdstep = 0
    #    series_end = False
    #    while not series_end:
    #        series_end = self.stream_step()        
    #        if mod(self.sgdstep, self.sgd_save_interval) == 0:
    #            self.save_progress_to_file()
    #        self.predict_and_measure()
    #        if self.sgdstep>self.run_quiet_steps and plotid == 'plot' and (mod(self.sgdstep, self.plotint)==0) and self.context_state=='full' and len(self.estimated_objective_list)>0:
    #            self.update_plots()
    #    print '---- the series has ended woohoo ----'
        
    def nnlearnstreamreg(self, plotid):
        print '----- start training neural network -----'
        self.plotid = plotid
        if plotid == 'plot':
            self.initialize_plots()
        self.sgdstep = 0
        series_end = False
        while not series_end:
            series_end = self.stream_step_reg()
            if mod(self.sgdstep, self.sgd_save_interval) == 0:
                self.save_progress_to_file()
            if self.sgdstep>self.run_quiet_steps:
                self.predict_and_measure()
                if plotid == 'plot' and (mod(self.sgdstep, self.plotint)==0) and self.context_state=='full' and len(self.estimated_objective_list)>0:
                    self.update_plots()
        print '---- the series has ended woohoo ----'
    
    def update_plots(self):
        # --- plot optimization curve ---
        self.ax[0].set_title('optimization curve sgd iteration = '+str(self.sgdstep))
        self.optimization_line.set_xdata(arange(len(self.estimated_objective_list)))
        self.optimization_line.set_ydata(self.estimated_objective_list)
        self.ax[0].axis([0, len(self.estimated_objective_list), 0, max(self.estimated_objective_list)])
        # --- plot context, prediction and history lines ---        
        for i in range(len(self.streamers)): 
            self.context_lines[i][0].set_xdata(arange(len(self.context[i].tolist()[0])))
            self.context_lines[i][0].set_ydata(self.context[i].tolist()[0])        
            self.prediction_lines[i][0].set_xdata(arange(self.context[i].shape[1], self.context[i].shape[1]+self.predlist[i].shape[1]))
            self.prediction_lines[i][0].set_ydata(self.predlist[i].tolist()[0])
            self.history_lines[i][0].set_xdata(arange(self.sgdstep))
            self.history_lines[i][0].set_ydata(self.streamers[i].series[0:self.sgdstep])
                    
        self.ax[4].axis([0, self.sgdstep, -1, +1])
        self.test_error_line.set_xdata(arange(len(self.errlist)))
        self.test_error_line.set_ydata(self.errlist)
        
        if len(self.errlist)>0:
            self.ax[5].axis([0, len(self.errlist), min(self.errlist), max(self.errlist)])
            self.ax[5].set_title('absolute percentage error '+str(mean(self.errlist))+' for '+str(self.predict_err_horizon)+' steps in future')
        
        if mod(self.sgdstep, self.plotint*3) == 0:
            self.ax[2].imshow(self.nn.layers[0].W); self.ax[2].axes.get_xaxis().set_visible(False); self.ax[2].axes.get_yaxis().set_visible(False)
            self.ax[3].imshow(self.nn.layers[1].W); self.ax[3].axes.get_xaxis().set_visible(False); self.ax[3].axes.get_yaxis().set_visible(False)
        
        draw()
        
    def initialize_plots(self):
        ion()
        # ----- make a number of plots and pop up a sized figure -----
        numplots = 6;
        figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
        self.ax = []
        for i in range(numplots):
            self.ax.append(subplot(numplots, 1, i + 1))
            self.ax[i].grid('on')
        
        self.ax[0].set_ylabel('obj value')
        self.ax[1].set_ylabel('ts value')
        self.ax[1].axis([0, self.contextlen+self.predhorizon, -1, +1])
        
        self.optimization_line, = self.ax[0].plot([], [])
        self.test_error_line, = self.ax[5].plot([], [], 'r')
        
        #self.context_lines, = self.ax[1].plot([], [])
        self.context_lines = []                
        self.prediction_lines = []
        self.history_lines = []
        for i in range(len(self.streamers)):
            self.context_lines.append(self.ax[1].plot([], []))
            self.prediction_lines.append(self.ax[1].plot([], [], 'r'))
            self.history_lines.append(self.ax[4].plot([], []))
         
    def save_progress_to_file(self):
        # ----- NOT SURE WHAT IS WRONG HERE WITH PICKLE -----
        #print self.gensavename()
        #print self
        #with open(self.gensavename(), 'wb') as output:
        #    pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return
    
    def gensavename(self):
        strname = '/oms/trainer_'
        # stream TS id
        #for streamer in self.streamers:
        #    strname+=streamer.id.split('.')[0].split('omd/')[1]
        #    strname+='_'
            
        # neural network nh1 nh2 ...
        strname+='nh'
        for i in range(len(self.nn.layers)):
            strname=strname + str(self.nn.layers[i].insz) + '_'
        
        strname = strname + str(self.nn.layers[len(self.nn.layers)-1].ousz) + '_'
        
        # learning rate
        strname = strname + 'lr' + str(self.lr) + '_'
        strname = strname + 'wc' + str(self.weightcost) + '_'
        strname = strname + 'sgdstep' + str(self.sgdstep)
        strname += '.pk'
        
        return strname
    
