import datetime
import os
import sys

todaydate = datetime.date.today;
today_day = datetime.date.today().day
today_month = datetime.date.today().month
today_year = datetime.date.today().year
time_now = datetime.datetime.now();

filename = '/omn/'+str(today_day)+'-'+ str(today_month)+'-'+str(today_year)+'_log.txt'

file = open(filename,'w')

file.write('LOG TIME: '+str(time_now)+'\n\n');
file.write('%-------------------------------------%\n')
file.write('TRACKING 8 HOUR WORK DAY: \n\n');
for i in range(8):    
    file.write(str((time_now.hour+i)%24).zfill(2)+': \n')
file.write('%-------------------------------------%\n')
file.write('NOTES: \n\n');
file.write('%-------------------------------------%\n')
file.write('TODOS: \n\n');
for i in range(5):
    file.write(str(i+1)+'. \n')
file.write('%-------------------------------------%\n')
file.write('REVIEW: \n\n');
for i in range(5):
    file.write(str(i+1)+'. \n')
file.write('%-------------------------------------%\n')
file.write('CHECK: \n')
file.write('1. DO YOU HAVE POTENTIAL THRASH JOBS RUNNING ON THE CS CLUSTERS? \n')
file.write('2. [CLOSE ALL CS MATLAB] DO YOU HAVE EMPTY MATLAB RUNNING, ESPECIALLY WITH LOADED MEMORY? \n')

file.write('\n');
file.write('SLEEP EARLY?')

#os.system('emacs -nw '+filename+' &')
