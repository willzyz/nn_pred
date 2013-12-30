### script to generate time series data from Will's computer ###
import os, timeit, sys, getpass
from time import sleep,time
from datetime import datetime
sys.path.append(os.getcwd()+'/include/');
from memorymonitor import MemoryMonitor
import commands

small_ratio = 48; ## how many hours of recording

# ----- Initialize -----
path = os.getcwd()+'/../../track/'; file = 'track_'+datetime.now().isoformat()+'.txt';
#   --- Memory monitor ---
mem_mon = MemoryMonitor(getpass.getuser());
T = 0;

# ----- Name & create files -----
tl=['cpu_','mem_']; tfile=[];
for t in tl:
    temp=path+t+file;os.system('touch '+temp);tf=open(temp,'a');tfile.append(tf);tf.write(datetime.now().isoformat()+'\n');tf.write(' record time: '+str(small_ratio)+' hours');

# ----- Start tracking -----
while(T<small_ratio*3600):
    upt = commands.getoutput('uptime');
    totalcpu = float(upt.split('averages: ')[1].split(' ')[0]);
    memuse = mem_mon.usage();
    tfile[0].write(str(time())+',  '+str(totalcpu)+'\n');
    tfile[1].write(str(time())+',  '+str(memuse)+'\n');
    T = T + 1;
    if T%300==1:
        print('\n'+'writing at time '+str(T)+'/'+str(3600*small_ratio)+'\n');
    else:
        sys.stdout.write('.'); sys.stdout.flush();
    sleep(1);
    
print(T);
