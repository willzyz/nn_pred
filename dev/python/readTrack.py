from pylab import *

file = '/home/wzou/apc/track/cpu_track_2012-08-15T00:48:55.284500.txt';
#file = '/home/wzou/apc/track/mem_track_2012-08-14T23:40:20.554979.txt';

f = open(file, 'r');
title = f.readline();title=title.split('\n')[0];
s1=[]; s2=[];
t =f.readline();
count = 1;
while 1:
    t = f.readline();    
    print t.split(', ');
    if size(t.split(', '))>1:
        diff = float(t.split(', ')[1].split('\n')[0]);
        print diff;
        s1.append(float(t.split(', ')[0]));
        s2.append(diff);
    if (not t) or (count>3600):
        break;
    count = count + 1;

print s1;
#print s2;

plot(s1, s2, linewidth=1.0); show();
xlabel('time');
ylabel('metric');
