execfile('/home/wzou/Dropbox/nn_pred/dev/python/test/startup.py')

g = generator('default', 'default', 'default')

#res = g.default_a(100, 5, 9)
res = g.default_p(100, 9)

#plot(res); show()

sig = g.generate_signal()
