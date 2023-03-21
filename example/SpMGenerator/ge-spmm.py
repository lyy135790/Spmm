import os
#import NonBalGenerator as bg
import BalGenerator as bg


cmd = "../ge-spmmav/spmm.out "
for i in range (0,10,1):
    append = " ./data/test" + str(i) + ".mtx"
    cmd += append

os.system(cmd+" 32")
print(cmd+" 32")

#../ge-spmmav/spmm.out ./data/test0.mtx