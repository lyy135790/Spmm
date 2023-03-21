import random
#写入文件
def writeFile(m,n,d,N):
    filepath = '/home/ustc/int.gary.li'
    file = filepath + '/test.mtx'
    with open(file,'w',encoding='utf-8') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(str(m)+' '+str(n)+' '+str(d)+'\n')
        f.write(N)


