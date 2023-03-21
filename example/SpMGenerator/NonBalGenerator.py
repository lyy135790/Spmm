import MGnonbalance as nbal
import random

def genecsr(M, N, density, file):
    nbal.save_mat(M, N, density, 'csr', file)

def genecoo(M, N, density, file):
    nbal.save_mat(M, N, density, 'coo', file)

for i in range (0,10,1):
    R = 500*random.randrange(10,20)#随机大小的矩阵
    genecsr(R, R, 0.05, r'./data/test' + str(i) + '.mtx')