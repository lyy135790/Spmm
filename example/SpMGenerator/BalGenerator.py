import MGbalance as bal
import random

def genecsr(M, N, density, file):
    bal.save_mat(M, N, density, 'csr', file)

def genecoo(M, N, density, file):
    bal.save_mat(M, N, density, 'coo', file)

for i in range (0,10,1):
    R = 100*random.randrange(10,20)
    genecsr(R, R, 0.1, r'./data/test' + str(i) + '.mtx')
