import MGbalance as bal
import random

def genecsr(M, N, density, file):
    bal.save_mat(M, N, density, 'csr', file)

def genecoo(M, N, density, file):
    bal.save_mat(M, N, density, 'coo', file)

R = 2000*random.randrange(10,20)
genecsr(R, R, 0.0005, r'./data/test.mtx')
