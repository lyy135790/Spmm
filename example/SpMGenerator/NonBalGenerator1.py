import MGnonbalance as nbal
import random

def genecsr(M, N, density, file):
    nbal.save_mat(M, N, density, 'csr', file)

def genecoo(M, N, density, file):
    nbal.save_mat(M, N, density, 'coo', file)

R = 2001*random.randrange(10,20)
genecsr(R, R, 0.0001, r'./data/test.mtx')