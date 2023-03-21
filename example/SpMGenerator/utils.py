import numpy as np
import scipy as sp

def read_mtx_coo(file):
    temp = sp.io.mmread(file)
    row = temp.row
    col = temp.col
    data = temp.data
    nnz = temp.nnz
    shape = temp.shape
    return row,col,data,nnz,shape

def get_matrix(file):
    return sp.sparse.coo_matrix(sp.io.mmread(file)).toarray()

def read_mtx_coo_sim(file):
    info = sp.io.mminfo(file)
    M = info[0]
    N = info[1]
    nnz=info[2]
    return M,N,nnz


# print(sp.io.mminfo("./data/test.mtx")[2])