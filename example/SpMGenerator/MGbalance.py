from scipy import sparse, io
import numpy as np

def save_mat(
        M, N, density, matrixformat, item_item_sparse_mat_filename):
    #np.random.seed(10)
    raw_user_item_mat=sparse.rand(M, N, density, matrixformat,dtype=None)
    #raw_user_item_mat = np.random.randint(0, 6, (5, 4))
    d = sparse.csr_matrix(raw_user_item_mat)
    e = sparse.csc_matrix(raw_user_item_mat)
    io.mmwrite(item_item_sparse_mat_filename, d)
    print("item_item_sparse_mat_file information: ")
    print(io.mminfo(item_item_sparse_mat_filename))
    k = io.mmread(item_item_sparse_mat_filename)
    #print(k.todense())

#save_mat(10000, 10000, 0.00005, 'csr', r'./data/test.mtx')
