from scipy import sparse, io, vstack
import numpy as np

def save_mat(
        M, N, density, matrixformat, item_item_sparse_mat_filename=r'./data/test.mtx'):
    #np.random.seed(10)
    fill = sparse.rand(int(0.499*M), N, density/100, matrixformat,dtype=None).todense()
    raw_user_item_mat=sparse.rand(int(0.002*M), N, 200*density, matrixformat,dtype=None)
    data = sparse.csr_matrix(vstack([fill, raw_user_item_mat.todense(), fill]))
    io.mmwrite(item_item_sparse_mat_filename, data)
    print("item_item_sparse_mat_file information: ")
    print(io.mminfo(item_item_sparse_mat_filename))

