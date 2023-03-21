import numpy as np
import utils
import os
 
def make_rand_matrix(side=20): # 随机矩阵
    a = np.random.random((side,side))
    for i in range(0,side):
        for j in range(0,side):
            if a[i,j]>0.8:
                a[i,j] = 1
            else:
                a[i,j] = 0
    return a
 
def get_min_std_matrix(nnz,shape): # 最均匀情况，相同nnz把所有非0按稀疏度重组，均匀度=1
    # gap = int(shape[0]*shape[1]/nnz)# 多少个数里面有一个非零
    # location_p = [[0] * int(shape[1]/10)] * int(shape[0]/10)
    return 0

 
def get_max_std_matrix(nnz,shape): # 最不均匀情况，相同nnz把所有非0堆在一起,均匀度=0
    rowfill = nnz/shape[0]#填满多少行
    rowfill100 = rowfill/100
    rowfillnot100 = rowfill%100 #不满10行的部分
    colfill = nnz/shape[1]#剩余多少列
    location_p = np.zeros(int(shape[0]*shape[1]/10000))
    for i in range(0,int(rowfill100*(shape[1]/100)),1):
        location_p[i]=100
    for j in range(int(rowfill100*(shape[1]/100)),int(rowfill100*(shape[1]/100)+(shape[1]/100)),1):
        ap=100 if colfill-100>=0 else colfill
        colfill=colfill-100 if colfill-100>=0 else 0
        location_p[j]=(rowfillnot100*100+ap)
    return np.std(location_p)


def get_rand_std(rand_matrix): # 输入矩阵为0,1矩阵，计算矩阵分布均匀程度
    sum_ = sum(sum(rand_matrix))
    [x_side, y_side] = rand_matrix.shape
    fit_x_side = int(x_side/10)
    fit_y_side = int(y_side/10)
    location_p = []
    for i in range(0, x_side-10, int(fit_x_side/2)): 
        for j in range(0,y_side-10, int(fit_y_side/2)):
            temp_matrix = rand_matrix[i:i+10,j:j+10]
            cnt_p = sum(sum(temp_matrix))
            location_p.append(cnt_p)
    std = np.std(location_p)#计算标准差
    return std
 
def get_rand_coo(file):#coo 格式 10*10一组
    row,col,data,nnz,shape = utils.read_mtx_coo(file)
    location_p = [[0] * int(shape[1]/100)] * int(shape[0]/100)
    for i in range(0, nnz, 1):
        rowindex = int(row[i]/100)
        colindex = int(col[i]/100)
        location_p[rowindex][colindex] += 1
    # print(location_p)
    std = np.std(location_p)
    return std

def get_stand_std(file): # 将均匀程度分布在0,1之间，1表示分布最均匀
    A = utils.get_matrix(file)
    M,N,nnz = utils.read_mtx_coo_sim(file)
    A_std = get_rand_coo(file)
    print(A_std)
    A_max_std = get_max_std_matrix(nnz,(M,N))
    print(A_max_std)
    A_min_std = get_min_std_matrix(nnz,(M,N))
    if A_std < A_min_std:
        return 1
    else:
        return max(0,(A_std-A_max_std)/(A_min_std-A_max_std))#均匀分布函数F(x)=x-a/b-a
    
def run_by_uni(uni,DataFilePath):
    if uni< 0.001:
        os.system("../ge-spmm2/spmm.out " + DataFilePath + " 32 2")
    else:
        os.system("../ge-spmm2/spmm.out " + DataFilePath + " 32 0")




if __name__ == "__main__":
    # A = make_rand_matrix(15) 
    # print(A)
    # print("\n")
    # print(get_min_std_matrix(A))
    # print("\n")
    # print(get_rand_coo("./data/test.mtx"))
    file = "./data/test2.mtx"
    std=get_stand_std(file)
    print(std)
    # print(get_max_std_matrix(392000,(28000,28000)))
    # os.system("../ge-spmm2/spmm.out ./data/test.mtx 32")
    run_by_uni(std,file)