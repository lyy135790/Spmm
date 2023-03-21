import scipy
import SpMGenerator.file as file

m=40
n=40
density=0.2
matrixformat='coo'
B=scipy.sparse.rand(m,n,density=density,format=matrixformat,dtype=int)
file.writeFile(m,n,m*n*density,str(B))
print(str(B))
