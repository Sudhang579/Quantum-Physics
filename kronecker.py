#Outputs the kronecker product of two matrices
#This can also be done by the numpy function np.kron(A,B)

import numpy as np
A=np.array([[1,2],[3,4]])
B=np.array([[1,1],[1,1]])
print(np.kron(A,B)) 
a,b=A.shape
m,n=B.shape
C=np.zeros((a*m,b*n))
for i in range(a*m): #manuallt calculating kronecker product
	for j in range(b*n):
		C[i][j]=B[int(i/a)][int(j/b)]*A[i%a][j%b]
print(C)
