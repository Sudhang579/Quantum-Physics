import numpy as np
from numpy import linalg as la
import math 
import cmath
import matplotlib.pyplot as plt
#D = int(input("Enter the number of rows:"))  
ham = [] 
#matrix=[[(int)(input()) for j in range(C)] for i in range(R)]
# Initialize matrix 
print("Enter the entries in a single line (separated by space): ")   
#entries = list(map(int, input().split())) 
ham = np.array([1,0,0,0,0,-1,2,0,0,2,-1,0,0,0,0,1]).reshape(4,4) 
E_val , E_vec = la.eig(ham)
#print(ham) 
#print("The eigen values of the Hamiltonian are:")
#print(E_val)
#print("and the corresponding eigen vectors are")
#print(E_vec)
#print("Enter the coefficients of pure state, separated by spaces")
#coef= np.array(input().split()).reshape(4,1)
coef=np.array([0.5, 0.5 , 0.5 , -0.5]).reshape(4,1)
#print(coef.shape, E_vec.shape)
abcd=la.solve(E_vec,coef)
#print(E_vec[0],E_vec[0].shape) [0. 0. 1. 0.] (4,)
#print(np.array([1,2,3,4]),np.array([1,2,3,4]).shape) [1 2 3 4] (4,)
a=abcd[0] 
b=abcd[1]
c=abcd[2]
d=abcd[3]
#print(a, b, c, d)
#print(E_vec[:,1].shape)
time=3
products=np.zeros((time,1))
state0=a*E_vec[:,0]+b*E_vec[:,1]+c*E_vec[:,2]+d*E_vec[:,3]
state0=state0.reshape(4,1)
for t in range(time):
    a=a*cmath.exp(-1j*E_val[0]/2*t)
    b=b*cmath.exp(-1j*E_val[1]/2*t)
    c=c*cmath.exp(-1j*E_val[2]/2*t)
    d=d*cmath.exp(-1j*E_val[3]/2*t)
    state=a*E_vec[:,0]+b*E_vec[:,1]+c*E_vec[:,2]+d*E_vec[:,3]
    state=state.reshape(4,1)
    inner_product=np.dot(np.conj(state.T),state0)
    inner_product=inner_product*inner_product.conjugate()
    products[t]=inner_product
x=products
y1=np.zeros((time,1))
for i in range(time):
	y1[i]=0.5*(1+math.cos(2*i))
plt.plot(x,y,x,y1,'ro')




#print(a,b,c,d)
#a=abcd[0][0], b=abcd[1][0], c=abcd[2][0], d=abcd[3][0]
#print(E_vec[:,0])

