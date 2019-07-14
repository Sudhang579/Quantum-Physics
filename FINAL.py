import numpy as np 
from numpy import linalg as la
import cmath
import matplotlib.pyplot as plt
import math


sigma_x = np.array ( [ [ 0 , 1 ] , [ 1 , 0 ] ] )
sigma_y = np.array ( [ [ 0 , -1j ] , [ 1j , 0 ] ] )
sigma_z = np.array ( [ [ 1 , 0 ] , [ 0 , - 1] ] )


def create_Ham():
    ''' 
    Computes the hamiltonian for a given number of spins for a specific structure
     
    Returns: 
    H -- the hamiltonian '''
    
    print ( "Enter number of spins. ")
    n_s = int (input())
    dH = 2 ** n_s
    H = np.zeros( ( dH , dH ) )
    I = np.diag ( np.ones (2) )
    
    j = np.zeros (n_s)
    for i in range(n_s): 
        
        termx = np.array( [ [ 1 ] ] )
        termy = np.array( [ [ 1 ] ] )
        termz = np.array( [ [ 1 ] ] )
        if (i == (n_s -1) ) :
            j = np.zeros (n_s)
            j[0] , j[i] = 1, 1
        else:   
            j = np.zeros (n_s)
            j[i] , j[i+1] = 1 , 1
        for x in range(n_s):
            termx = np.kron ( termx , ( j[x] * sigma_x + (1 - j[x])*I ) ) 
            termy = np.kron ( termy , ( j[x] * sigma_y + (1 - j[x])*I ) ) 
            termz = np.kron ( termz , ( j[x] * sigma_z + (1 - j[x])*I ) ) 
        H = H + termx + termy + termz
    return H

def purify(Dx):
    ''' 
    Computes the purification with the input dimension after asking the user for the mixed density matrix
    
    Arguments:
    Dx -- dimension of the purified state
    
    Returns: 
    pure -- the purified state ''' 
    
    DA = int ( 2**( math.log2 ( Dx ) /2 ) )
    
  
    
    Dpure = Dx
    
    ### To randomly generate density matrices and verfiy purification scheme ###
    
    U = unitary_group.rvs( dim = DA )
    val1 = np.diag ( np.abs( np.random.rand (  DA  ) ) )
    denA = np.dot ( np.dot ( U , val1) , np.conjugate( U ).T ) 
    denA = denA / np.trace ( denA )
    IA = np.diag ( np.ones (DA) )
    
    ### To recieve mixed state from the user ###
    
    #print( "Enter the dimension of state A " )
    #DA = int (input())

    #denA = np.zeros ((DA , DA))
    #print( "Enter the density matrix of state A " )
    #denA = np.array(list(map(float, input().split()))).reshape( DA , DA )
    #denA = 1/DA * np.diag ( (np.ones ( DA) ) ) 
 
    valA , vecA = np.linalg.eig ( denA )            
    Scoeff = np.sqrt ( valA )
    
    IR = vecA   
    print ( 'Input density matrix  \n' , np.around ( denA , decimals = 2 ) )
    pure = np.zeros ( ( Dpure , 1 ))
    print()
    for i in range ( DA ):
        pure = pure + Scoeff[i] * np.kron ( vecA[ : , i ].reshape(DA,1) , IR [ : , i].reshape(DA,1 ) )
    pure = np.around ( pure , decimals = 5 )
    print ( " Purified State \n" , pure )
    print()
    print ( ' Partial trace of purified state: \n' , np.around ( partial_trace ( pure , DA ) , decimals = 2) )
    print () 
    return pure


def partial_trace(ket1 , dA):
    ''' 
    Computes the partial trace of the density matrix of pure state ket 
    
    Arguments:
    ket1 -- a pure state vector
    dA -- dimension of system of interest ( A )
    
    Returns: 
    ptr -- the partial trace of the density matrix of pure state ket '''
    
    d = len(ket1)             #dimension of input state vector
    ket = np.array(ket1).reshape(d,1)  
    dR = int ( len(ket)/ dA )         #dimension of reference system R
    IA = np.diag(np.ones(dA))  #creating identity operator for subsystem A 
    IB = np.diag(np.ones(dR))  #creating identity operator for subsystem R 
    bra = np.conj(ket).T       #bra of input state vector
    ptr = np.zeros((dA,dA),dtype=np.complex)
    
    for i in range (dR):     #loop evaluates ptr according to the standard definiton of partial trace
        Iket = np.kron( IA , IB[:,i].reshape(dR,1) )
        Ibra = np.kron( IA , IB[:,i].reshape(1,dR) )
        ptr = ptr + np.dot( np.dot(Ibra,ket) , np.dot(bra,Iket) )
    return ptr

def diagonalize(den):
    ''' 
    Diagonalizes the input matrix
    
    Arguments:
    den -- a density matrix
    
    Returns: 
    diag --- the diagonalized matrix '''
    
    den_loc = den
    val,vec = np.linalg.eig(den_loc)
    diag = np.dot( np.dot( np.conjugate(vec).T , den_loc) , vec )   #multiplication with  matrix with eigen vectors as columns 
    return diag

def entropy (den):
    '''
    Calculates Von Neumann entropy of input density matrix
    
    Arguments:
    den -- a density matrix
    
    Returns: 
    Eentropy --- the quantum entropy'''
    
    dim = len(den) 
    den_log = np.zeros ((dim,dim),dtype=np.complex)     #to store log(den)
    den_loc =  den                                         #creating local copy of den
    val , vec = np.linalg.eig(den_loc)                  
    for i in range(dim):    #loop constructs log(den) 
        den_log = den_log + cmath.log(val[i],2) * np.outer ( vec[ : , i] , np.conjugate(vec [ : , i] ).T) 
    Eentropy = - np.trace(np.matmul(den_loc,den_log))    #calculating -Tr(den*log(den))
    return Eentropy   

### CREATING A SPECIFIC HAMILTONIAN ###
    
ham = create_Ham()
D = int(len(ham))  

### FOR MANUAL INPUT OF EVERY ENTRY OF THE HAMILTONIAN ###

#print("Enter the dimension of Hamiltonian")
#D = int (inpput())
#print("Enter the entries of the Hamiltionian in a single line (separated by space): ")     
#ham = np.array(list(map(float, input().split()))).reshape(D,D) 

#This technique can also be used to calculate distance measure
# between two pure states as they evolve under different hamiltonians (ham1 and ham2)

E_val , E_vec = la.eig(ham)   #storing eigen values (array(d,) and eigen vectors  
print("The Hamiltionian is:")
print(ham)
print("Its eigen values are:")
print(E_val)
print("Corresponding eigen vectors are:")
print(E_vec)

### RECIEVING PURE STATES A AND B FROM USER ###

print("Enter the coefficients of first pure state, separated by spaces") 
coefA = np.array(list(map(float,input().split()))).reshape(D,1) 
print("Enter the coefficients of second pure state, separated by spaces")
coefB = np.array(list(map(float,input().split()))).reshape(D,1)
 
### RANDOM GENERATOR ####
   
#coefA = mixed_input(D)
#coefB = mixed_input(D)

print("The state A is:")
print(coefA)
print("The state B is:")
print(coefB) 

E_coefA0 = np.zeros((D,1))   #array that stores the coefficients of eigenbasis representation of state A                
E_coefA0 = la.solve( E_vec , coefA ).reshape(D,1)
E_coefB0 = np.zeros((D,1))   #array that stores the coefficients of eigenbasis representation of state  B
E_coefB0 = la.solve( E_vec , coefB ).reshape(D,1)

print("Density matrix for state A at t=0 is:")
denA0 = np.outer(coefA , coefA)  #density matrix for state A at time = 0 
print(denA0)
print("Density matrix for state B at t=0 is:")
denB0 = np.outer(coefB , coefB)  #density matrix for state B at time = 0   
print(denB0)

print("Set time till which the states have to be evolved:-")
end_time = float(input())    
print("Enter number of steps on computation:") 
steps = int( input())

fidelityAB = np.zeros((steps,1)).reshape(steps,1)  #array to store fidelities 
fidelityA = np.zeros((steps,1)).reshape(steps,1) 
fidelityB = np.zeros((steps,1)).reshape(steps,1)   
trace_dAB = np.zeros((steps,1)).reshape(steps,1)   #arrat to store trace distance
trace_dA = np.zeros((steps,1)).reshape(steps,1)   
trace_dB = np.zeros((steps,1)).reshape(steps,1)
entropyA = np.zeros((steps,1)).reshape(steps,1)
entropyB = np.zeros((steps,1)).reshape(steps,1)
   
stateA0 = np.zeros((D,1)).reshape(D,1)      
stateB0 = np.zeros((D,1)).reshape(D,1)

count = 0  #counter                  

for k in range(D):    #loop to reconstruct states from E_coef and verify that it is correct
    stateA0 = stateA0 +E_coefA0[k] * (E_vec[:,k].reshape(D,1))
    stateB0 = stateB0 +E_coefB0[k] * (E_vec[:,k].reshape(D,1))
print("State A at t = 0:  ", stateA0)
print()
print("State B at t = 0:  ", stateB0)

for t in np.linspace(0,end_time,steps):
    stateAt = np.zeros((D,1),dtype=np.complex).reshape(D,1)  #state of A at time t
    stateBt = np.zeros((D,1),dtype=np.complex).reshape(D,1)  #state of B at time t 
    E_coefAt = np.zeros((D,1),dtype=np.complex).reshape(D,1)   #eigen coefficients of A at time t
    E_coefBt = np.zeros((D,1),dtype=np.complex).reshape(D,1)   #eigen coefficients of B at time t 
    ## evolution of eigen coefficients with time ##
    for z in range(D):            
        E_coefAt[z]=E_coefA0[z]*cmath.exp(-1j*E_val[z]*t/2)   
        E_coefBt[z]=E_coefB0[z]*cmath.exp(-1j*E_val[z]*t/2)
    ## reconstruction of state A and state B from their eigen coefficients ##
    for k in range(D):   
        stateAt = stateAt + E_coefAt[k] * (E_vec[:,k].reshape(D,1))
        stateBt = stateBt + E_coefBt[k] * (E_vec[:,k].reshape(D,1))
    ## computing denisty matrices of state A and B at time t ## 
    denAt = np.outer(np.conj(stateAt) , stateAt).reshape(D,D) 
    denBt = np.outer(np.conj(stateBt) , stateBt).reshape(D,D)
    
    ### CALCULATION OF TRACE DISTANCE AT TIME T ### 
    
    eigvAB = np.linalg.eig((denAt-denBt))[0].reshape(D,1)   #calculating eigen values of (denAt - denBt)
    eigvA = np.linalg.eig((denAt-denA0))[0].reshape(D,1)   
    eigvB = np.linalg.eig((denBt-denB0))[0].reshape(D,1)   
    eigvAB = np.abs(eigvAB)
    eigvA = np.abs(eigvA)
    eigvB = np.abs(eigvB)
    trace_dAB[count]=np.sum(eigvAB,axis=0)*1/2   #trace distance is sum of magnitude of eigen values 
    trace_dA[count]=np.sum(eigvA,axis=0)*1/2    
    trace_dB[count]=np.sum(eigvB,axis=0)*1/2    
    
    ### CALCULATION OF FIDELITY AT TIME t ###
    
    inner_productA=np.dot(np.conj(stateAt).T,stateA0)   
    inner_productB=np.dot(np.conj(stateBt).T,stateB0)   
    inner_productAB=np.dot(np.conj(stateAt).T,stateBt)    
    inner_productAB2=inner_productAB*(inner_productAB.conjugate())  #magnitude square of inner_product
    inner_productA2=inner_productA*(inner_productA.conjugate())  
    inner_productB2=inner_productB*(inner_productB.conjugate())  
    fidelityAB[count]=inner_productAB2
    fidelityA[count]=inner_productA2
    fidelityB[count]=inner_productB2
    
    ###CALCULATION OF ENTANGLEMENT ENTROPY AT TIME t ###
    entropyA[count] = entropy(partial_trace( stateAt, 2 ))
    entropyB[count] = entropy(partial_trace( stateBt, 2 ))
    
    #print( "rhoA : \n" , np.around (partial_trace(stateAt , D) , decimals =2 ) )
    #print( "rhoB : \n" , np.around( partial_trace(stateBt , D) , decimals =2 ) )
    
    #print ( "Entropy A : " , entropyA[count])
    #print ( "Entropy B : " , entropyB[count])
    
    
    count=count+1   # counter update
    print()
    
x = np.array(np.linspace(0 , end_time , steps))   # array to store times at which computation is done
plt.subplots_adjust(hspace=1.0, wspace=0.4)

### PLOTTING RESULTS ###  
plt.subplot(3, 2, 1)
plt.title('State A : T and F')
plt.plot(x,fidelityA,label='Fidelity')
plt.plot(x,trace_dA,'r--',label='Trace Distance')
plt.legend(loc='upper left')
plt.xlabel('Time')

plt.subplot(3 ,2 ,2)
plt.title('State B : T and F')
plt.plot(x,fidelityB,label='Fidelity')
plt.plot(x,trace_dB,'r--',label='Trace Distance')
plt.legend(loc='upper left')
plt.xlabel('Time')

plt.subplot(3 ,2 ,3)
plt.title('State A: F and S')
plt.plot(x,fidelityA,label='Fidelity')
plt.plot(x,entropyA,'r--',label='Entropy')
plt.legend(loc='upper left')
plt.xlabel('Time')

plt.subplot(3 ,2 ,4)
plt.title('State B: F and S')
plt.plot(x,fidelityB,label='Fidelity')
plt.plot(x,entropyB,'r--',label='Entropy')
plt.legend(loc='upper left')
plt.xlabel('Time')

plt.subplot(3, 2 ,5)
plt.title('State A,B: F and T')
plt.plot(x,fidelityAB,label='Fidelity')
plt.plot(x,trace_dAB,'r--',label='Trace')
plt.legend(loc='upper left')
plt.xlabel('Time')

plt.subplot(3,2,6)
StateA = np.around ( stateA0.reshape(1,D) , decimals = 2)
StateB = np.around ( stateB0.reshape(1,D) , decimals = 2)

plt.text(0.5,0.5 ,( 'state A is :' + str(StateA)+ '\n' + 'state B is :' +str(StateB) ), fontsize = 11, ha='center') 

plt.show()



    
    