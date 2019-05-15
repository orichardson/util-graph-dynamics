import numpy as np
import itertools


def stoch(X):
    return (X.T / np.sum(X.T, axis=0)).T

def logit(U):
    n_vert = len(U)
    A = np.zeros((n_vert, n_vert))
    
    normalizer =  sum(np.exp(U[s]) for s in range(n_vert))
    
    for v in range(n_vert):
        A[:,v] = np.exp(U[v]) / normalizer 
    return A

#################### transformation functions. ################

def transform(Q, U, k): 
    n_vert = len(Q)
    A = np.zeros(np.shape(Q))
    
    for N in itertools.product(*[range(n_vert) for i in range(k)]):
        normalizer = sum(np.exp(U[s]) for s in set(N))
        for u in range(n_vert):
            for v in range(n_vert):
                if v in N:
                    A[u,v] += np.prod([Q[u,s] for s in N]) * np.exp(U[v]) / normalizer
                    
    return A

    
def trans_alt(Q, U):
    A = np.zeros(Q.shape)    
    eU = np.exp(U)
    A = Q / (1 + np.outer(eU, 1/eU) ) # exp(U[i] - U[j])

    np.fill_diagonal(A, 0)
    A[np.diag_indices(Q.shape[0])] = 1 - np.sum(A, axis=1)
    
    ###### slower ELEMENT-WISE ALGORITHM ########
    # for i in range(n):
    #     for j in range(n):
    #         A[i,j] = Q[i,j] / (1 + np.exp(U[i] - U[j]))
    # 
    # for i in range(n):
    #     A[i,i] = Q[i,i] + sum( Q[i,s] / (1 + np.exp(U[s] - U[i])) for s in range(n) if i != s)

    return A    
    


## some tests ##    
# Q_rand = stoch(np.random.rand(5,5))
# U = np.random.rand(5,)
# 
# np.sum(Q_rand, axis=1)
# np.sum( transform(Q_rand, U, 4), axis=1)
# np.sum( trans_alt(Q_rand, U), axis =1)
#################


def error( Gs, Us ):
    
