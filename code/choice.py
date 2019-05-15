import numpy as np
import itertools

from numpy.linalg import svd

def stoch(X):
    return X / np.sum(X, axis=1)[:, None]
    
def stoch3(X):
    return X / np.sum(X, axis=2)[:,:,None]

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
    


def mshow( M, ms_delay=100):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if len(M.shape) == 2:
        plt.matshow(M, cmap='Blues')
        plt.axis('off')
        plt.show()
    elif len(M.shape) == 3:
        fig = plt.figure()
        ax = plt.gca()

        ims = []
        for mat in M:
            im = plt.imshow(mat, cmap='Blues', animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=ms_delay, blit=True,
                                        repeat_delay=1000)
        mshow.cur_ani = ani
        mshow.save = ani.save
        # ani.save('dynamic_images.mp4')

        plt.show()
    
## some tests ##    
# Q_rand = stoch(np.random.rand(5,5))
# U = np.random.rand(5,)
# 
# np.sum(Q_rand, axis=1)
# np.sum( transform(Q_rand, U, 4), axis=1)
# np.sum( trans_alt(Q_rand, U), axis =1)
#################

#from numpy import linalg

#u, s, vh = svd(trans_alt(Q_rand, U))
#linalg.det(Q_rand)

# def error( Gs, Us ):
    
