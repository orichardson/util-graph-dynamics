
import numpy as np
import networkx as nx

from gensim.models import Word2Vec


class NoMoreNeighbors(Exception): pass

def sample(distrib):
    # shitty, non-alias sampling to start
    try:
        return np.random.choice(list(distrib.keys()), p = list(distrib.values()))
    except ValueError:
        raise NoMoreNeighbors()

def normalize(distrib):
    n_const = sum(distrib.values())
    if n_const == 0:
        return {}
    return { k : v / n_const for (k,v) in distrib.items() }

def biased_rwalk(G, pi, u, l):
    walk = [ u ]    
    try :
        for i in range(l):
            walk.append( sample( pi[(walk[-2], walk[-1]) if len(walk) > 1 else walk[-1] ])  ) 
    except NoMoreNeighbors:
        pass
        
    return walk 
    
def simulate_walks(G_original, p, q, r, l):
    # π = preprocessModifiedWeights(G,p,q)
    G = nx.Graph(G_original)
    transition_p = {}   # : (Node x Node2) -> Dist[Neighbors[Node2]]
                        # indexed by pairs of nodes, this is actually directed.
                    
    # this is the weighting factor    
    def alpha(t, x):
        if t == x:              # with weighting proportional to 1/p, go back
            return 1.0 / p
        elif t not in G[x]:     # with weighting proportional to 1/q go out (since w is 2 hops from u)
            return 1.0 / q
        return 1.0
    
    # set up transition_p, by iterating through each node, neighbor pair (recall: directedness matters)
    for n in G:
        transition_p[n] = normalize({ v : G[n][v].get('weight', 1) for v in G[n]}) 
            # sample according to weights if this is the first node
        for m in G[n]:
            transition_p[(n,m)] = normalize({ v : alpha(n,v) * G[m][v].get('weight', 1)  for v in G[m] })
            # otherwise, if n was the previous node, m is the current node, reweight v by alpha.
    
    #print(transition_p)
    
    walks = []
    for x in range(r):
        for v in G:
            walks.append( biased_rwalk(G, transition_p, v, l))
    return walks


def learnFeats(G_original, d, r, l, k, p, q, workers=8, iter=1, outfile=None, precomputed_walks=None):
    walks = simulate_walks(G_original, p, q, r, l) if precomputed_walks is None else precomputed_walks;
    
    ### learn embeddings: SGD and skip-grams outsourced to another library
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=d, window=k, min_count=0, sg=1, workers=workers, iter=iter)
    
    if outfile:
        model.save_word2vec_format(outfile) #"output/graph"
    
    return model
