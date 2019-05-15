
class TGraph:
    def __init__(self, edgelist, kernel):
        self.edgelist = edgelist
        self.ker = kernel

    def __call__(self, self, t): # return G(t)
        pass
        
        
        
def graph2QU(G) :
    W = nx.adjacency_matrix(G, 'n_links')
