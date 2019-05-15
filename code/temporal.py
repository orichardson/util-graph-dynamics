import os
import json
import networkx as nx
import numpy as np

from choice import stoch

class TGraph:
    def __init__(self, 
        Gs, times, filename, node_attrs, edge_attrs,
        ker = None, u_attr = None, q_attr = None, **kwargs
    ):
        self.Gs = Gs
        self.times = times
        self.filename = filename
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs

        self.ker = ker
        self.u_attr = u_attr
        self.q_attr = q_attr

    def __call__(self, t): # return G(t)
        pass
        
    def save(self):
        try: # ensure that the directory exists
            os.mkdir('../data/tgraphs/'+self.filename)
        except:
            pass
            
        with open('../data/tgraphs/'+self.filename+'/meta.json', 'w') as meta_file:
             meta_file.write(json.dumps( {k : self.__dict__[k] for k in self.__dict__.keys() - {'Gs'}}  ))
            
        for i,G in enumerate(self.Gs):
            for e in list(G.edges):
                for p, v in dict(G.edges[e]).items():
                    newpname = p.replace('_','');
                    G.edges[e][newpname] = G.edges[e][p]
                    if '_' in p:
                        del G.edges[e][p]
            for n in list(G.nodes):
                for p, v in dict(G.nodes[n]).items():
                    newpname = p.replace('_','');
                    G.nodes[n][newpname] = G.nodes[n][p]
                    if '_' in p:
                        del G.nodes[n][p]        
                
            nx.write_gml(G, '../data/tgraphs/%s/%d.gml' %(self.filename,i))
    
    def V(self):
        if hasattr(self, '_V'):
            return self._V
            
        nodes = set()
        for G in self.Gs:
            nodes |= set(G.nodes)
            
        self._V = sorted(nodes)
        return self._V
            
        
    
    def QQ(self):
        if self.q_attr is None:
            raise ValueError("No attribute to build matrix weights from; set q_attr first")
        
        
        Qs = []
        V = self.V()
        for G in self.Gs:
            W = nx.adj_matrix(G, weight=self.q_attr, nodelist=V).toarray() + np.eye(len(V)) / 1E7
            # np.fill_diagonal(W, 0)
            Qs.append(stoch(W))
        
        return np.dstack( tuple(Qs) )

def gload(fname):
    try:
        with open('../data/tgraphs/%s/meta.json'%fname) as meta_file:
            data = json.load(meta_file) 
    except:
        data = dict(filename = fname, times = None, node_attrs = [], edge_attrs=[])
        
    Gs = []
    i = 0
    
    while True :
        try:
            Gs.append(nx.read_gml('../data/tgraphs/%s/%d.gml' %(fname,i)) )
            i += 1
        except FileNotFoundError:
            break
            
    if data['times'] is None:
        data['times'] = list(range(i))
        
    return TGraph(Gs, **data)
    
