import os
import json
import networkx as nx
import numpy as np
from datetime import datetime

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
    
    def V(self, key=None, recompute=False):
        if hasattr(self, '_V') and not recompute:
            return self._V
            
        nodes = set()
        for G in self.Gs:
            nodes |= set(G.nodes)
            
        self._V = sorted(nodes,key = key)
        return self._V
        
    def sortVbyU(self):
        self.V(key = lambda n:sum(G.nodes.get(n,{self.u_attr:0})[self.u_attr] for G in self.Gs) / len(self.Gs), recompute=True )
            
        
    def trim(self):
        n = len(self.Gs)
        self.times = self.times[:n]
        if self.filename[-7]+self.filename[-4]+self.filename[-1] == '(-)':
            self.filename = self.filename[:-3]+str(datetime.fromtimestamp(self.times[-1]).year)[2:]+')'
            print("New Filename: "+ self.filename)
        
    def __repr__(self):
        t_ss, t_es = (datetime.fromtimestamp(t).strftime('%b %y') \
                for t in [self.times[0],self.times[-1]])
        
        return f"<TGraph with {len(self.V())} nodes across {len(self.Gs)} steps [{t_ss} -- {t_es}]\n\t" + \
            f" U: {self.u_attr};  W: {self.q_attr}>" 
    
    def Ws(self, clear_diag = False):
        if self.q_attr is None:
            raise ValueError("No attribute to build matrix weights from; set q_attr first")

        Ws = []
        V = self.V()
        for G in self.Gs:
            W = nx.adj_matrix(G, weight=self.q_attr, nodelist=V).toarray() + np.eye(len(V)) / 1E7
            if clear_diag:
                np.fill_diagonal(W, 0)
            Ws.append(W)
        
        return np.stack( tuple(Ws), axis=0)            
                
    def get_transitions(self):
        return stoch3(self.Ws())
        
    def Us(self):
        return np.stack( [[ G.nodes.get(n,{self.u_attr:0}).get(self.u_attr,0) \
            for n in self.V()] for G in self.Gs], axis=0 )
        
    def convolve( self, ker ):
        pass
        
    def __mul__(self, ker):
        return self.convolve(ker)
        

def gload(fname):
    defaults = dict(filename = fname, times = None, node_attrs = [], edge_attrs=[], q_attr = "nlinks", u_attr="activity")
    try:
        with open('../data/tgraphs/%s/meta.json'%fname) as meta_file:
            data = json.load(meta_file)
        
        for k, v in defaults.items():
            if k not in data:
                data[k] = v 
    except:
        data = defaults
        
    Gs = []
    i = 0
    
    while True :
        try:
            Gs.append(nx.read_gml('../data/tgraphs/%s/%d.gml' %(fname,i)) )
            i += 1
        except FileNotFoundError:
            break
            
    if 'times' not in data or data['times'] is None:
        data['times'] = list(range(i))
        
    return TGraph(Gs, **data)
    
