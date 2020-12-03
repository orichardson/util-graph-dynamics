import numpy as np
from datetime import datetime, timedelta
from numpy import linalg
from scipy.linalg import fractional_matrix_power

from collections import defaultdict

from temporal import TGraph
from choice import transform, trans_alt, stoch, stoch3

from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error

from sklearn.base import BaseEstimator, RegressorMixin
from difflib import SequenceMatcher
import itertools

from functools import partial


class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__


class rsltdict(dict):
	"""For frozen set items keys (dict keys), want to query results in any order"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def query(self, q = '', **kwargs):  
		max_score = (0,0) #kwarg matches, max arg similarity, lexicographically ordered.
		arg_max = None
		for k,v in self.items():
			matchlength = max(  \
			 	SequenceMatcher(None, q, str(v)).find_longest_match(0, len(q), 0, len(str(v))).size \
					for v in  dict(k).values() )
			score = (len(set(kwargs.items()) & k), matchlength)
			
			if score > max_score:
				max_score = score
				arg_max = v			
			
		return arg_max

def make_learning_problem(TG, offset=1, incl_Us=None, testfraction=0.15, predict = 'Q', clear_diag=False):
	stackedweights = np.nan_to_num(stoch3(TG.Ws(clear_diag=clear_diag)).reshape(len(TG.Gs), -1))
	
	split_idx = int(len(TG.Gs) * (1-testfraction))
	
	def mkdata( offset ):
		Xs = stackedweights
		if not incl_Us is None:
			Us = TG.Us()
			if incl_Us == 'log':
				Us = np.log(Us + 1)
			elif incl_Us.startswith('div'):
				Us = Us / float(incl_Us[3:])
			elif incl_Us.startswith('norm'):
				Us = float(incl_Us[4:]) * Us / np.max(Us)
			
			Xs = np.hstack([Xs, Us])
		
		Xs = Xs[:-offset]
		Ys = (stackedweights if predict == 'Q' else TG.Us())[offset:]        
		
		X_train, X_test = Xs[:split_idx], Xs[split_idx:]
		Y_train, Y_test = Ys[:split_idx], Ys[split_idx:]
		return (X_train, Y_train), (X_test, Y_test)
	
	if type(offset) is int:
		return mkdata(offset)
		
	elif offset == 'all':
		pass # do something fancy to make sure no data leaks.
	

class SKWrapper(BaseEstimator, RegressorMixin):
	def __init__(self, f = lambda X: X, name=None, alpha=1, interp=None):
		self.f = f
		self.alpha = alpha
		self.interp = interp
		self.name = name
	
	def fit(self, X, y=None):
		d2 = X.shape[1]
		d = int(np.sqrt(d2))
		
		if d **2 == d2: # this is a perfect square;
			self.delim = None
			self.N = d
		else:
			self.N = int((np.sqrt(4*d2 + 1) - 1) / 2)
			self.delim =  self.N ** 2

		return self # this is a dummy wrapper, still needs to return self
		
	def _fi(self, x):
		Q,U = (x,None) if self.delim is None else (x[:self.delim], x[self.delim:])
		Q = Q.reshape(self.N, self.N)
		
		try:
			predict = self.f(Q=Q, U=U) 
		except Exception as e:
			print(e)
			predict = self.f(Q) 
		
		if not self.interp:
			return predict.flatten()
		if self.interp == 'linear': 
			return (np.multiply(1-self.alpha, Q) + np.multiply(self.alpha, predict) ).flatten()
		elif self.interp == 'multiplicative':
			return np.real(fractional_matrix_power( predict @ linalg.inv(Q) , self.alpha ) @ Q ).flatten()
		
	def predict(self, X):
		return np.array([self._fi(x) for x in X])

class AvgQInv(BaseEstimator, RegressorMixin):
	""" Compute average of multiplicative diffs between matrices """

	def fit(self, X, y=None):
		d2 = X.shape[1]
		d = int(np.sqrt(d2))
		
		if d * d == d2: # this is a perfect square;
			self.delim = None
			self.N = d
			
			Qs = X.reshape(X.shape[0], d, d)
				
		else:
			self.N = d = int((np.sqrt(4*d2 + 1) - 1) / 2)
			self.delim =  d * d
			
			Qs = X[:,:self.delim].reshape(X.shape[0], d, d)
			
		T = np.zeros((d,d))
		for Q_prev, Q_here in zip(Qs, Qs[1:]):
			T += (Q_here @ np.clip(linalg.inv(Q_prev + np.random.normal(0,1E-2, T.shape)), -1, 1)) #  
		T /= Qs.shape[0] - 1
		self.T = T

		return self # this is a dummy wrapper, still needs to return self
		
	def _fi(self, x):
		Q,U = (x,None) if self.delim is None else (x[:self.delim], x[self.delim:])
		Q = Q.reshape(self.N, self.N)
		
		return (self.T @ Q).flatten()
		
	def predict(self, X):
		return np.array([self._fi(x) for x in X])


def predictor_name( p ):
	if type(p) is SKWrapper and p.name:
		return p.name
	else:
		return type(p).__name__
		
LOG = []

def error_multidim(predictor, data):
	(X_train, Y_train), (X_test, Y_test) = data
	
	#print(X_test.shape)
	if(X_test.shape[0] <= 1): # Don't use N=1 values.
		return None
		
	ers = []
	for k in range(Y_train.shape[1]):
		f = predictor.fit(X_train, Y_train[:,k])
		
		predicted = f.predict(X_test)
		if np.size(predicted) == np.size(Y_test):
			try:
				ers = [mean_squared_error(predicted, Y_test)]
			except Exception as e:
				LOG.append([predicted, predictor])
				raise
			break
		else:
			if k % 20 == 7:
				prc = k / Y_train.shape[1]
				print("\r %.1f%%" % (100*prc) + ' \t|'+('*'*int(30*prc))+ (' '*int(30-30*prc))+ '|', end='\r')
	
		ers.append(mean_squared_error(predicted, Y_test[:,k]))
		
	return np.mean(ers)

def error_by_preditor( predictors, data ):    
	return  { predictor_name(p) : error_multidim(p, data) for p in predictors }

def fz (**kwargs): # short freeze
	return frozenset(kwargs.items())

def run_heatmaps( TG : TGraph, result_dict = rsltdict()):
	"""
	:param result_dict the place to put things so that there are intermediate results 
		to be plotted while other things are running
	"""
	## PARAMETERS:
	PARAMS = dotdict(
		offsets = [1,2,3,4,6,8,10,16],
		alpha =  np.linspace(0,1,20), #[2**(-i) for i in range(30)][::-1],
		clear_diag = [False], #True
		U_mode = ['log', 'norm10', 'norm5', 'norm30', 'div1000'], #div3000
		α_predictors = [
			SKWrapper(name="trans_alt", f = trans_alt, interp='linear'),
			SKWrapper(name="$\Psi^{(2)}$", f = lambda Q, U: transform(Q, U, k=2), interp='linear' ),
			SKWrapper(name="$Q^3$", f = lambda Q, U : Q  @ Q @ Q, interp='multiplicative' ),
			SKWrapper(name="$Q^{100}$", f = lambda Q, U : linalg.matrix_power(Q,100), interp='linear' ),
		])
	
	run_heatmaps.results = result_dict 
	# save most recent result dict as a method property so it's not lost in the event of inpterrupt
	
	###### PLOT  (dT, alpha) -> Err #####   
	data = {}
	
	for cd, predictor, iU in itertools.product(PARAMS.clear_diag,PARAMS.α_predictors, PARAMS.U_mode):
		#for interp in 'linear', :# 'multiplicative':
		try:
			# predictor.interp = interp
			mat_params = fz(name = predictor.name, iU=iU)#, clear_diag=cd, interp=predictor.interp)
			
			mat =  np.zeros((len(PARAMS.offsets), len(PARAMS.alpha)))
						
			for i,o in enumerate(PARAMS.offsets):
				data_params = fz(offset=o, clear_diag=cd, incl_Us=iU)
				if not (data_params in data):
					data[data_params] = D = make_learning_problem(TG, **dict(data_params))
				D = data[data_params] 
				
				for j,α in enumerate(PARAMS.alpha):
					predictor.alpha = α
					mat[i,j] =  error_multidim(predictor, D)

			result_dict[mat_params] = mat
			print('ALPHA map finished for ', mat_params)
		except Exception as e:
			print(e)
	return rsltdict(result_dict)    
	

	
def run_regressions( TG : TGraph, include_expensive=False, test_fraction=0.15, norm=10, offset=1):
	import warnings
	predictors = [
		SKWrapper(name="trans_alt α=0.06+", f = trans_alt, alpha = 0.06, interp='linear'),
		#SKWrapper(name="trans_alt α=0.06*", f = trans_alt, alpha = 0.06, interp='multiplicative'),
		#SKWrapper(name="trans_alt α=0.04+", f = trans_alt, alpha = 0.04, interp='linear'),
		#SKWrapper(name="trans_alt α=0.04*", f = trans_alt, alpha = 0.04, interp='multiplicative'),
		SKWrapper(name="trans_alt α=0.13+", f = trans_alt, alpha = 0.13, interp='linear'),
		SKWrapper(name="trans_alt α=0.2+", f = trans_alt, alpha = 0.2, interp='linear'),
		SKWrapper(name="trans_alt α=0.25+", f = trans_alt, alpha = 0.25, interp='linear'),
		SKWrapper(name="trans_alt α=0.35+", f = trans_alt, alpha = 0.35, interp='linear'),
		#SKWrapper(name="trans_alt α=0.11*", f = trans_alt, alpha = 0.11, interp='multiplicative'),
		SKWrapper(name="transform k=2 α=0.1+", f = partial(transform,k=2), alpha = 0.1, interp='linear'),
		SKWrapper(name="transform k=2 α=0.2+", f = partial(transform,k=2), alpha = 0.2, interp='linear'),
		SKWrapper(name="transform k=3 α=0.1+", f = partial(transform,k=2), alpha = 0.1, interp='linear'),
		SKWrapper(name="$Q$", f = lambda Q, U : Q ),
		AvgQInv(),
	]
	expensive_predictors = [
		SGDRegressor(tol=1E-5, max_iter=1000),
		LinearSVR(tol=1E-5),
		SVR(gamma='scale'),
	]
	
	results = rsltdict()
	run_regressions.results = results # save progress
	
	for incl_Us in ['norm'+str(norm)] + ([None] if include_expensive else []):
		data = make_learning_problem(TG, offset = offset, incl_Us = incl_Us, testfraction=test_fraction)

		for predictor in predictors + (expensive_predictors if include_expensive else []):
			try:
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					e = error_multidim(predictor, data)
				results[fz(incl_Us = incl_Us, name=predictor_name(predictor))] = e

				print('Score [%0.8f] computed for '%e, predictor_name(predictor), ' for inclU = ', incl_Us)
			except Exception as e:
				print('!!',predictor_name(predictor),'!!\t ', e)
				
	return results
	
def lmap(f,x):
	return list(map(f,x))
	


def calc_err( Qs, f, Us = None, nsamples = 1):
	series = []
	for i in ilinspace(0, len(Qs)-1, nsamples):
		ii = int(i)
		prediction = f(Qs[ii]) if Us is None else f( Q=Qs[ii], U =Us[ii] )
		series.append([ mean_squared_error( prediction, Q) for Q in Qs ])

	return series if nsamples > 1 else series[0]

def ptwise_convolve( As, ker ):
	""" Convolve a list of matrices by this kernel, entry-wise, across time """
	return np.apply_along_axis(lambda seq : np.convolve(seq, ker, 'valid'), 0, As)
	



	

def avg_err_vs_dt(Qs, f, Us = None, times = None, kernel = [1]) :
	diffs = defaultdict(list)
	kernel = np.array(kernel) / sum(kernel)
	
	if times == None:
		times = np.arange(len(Qs))
	
	for i1, (t1, Q1) in enumerate(zip(times, Qs)):
		for t2, Q2 in zip(times, Qs):
			# use Q1 to predict Q2; => dT = t2 - t
			prediction = f(Q1) if Us is None else f( Q=Q1, U=Us[i1] )
			diffs[t2-t1].append( Q2 - prediction )
		
	avgs = { dt : (np.mean(ptwise_convolve(ds, kernel), axis=0)**2 ).mean() for dt, ds in diffs.items() } 
	
	T, E = zip(*sorted(avgs.items()))
		
	return T,E
