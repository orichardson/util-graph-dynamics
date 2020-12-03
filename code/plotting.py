import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
from choice import stoch3, mix
from experiments import avg_err_vs_dt

from datetime import timedelta, datetime

def ilinspace(start, end, number):
	"""
	Calculate the "interior" linspace; this is like np.linspace, except
	it adds
	"""
	offset = (end - start) / (2*number)
	return np.linspace(start+offset, end-offset, number)

def plot_errors(errs, times = None, title = None, 
		xlabel = None, ylabel = None, cmap="jet",
		show = True, box = True, cbar_labels='time',
		max_xticks = 7, max_bars = 5, convolve = [1,2,2.1,2,1], 
		reuse = False, msize = 5
):
	convolve = np.array(convolve) / sum(convolve)
	
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	
	n_items, npts = np.shape(errs)
	nbars = min(max_bars, n_items)
	n_xticks = min(max_xticks, npts)

	## create a fake colorbar
	#cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])
	#cmap = plt.get_cmap('inferno', nbars)
	if type(cmap) is str:
		cmap = plt.get_cmap(cmap, nbars)
	elif type(cmap) is list:
		cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycolors',cmap)

	norm = matplotlib.colors.BoundaryNorm(np.arange(nbars+1) +0.5, nbars)
	sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
	sm.set_array([])
	###
	
	Xs = times if times else list(range(npts))
	
	
	fig, ax = (plt.gcf(), plt.gca()) if reuse else plt.subplots(dpi=200) 
	
	for i in range(nbars):
		Ys = errs[ (i*n_items) // nbars ]

		cv_Xs = np.convolve(Xs, convolve, 'valid')
		cv_Ys = np.convolve(Ys, convolve, 'valid')
		
		ax.plot(Xs, Ys, 'o', markersize=msize, alpha=0.7, 
			color=cmap(i/nbars),markeredgewidth=0.2, markeredgecolor='black')
		ax.plot(cv_Xs, cv_Ys, '-', alpha=0.4, color=cmap(i/nbars), linewidth=3.0)
		
	
	if times:
		offset = npts // (n_xticks * 2)
		time_tick_pts = ilinspace(times[0], times[-1], n_xticks)
		#time_tick_pts = [ times[int(idx)] for idx in ilinspace(0, npts, n_xticks) ]
		#time_tick_pts = [ times[ (i*npts) // n_xticks + offset] for i in range(n_xticks) ]
		ax.set_xticks(time_tick_pts)
		
		if np.min(times) < 18000: # Before 1 Jan 1970 => amount of time
			ax.set_xticklabels([ timedelta(seconds=t).days \
				for t in time_tick_pts])
			xlabel = "$\Delta$T (days)"
		else:    
			ax.set_xticklabels([ datetime.fromtimestamp(t).strftime('%d %b %y') \
				for t in time_tick_pts], rotation=25)

			
	if title: fig.suptitle(title)
	if xlabel: ax.set_xlabel(xlabel)
	if ylabel: ax.set_ylabel(ylabel)
	
	if not box: plt.box(False)
	
	if reuse:
		cbar = fig.colorbar(sm, ticks=None, fraction=0.05)
		cbar.minorticks_off()
		cbar.ax.set_yticklabels([''*nbars])
		#plot_errors.cbar.set_padding(0)
	else:
		cbar = fig.colorbar(sm, ticks=np.arange(1., nbars+1))
		if cbar_labels == 'time' and times:
			cbar.ax.set_yticklabels(
				[ datetime.fromtimestamp(int(t)).strftime('%d %b %y') 
					for t in ilinspace(times[0], times[-1], nbars) ] )
		elif type(cbar_labels) in { np.array, list } :
			print( ilinspace(0, n_items-1, nbars))
			cbar.ax.set_yticklabels(
				[ cbar_labels[int(np.round(i))] for i in ilinspace(0, n_items-1, nbars) ])
	
	plot_errors.cbar = cbar

	if show : plt.show()
	return fig


def heatplots( *matrices, names = None):
	plt.matshow(1/(1E-15+np.vstack(matrices)), cmap='inferno')
	plt.axis('off')
	plt.show() 

def mshow( M, ms_delay=100):
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
		
def plot_errs_vs_dt_geni(TG, f, mixer=mix, v_name="\\alpha",
	vals = (0,0.05,0.07,0.1,0.15,0.25,0.4,0.5), **kw_for_plot
) :
	fs,labels = zip(* ( (mixer(f, v), "$%s=%.2f$" % (v_name,v)) for v in vals ))
	return plot_errs_vs_dt(TG, fs, labels=list(labels), **kw_for_plot)


def plot_errs_vs_dt(TG, fs, labels=None, width=40, return_computation=False, **kw_for_plot):
	def chop(*ars ):
		ends = (len(ars[0]) - width) //2
		return [a[ends:-ends] for a in ars]
	
	Us = TG.Us()
	Us = Us / Us.max() * 15
	
	Qs = stoch3(TG.Ws())
	
	Es = []
	for f in fs:
		T,E = avg_err_vs_dt(Qs, f, Us=Us, times=TG.times)
		Es.append(E)
		
	plot_errors(chop(*Es), chop(T)[0], cbar_labels=labels, max_bars = 100, **kw_for_plot)
	
	if return_computation:
		return T, Es
