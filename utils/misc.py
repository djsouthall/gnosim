'''
Misc functions that might be useful to reuse.
'''

import numpy
import pylab


def getColorMap(n):
	'''
	Returns a list of unique colors which can be used when plotting.  Does not include white.

	Parameters
	----------
	n : int
		The number of colors to include in the color map.

	Returns
	-------
		colors : list of tuples
			Each tuple represents a color for plotting.
	'''
	colormap = pylab.cm.gist_ncar #nipy_spectral, Set1,Paired   
	colors = [colormap(i) for i in numpy.linspace(0, 1,n+1)]
	return colors