'''
Misc functions that might be useful to reuse.
'''

import numpy
import pylab
import os
import os.path
import subprocess
import glob
import subprocess
import cProfile, pstats, io

def profile(fnc):
    '''
    A decorator that uses cProfile to profile a function
    This is lifted from https://osf.io/upav8/
    
    Required imports:
    import cProfile, pstats, io
    
    To use, decorate function of interest by putting @gnosim.utils.misc.profile above
    its definition.
    
    Meanings:
    ncalls  - for the number of calls.  When there are two numbers (for example 3/1), 
              it means that the function recursed. The second value is the number 
              of primitive calls and the former is the total number of calls. Note 
              that when the function does not recurse, these two values are the same, 
              and only the single figure is printed.
    tottime - for the total time spent in the given function (and excluding time made 
              in calls to sub-functions)
    percall - is the quotient of tottime divided by ncalls
    cumtime - is the cumulative time spent in this and all subfunctions (from invocation 
              till exit). This figure is accurate even for recursive functions.
    percall - is the quotient of cumtime divided by primitive calls
    filename:lineno(function) - provides the respective data of each function
    '''
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s)
        ps.strip_dirs().sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

def getGitRootPath():
    '''
    Returns the root directory of the git repository.  Helpful for pathing within the git.

    Returns
    -------
    root_path : str
        The string cooresponding to the root directory of the git.  This will contain the name of the git,
        i.e. PATH_TO_GIT/gnosim
        So if you wish to just have the path of the folder containing gnosim you should remove the last bit.
    '''
    root_path = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
    if root_path == '':
        print('Returning path from system variable GNOSIM_DIR which should be one directory up from the main git root path.')
        root_path = sys.path.append(os.environ['GNOSIM_DIR'])
    return root_path

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

