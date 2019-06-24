'''
This file contains the properties of any ice model (Greenland v.s. Antarctic ice)
As well as functions for using these models.

This combines what were previously rhe antarctic.py and greenland.py files.
'''
import os
import sys
sys.path.append(os.environ['GNOSIM_DIR'])
import numpy
import pylab
import scipy.interpolate
import gnosim.utils.constants
import scipy.optimize

pylab.ion()

############################################################
def getAcceptedIceModels():
    '''
    Returns an array of the acceptable ice model labels.

    Returns
    -------
    ice_models : numpy.ndarray of str
        Array of the acceptable ice model labels.
    Sources
    -------
    antarctica_newcomb
        From South Pole Ice Characteristics, http://icecube.wisc.edu/~mnewcomb/radio/#icerefraction -> https://icecube.wisc.edu/~araproject/radio/#icerefraction
        Talking with Abby we have decided that the parameteric fit they give for attenuation is far to specific to
        try and use for general use.   This source is relatively outdated.  For more recent attenuation models
        try antarctica_allison.
    antarctica_allison
        This model uses temperature measurements from antarctica_newcomb
        The density comes from spicecore measurements fit to a function: density = 917 - a * exp(depth/b)
        The attenuation comes from the calculation described in the paper below by P Allison et al.
        From the paper Design and Initial Performance of the Askaryan Radio Array Prototype EeV Neutrino Detector at the South Pole
        https://arxiv.org/pdf/1105.2854.pdf
    greenland_avva
        This model uses temperature data from:
        Johnsen, Sigfus J., et al. "Greenland palaeotemperatures derived from GRIP bore hole temperature and ice core isotope profiles." Tellus B: Chemical and Physical Meteorology 47.5 (1995): 624-629.
        And density data from:
        https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/jgrf.20089
        Robert J. Arthern, Hugh F. J. Corr, Fabien Gillet-Chaulet, Robert L. Hawley, nd Elizabeth M. Morris
        And attenuation data from Figure 7 of: 
        An in situ measurement of the radio-frequency attenuation in ice at Summit Station, Greenland - Avva et al.
        https://arxiv.org/pdf/1409.5413.pdf
    greenland_arthern
        This model uses temperature data from:
        Johnsen, Sigfus J., et al. "Greenland palaeotemperatures derived from GRIP bore hole temperature and ice core isotope profiles." Tellus B: Chemical and Physical Meteorology 47.5 (1995): 624-629.
        And density data from:
        https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/jgrf.20089
        Robert J. Arthern, Hugh F. J. Corr, Fabien Gillet-Chaulet, Robert L. Hawley, nd Elizabeth M. Morris
        And calculates attenuation from the temperature using the fit described in newcomb.  This fit is highly site specific
        and probably shouldn't be used here, it is recommended that you use 'greenland_avva' unless you know better.
    '''
    ice_models = numpy.array(['antarctica_newcomb', 'greenland_arthern','parametric_greenland','empirical_greenland','ross','polar','arthern','antarctica_allison','greenland_avva'])
    return ice_models


def checkIceModel(ice_model_in):
    '''
    If the input is in the accepted list of ice models (which can be obtained from getAcceptedIceModels), then it will
    be returned unchcanged, unless it is an accepted ice model that has been renamed (i.e. arthern -> greenland),
    in which case the new name will be returned.  If this input is not one of the accepted ice models then the default
    ice model will be returned (currently the first element in the array returned by getAcceptedIceModels).

    Parameters
    ----------
    ice_model_in : str
        Input ice model to be checked.

    Returns
    -------
    ice_model_out : str
        Output ice model, which will be an accepted ice model.
    '''
    possible_models = getAcceptedIceModels()
    ice_model_default = possible_models[0]
    if ~numpy.isin(ice_model_in, possible_models):
        print("Requested ice model %s not in available ice model list, defaulting to %s"%(ice_model_in,ice_model_default))
        ice_model_out = ice_model_default
    elif ice_model_in == 'polar':
        ice_model_out = 'antarctica_newcomb' #For some backwards compatibility
    elif ice_model_in == 'arthern':
        ice_model_out = 'greenland_arthern'  #For some backwards compatibility
    elif ice_model_in == 'antarctica':
        ice_model_out = 'antarctica_allison'
    elif ice_model_in == 'greenland':
        ice_model_out = 'greenland_avva'
    else:
        ice_model_out = ice_model_in
    return ice_model_out

class Ice:
    '''
    Stores the attributes, information, and functions for the ice used in a simulation or related proceses.
    
    Parameters
    ----------
    ice_model : str
        The desired ice model (label) to be loaded when the ice is initiated.  This will be checked with checkIceModel to ensure
        it is an acceptable ice model.
    suppress_fun : bool, optional
        If fun is not suppressed (suppress_fun = False) then a nice little ASCII image representing the selected ice model will be 
        printed to the command line, if True then this will not be printed.  (Default is False).

    Attributes
    ----------
    ice_model : str
        The ice model (label) that is loaded for the ice.
    ice_thickness : float
        The thickness of the ice.
    profile_deep : numpy.ndarray of floats, optional
        Density (Mg m^-3) data and corresponding depths (m) from a particular ice model.  Only present for certain emperical models.
    elevation_deep : numpy.ndarray of floats, optional
        The depth data points for the density model (now seperated from profile_deep).  Given in m where negative values are below 
        the surface.  Only present for certain emperical models.
    density_deep : numpy.ndarray of floats or float
        The density data points from the density model in profile_deep, except converted to nucleons/m^3.  For certain models this
        might be a single value.
    profile_firn : numpy.ndarray of floats, optional
        Density (kg m^-3) data and corresponding depths (m) from a particular ice model in just the firn region.  Only present for 
        certain emperical models.  Depths below the surface are positive values.
    elevation_firn : numpy.ndarray of floats, optional
        The depth data points for the firn density data (now seperated from profile_firn).  Given in m where negative values are below 
        the surface.  Only present for certain emperical models.
    density_firn : numpy.ndarray of floats or float
        The firn density data points from the density model in profile_firn, except converted to nucleons/m^3. 
    f_density : scipy.interpolate.interp1d cubic interpolation function
        Funtion that returns the density in nucleons/m^3.  For certain emperical ice models this is a cubic interpolation function 
        of for desnity_deep.  For others it might represent a parametric density model.
    z_data : numpy.ndarray of floats
        The depth data for the temperature model given in m.
    t_data : numpy.ndarray of floats
        The temperature data for the temperature model gien in C.
    f_temperature : scipy.interpolate.interp1d cubic interpolation function
        Tbe cubic interpolation function for the temperature model.
    '''
    def __init__(self, ice_model, suppress_fun = False):
        self.ice_model = checkIceModel(ice_model)

        images =    {
                    'antarctica_newcomb'    :"\n                ++++++++++              \n                ++++++++++++++          \n               ++++++++++++++++++++     \n              +++++++ANTARCTICA++++     \n +            ++++++++++++++++++++++    \n ++++++        +++++++++++++++++++++    \n   ++ ++     + +++++++++++++++++++      \n    + +++   + +++++++ANTARCTICA+++++    \n     +++ +  ++++++++++++++++++++++++++  \n     ++++++++++++++++++++++++++++++++++ \n     ++++++++++++++++++++++++++++++++++ \n    +++++++++++++++++ANTARCTICA++++++++ \n       ++++++++++ +++++++++++++++++++++ \n       ++++++++++   ++++++++++++++++++  \n       +++++++++      ++++++++++++++++  \n          +++++ +     +++++++++++++++   \n                      ++++++++++++++    \n                       ++++++++++++     \n                       +++++++++++      \n                      ++++++++++\n",
                    'antarctica_allison'    :"\n                ++++++++++              \n                ++++++++++++++          \n               ++++++++++++++++++++     \n              +++++++ANTARCTICA++++     \n +            ++++++++++++++++++++++    \n ++++++        +++++++++++++++++++++    \n   ++ ++     + +++++++++++++++++++      \n    + +++   + +++++++ANTARCTICA+++++    \n     +++ +  ++++++++++++++++++++++++++  \n     ++++++++++++++++++++++++++++++++++ \n     ++++++++++++++++++++++++++++++++++ \n    +++++++++++++++++ANTARCTICA++++++++ \n       ++++++++++ +++++++++++++++++++++ \n       ++++++++++   ++++++++++++++++++  \n       +++++++++      ++++++++++++++++  \n          +++++ +     +++++++++++++++   \n                      ++++++++++++++    \n                       ++++++++++++     \n                       +++++++++++      \n                      ++++++++++\n",
                    'ross'                  :"\n                ++++++++++              \n                ++++++++++++++          \n               ++++++++++++++++++++     \n              +++++++ANTARCTICA++++     \n +            ++++++++++++++++++++++    \n ++++++        +++++++++++++++++++++    \n   ++ ++     + +++++++++++++++++++      \n    + +++   + +++++++ANTARCTICA+++++    \n     +++ +  ++++++++++++++++++++++++++  \n     ++++++++++++++++++++++++++++++++++ \n     ++++++++++++++++++++++++++++++++++ \n    +++++++++++++++++ANTARCTICA++++++++ \n       ++++++++++ +++++++++++++++++++++ \n       ++++++++++   ++++++++++++++++++  \n       +++++++++      ++++++++++++++++  \n          +++++ +     +++++++++++++++   \n                      ++++++++++++++    \n                       ++++++++++++     \n                       +++++++++++      \n                      ++++++++++\n",
                    'greenland_arthern'     :"\n                          +++++         \n                       +++ ++++++       \n           + +++  +  +++++++++++++      \n         + ++++ ++++ ++++++++  +        \n       +++++++++++++++++++++++  ++  +   \n       +++++++++++++++++++++++++ ++++++ \n      ++++++++++++++++++++++++ ++++++   \n ++++++++++++++GREENLAND++++++++++      \n  ++++++++++++++++++++++++++++++++      \n ++ ++++++++++++++++++++++++++++++      \n  +++++++++++++++++++++++++++++++       \n + + ++++++++++GREENLAND+++++++++       \n       +++++++++++++++++++++++++++      \n       ++++++++++++++++++++++++         \n        +++++++++++++++++++++++++       \n        +++++++GREENLAND++++++++        \n        ++++++++++++++++++++++          \n       ++++++++++++++++++++             \n       + ++++++++++++++++++             \n       + ++++++GREENLAND+++++           \n      ++++++++++++++++++++  ++          \n        ++++++++++++++++++++++          \n        +++++++++++++++++++             \n       +++++++++++++++                  \n     + +++++++++++++                    \n      +++++++++++++                     \n       ++++++++                         \n      ++++++++                          \n       ++++++                           \n       +++++                            \n         +++                            \n                                        \n",
                    'greenland_avva'        :"\n                          +++++         \n                       +++ ++++++       \n           + +++  +  +++++++++++++      \n         + ++++ ++++ ++++++++  +        \n       +++++++++++++++++++++++  ++  +   \n       +++++++++++++++++++++++++ ++++++ \n      ++++++++++++++++++++++++ ++++++   \n ++++++++++++++GREENLAND++++++++++      \n  ++++++++++++++++++++++++++++++++      \n ++ ++++++++++++++++++++++++++++++      \n  +++++++++++++++++++++++++++++++       \n + + ++++++++++GREENLAND+++++++++       \n       +++++++++++++++++++++++++++      \n       ++++++++++++++++++++++++         \n        +++++++++++++++++++++++++       \n        +++++++GREENLAND++++++++        \n        ++++++++++++++++++++++          \n       ++++++++++++++++++++             \n       + ++++++++++++++++++             \n       + ++++++GREENLAND+++++           \n      ++++++++++++++++++++  ++          \n        ++++++++++++++++++++++          \n        +++++++++++++++++++             \n       +++++++++++++++                  \n     + +++++++++++++                    \n      +++++++++++++                     \n       ++++++++                         \n      ++++++++                          \n       ++++++                           \n       +++++                            \n         +++                            \n                                        \n",
                    'parametric_greenland'  :"\n                          +++++         \n                       +++ ++++++       \n           + +++  +  +++++++++++++      \n         + ++++ ++++ ++++++++  +        \n       +++++++++++++++++++++++  ++  +   \n       +++++++++++++++++++++++++ ++++++ \n      ++++++++++++++++++++++++ ++++++   \n ++++++++++++++GREENLAND++++++++++      \n  ++++++++++++++++++++++++++++++++      \n ++ ++++++++++++++++++++++++++++++      \n  +++++++++++++++++++++++++++++++       \n + + ++++++++++GREENLAND+++++++++       \n       +++++++++++++++++++++++++++      \n       ++++++++++++++++++++++++         \n        +++++++++++++++++++++++++       \n        +++++++GREENLAND++++++++        \n        ++++++++++++++++++++++          \n       ++++++++++++++++++++             \n       + ++++++++++++++++++             \n       + ++++++GREENLAND+++++           \n      ++++++++++++++++++++  ++          \n        ++++++++++++++++++++++          \n        +++++++++++++++++++             \n       +++++++++++++++                  \n     + +++++++++++++                    \n      +++++++++++++                     \n       ++++++++                         \n      ++++++++                          \n       ++++++                           \n       +++++                            \n         +++                            \n                                        \n",
                    'empirical_greenland'   :"\n                          +++++         \n                       +++ ++++++       \n           + +++  +  +++++++++++++      \n         + ++++ ++++ ++++++++  +        \n       +++++++++++++++++++++++  ++  +   \n       +++++++++++++++++++++++++ ++++++ \n      ++++++++++++++++++++++++ ++++++   \n ++++++++++++++GREENLAND++++++++++      \n  ++++++++++++++++++++++++++++++++      \n ++ ++++++++++++++++++++++++++++++      \n  +++++++++++++++++++++++++++++++       \n + + ++++++++++GREENLAND+++++++++       \n       +++++++++++++++++++++++++++      \n       ++++++++++++++++++++++++         \n        +++++++++++++++++++++++++       \n        +++++++GREENLAND++++++++        \n        ++++++++++++++++++++++          \n       ++++++++++++++++++++             \n       + ++++++++++++++++++             \n       + ++++++GREENLAND+++++           \n      ++++++++++++++++++++  ++          \n        ++++++++++++++++++++++          \n        +++++++++++++++++++             \n       +++++++++++++++                  \n     + +++++++++++++                    \n      +++++++++++++                     \n       ++++++++                         \n      ++++++++                          \n       ++++++                           \n       +++++                            \n         +++                            \n                                        \n"
                    }
        if numpy.isin(self.ice_model, list(images.keys())):
            print('******************\nUsing ice model: %s\n******************'%self.ice_model)
            if suppress_fun == False:
                print(images[self.ice_model])
        else:
            print('******************\nUsing ice model: %s\n******************'%self.ice_model)


        ###
        #Defining ice depths (used in p_interact and prem)
        ###
        if numpy.isin(self.ice_model, ['antarctica_newcomb','ross','antarctica_allison']):
            self.ice_thickness = 3000. #m
        elif numpy.isin(self.ice_model, ['greenland_arthern','parametric_greenland','empirical_greenland','greenland_avva']):
            self.ice_thickness = 3000. #m
        else:
            self.ice_thickness = 3000. #m
        ###
        #Defining density profiles
        ###

        if numpy.isin(self.ice_model, ['antarctica_newcomb','ross']):
            # From South Pole Ice Characteristics, http://icecube.wisc.edu/~mnewcomb/radio/#icerefraction -> https://icecube.wisc.edu/~araproject/radio/#icerefraction
            # Density (Mg m^-3)
            # Elevation, negative is down (m)
            '''
            density_depth_newcomb, density_value_newcomb = zip(*[   [-0, 340.45281564985586],
                                                            [-4.911075949367088, 420.41279565630263],
                                                            [-9.446202531645568, 489.37312152072965],
                                                            [-20.117088607594937, 550.3004782771569],
                                                            [-39.324683544303795, 629.961576368985],
                                                            [-59.59936708860758, 680.6307003406309],
                                                            [-79.87405063291135, 739.8735797607743],
                                                            [-100.14873417721515, 790.6555163041111],
                                                            [-119.88987341772147, 844.7156398100395],
                                                            [-139.6310126582278, 890.5263440110814],
                                                            [-160.06898734177224, 900.9345769674584],
                                                            [-165.4272151898738, 910.6305047609922],
                                                            [-169.94050632911453, 910.4113137811773],
                                                            [-180.24113924050707, 921.0183457185919],
                                                            [-184.4050632911392, 920.6573566693093],
                                                            [-195.3417721518987, 920.6661381515329],
                                                            [-199.59493670886067, 920.6695531723976],
                                                            [-299.69430379746825, 921.1142590188786],
                                                            [-400.00063291139224, 921.1947986270224],
                                                            [-499.23987341772136, 921.2744814308241],
                                                            [-599.5462025316455, 921.3550210389679]])
            densityFit = lambda z, a, b : 917.0 - a*numpy.exp(z/b)
            popt_newcomb, pcov_newcomb = scipy.optimize.curve_fit(densityFit, density_depth_newcomb, density_value_newcomb,p0 = [500,50])
            self.f_density = lambda z : 917.0 - (popt_newcomb[0])*numpy.exp(z/popt_newcomb[1])
            '''
            self.profile_deep = numpy.array([[0.7230343614334508, -63.09751434034433],
                                         [0.7864140997923716, -86.0420650095607],
                                         [0.8212633137439305, -103.2504780114723],
                                         [0.8398757018985611, -114.72275334608048],
                                         [0.8685699534228556, -137.6673040152964],
                                         [0.8742983634188128, -143.40344168260026],
                                         [0.8795439563972824, -149.13957934990412],
                                         [0.88874603766566, -160.6118546845123],
                                         [0.8998398145522166, -177.8202676864248],
                                         [0.9083582711221764, -195.0286806883364],
                                         [0.9167228067791977, -217.97323135755323],
                                         [0.9237783972356667, -246.653919694073],
                                         [0.9290136117978872, -281.0707456978971],
                                         [0.933129801265018, -332.69598470363326],
                                         [0.9351232837375564, -390.0573613766737],
                                         [0.9361581618621482, -476.0994263862335],
                                         [0.936450650352632, -573.613766730402],
                                         [0.9365299601324286, -757.1701720841302],
                                         [0.9365347474915189, -952.198852772467],
                                         [0.936534974411263, -1101.3384321223714],
                                         [0.9365349990985374, -1319.311663479924],
                                         [0.9365349999002391, -1462.715105162524],
                                         [0.9365349999994935, -1806.8833652007647],
                                         [0.9365349999999864, -2042.0650095602296],
                                         [0.936534999999999, -2214.1491395793496],
                                         [0.9365349999999999, -2334.6080305927344],
                                         [0.936535, -2466.5391969407265],
                                         [0.936535, -2621.414913957935],
                                         [0.936535, -2845.1242829827916],
                                         [0.936535, -3000],
                                         [0.936535, -10000000000.0]]) #Filler
            self.elevation_deep = self.profile_deep.transpose()[1]
            self.density_deep = self.profile_deep.transpose()[0] * 1.e3 / gnosim.utils.constants.mass_proton # nucleons m^-3
            self.f_density = scipy.interpolate.interp1d( self.elevation_deep[::-1] , self.density_deep[::-1], kind='cubic', bounds_error = None, fill_value= 'extrapolate' ) #extroplate appears to do a good job here to 0m - DS
        elif numpy.isin(self.ice_model, ['antarctica_allison']):
            '''
            This uses the spicecore data given by Cosmin (no official source yet).  It fits it with a simple exponential
            which fixes the asymptotic behaviour to lead to 917.  These are given in kg/m^3 and then f_density is converted
            to neclei/m^3 using gnosim.utils.constants.mass_proton.
            '''






























































































































































































            densityFit = lambda z, a, b : 917.0 - a*numpy.exp(z/b)

            self.popt_spicecore, self.pcov_spicecore = numpy.array([570.05140003,  65.03495904]) , numpy.array([[41.38266317, -4.74136409],[-4.74136409,  0.89063516]]) #Fit of data to above


            def f_density(z):
                '''
                This is the density for Antarctica calculated from the spicecore data.

                Parameters
                ----------
                z : numpy.ndarray of float
                    These are the depths for which to calculate the density.
                Returns
                -------
                density : numpy.ndarray of float
                    This is the corresponding density values in units of nuclei / m^3.
                '''
                if type(z) is list:
                    z = numpy.array(z)
                density = (917.0 - (self.popt_spicecore[0])*numpy.exp(z/self.popt_spicecore[1]))/gnosim.utils.constants.mass_proton #This is in nuclei/m^3
                if type(z) is not numpy.ndarray:
                    try:
                        if z > 0.:
                            density = 0.
                    except Exception as e:
                        print('Error in f_density.  This should be given an numpy.ndarray of z values, or a single float.')
                        print(e)
                else:
                    density[z > 0] = 0.0
                return density
            self.f_density = f_density

        elif numpy.isin(self.ice_model, ['empirical_greenland']):
            # From
            # Density (Mg m^-3)
            # Elevation, negative is down (m)
            self.profile_deep = numpy.array([#[0.87, -63.09751434034433],
                                        #[0.8798869143780291, -86.0420650095607],
                                        #[0.8898707592891761, -103.2504780114723],
                                        [0.8998546042003231, -114.72275334608048],
                                        #[0.9101292407108239, -137.6673040152964],
                                        [0.911583198707593, -137.6673040152964],
                                        [0.9132310177705978, -143.40344168260026],
                                        [0.9142003231017771, -149.13957934990412],
                                        [0.9149757673667206, -160.6118546845123],
                                        [0.9156542810985461, -177.8202676864248],
                                        [0.9163327948303716, -195.0286806883364],
                                        [0.9169143780290792, -217.97323135755323],
                                        [0.9173990306946689, -246.653919694073],
                                        [0.9178836833602585, -281.0707456978971],
                                        [0.9182714054927302, -332.69598470363326],
                                        [0.918659127625202, -390.0573613766737],
                                        [0.9190468497576737, -476.0994263862335],
                                        [0.9193376413570276, -573.613766730402],
                                        [0.9197253634894992, -757.1701720841302],
                                        [0.9201130856219709, -952.198852772467],
                                        [0.9203069466882068, -1101.3384321223714],
                                        [0.9205008077544427, -1319.311663479924],
                                        [0.9205977382875606, -1462.715105162524],
                                        [0.9206946688206785, -1806.8833652007647],
                                        [0.9206946688206785, -2042.0650095602296],
                                        [0.9205977382875606, -2214.1491395793496],
                                        [0.9201130856219709, -2334.6080305927344],
                                        [0.9196284329563813, -2466.5391969407265],
                                        [0.9188529886914378, -2621.414913957935],
                                        [0.9175928917609047, -2845.1242829827916],
                                        [0.9167205169628433, -3000], 
                                        [0.9167205169628433, -1.e10]]) # Filler
            self.elevation_deep = self.profile_deep.transpose()[1]
            self.density_deep = self.profile_deep.transpose()[0] * 1.e3 / gnosim.utils.constants.mass_proton # nucleons m^-3

            # From 
            # Depth, positive is down (m)
            # Density (kg m^-3)
            self.profile_firn = numpy.array([[-1.e10, 0.], # Filler
                                        [-1.e-10, 0.], # Filler
                                        [0., 256.9169960474308],
                                        [0.46728971962616633, 270.75098814229256],
                                        [0.778816199376947, 292.49011857707507],
                                        [1.5576323987538898, 322.1343873517786],
                                        [2.4922118380062313, 341.89723320158106],
                                        [3.4267912772585656, 371.5415019762846],
                                        [4.8286604361370715, 401.1857707509881],
                                        [6.230529595015572, 428.85375494071144],
                                        [7.788161993769471, 458.498023715415],
                                        [9.501557632398756, 494.0711462450593],
                                        [11.682242990654204, 521.7391304347826],
                                        [14.01869158878504, 541.501976284585],
                                        [17.289719626168225, 565.2173913043478],
                                        [21.028037383177566, 584.9802371541502],
                                        [23.98753894080996, 596.8379446640316],
                                        [27.258566978193137, 616.6007905138341],
                                        [31.152647975077876, 644.2687747035573],
                                        [37.22741433021806, 662.0553359683794],
                                        [42.83489096573209, 691.699604743083],
                                        [50.155763239875384, 719.3675889328063],
                                        [56.54205607476635, 745.0592885375495],
                                        [65.42056074766354, 776.6798418972331],
                                        [73.05295950155762, 796.4426877470355],
                                        [79.12772585669781, 820.1581027667984],
                                        [87.85046728971962, 847.8260869565217],
                                        [95.4828660436137, 861.6600790513833]])
                                        #[100, 867.5889328063241]])

            self.elevation_firn = -1. * self.profile_firn.transpose()[0]
            self.density_firn = self.profile_firn.transpose()[1] / gnosim.utils.constants.mass_proton # nucleons m^-3

            self.f_density = scipy.interpolate.interp1d(numpy.concatenate([self.elevation_deep[::-1], self.elevation_firn[::-1]]), 
                                                   numpy.concatenate([self.density_deep[::-1], self.density_firn[::-1]]), 
                                                   kind='linear', bounds_error=False, fill_value=0.)
        
        ###
        #Defining temperature profiles
        ###

        if numpy.isin(self.ice_model, ['antarctica_newcomb','ross','antarctica_allison']):
            # Berkeley Group (antarctic)
            self.t_data, self.z_data = zip(*[[-51.0696, 0.0],
                                   [-50.8110708872, -96.0269567899],
                                   [-50.5632416462, -185.024005638],
                                   [-50.3384290394, -261.322292208],
                                   [-50.0121098059, -363.04453156],
                                   [-49.6202011626, -471.122759107],
                                   [-49.1394644123, -585.530546624],
                                   [-48.5748721854, -699.925120028],
                                   [-48.0672072572, -788.895740651],
                                   [-47.4471795627, -884.209135357],
                                   [-46.2678790842, -1036.71320971],
                                   [-45.2748570915, -1144.72536669],
                                   [-43.5441984447, -1303.57221513],
                                   [-42.4200531809, -1392.52962164],
                                   [-39.902500389, -1564.15451702],
                                   [-38.1748023155, -1665.88997049],
                                   [-36.6314481824, -1748.58388759],
                                   [-34.9561656655, -1831.30423292],
                                   [-31.802185966, -1971.37382725],
                                   [-29.5748535016, -2060.54265956],
                                   [-27.697431353, -2130.6567414],
                                   [-25.3298479683, -2213.52244197],
                                   [-21.3562903244, -2341.09148571],
                                   [-18.3255166962, -2430.43210148],
                                   [-11.8672562746, -2602.83662952],
                                   [-4.55379399077, -2775.34687046],
                                   [3.66965515649, -2947.89675373],
                                   [7.97033030664, -3030.98709422]])

            self.f_temperature = scipy.interpolate.interp1d(self.z_data,  self.t_data, bounds_error=False, fill_value=-50.)

            if numpy.isin(self.ice_model, ['antarctica_newcomb']):
                #I want to add a scale factor to each of these such that they match the curves given in the papers
                #at the respective frequencies those papers present. 
                #The lines were used (not here but with proper setup) to calculate the number below.
                #z_scale = numpy.arange(-2000,0)
                #scale_antarctica = numpy.mean(numpy.divide(ice_antarctica_allison.attenuationLength(z_scale, frequency=0.300), 0.5*ice_antarctica.attenuationLength(z_scale, frequency=0.300) ))
                self.scale_antarctica = 1.481860 #This was calculated such that the curve at 300MHz matches the antarctica_300MHz model.
                if self.scale_antarctica != 1.0:
                    print('WARNING! The attenuation is being scaled from the original mnewcomb websites formula to make it match with other works.')
                    print('The attenuation is scaled by a factor of %f'%self.scale_antarctica)

            if numpy.isin(self.ice_model, ['antarctica_allison']):
                '''
                Below is attenuation lengths taken from Figure 23 of: 
                Design and Initial Performance of the Askaryan Radio Array Prototype EeV Neutrino Detector at the South Pole - P Allison et al.
                https://arxiv.org/pdf/1105.2854.pdf

                I extrapolate it and fit with an order 7 polynomial which seems to do a really good job of capturing the features
                and extending the range to the full 0-3000m depths.  

                This is given for only 300 MHz, and cannot be reliably used for other frequencies.
                '''
                self.att_data, self.zatt_data = zip(*[  [264.6573448538895, -2625.694076197364],
                                                        [292.5367216433327, -2575.9530468719527],
                                                        [320.41772947336506, -2528.6683646737224],
                                                        [350.29169050820985, -2480.1938893357637],
                                                        [382.1592928431161, -2431.5658923023566],
                                                        [414.0281439434733, -2384.8185360381967],
                                                        [447.88966791354176, -2336.4592019718248],
                                                        [483.74533155718507, -2288.6968967210873],
                                                        [519.6011537742189, -2241.1734029966033],
                                                        [557.4509253767582, -2193.9603642562492],
                                                        [595.3010775554352, -2147.3204731789037],
                                                        [633.1516578822669, -2101.325373222443],
                                                        [672.9956960170933, -2054.900412518726],
                                                        [712.8398768679718, -2008.6903821886376],
                                                        [754.6778009589478, -1962.4803518585486],
                                                        [796.5137270252025, -1913.2612962976636],
                                                        [838.3522219803846, -1867.910987462088],
                                                        [882.1831757312012, -1820.6263052638574],
                                                        [924.019958093765, -1772.6968319447421],
                                                        [965.857726494503, -1726.2523321161498],
                                                        [1009.6885505034546, -1678.7722586691661],
                                                        [1051.5260464462763, -1631.9174372181926],
                                                        [1093.3629715248912, -1584.2028942727054],
                                                        [1135.2003247516614, -1537.1331424481036],
                                                        [1177.037133062598, -1489.242747378739],
                                                        [1218.8734613286342, -1440.6294046889882],
                                                        [1256.7240099407882, -1394.586542427277],
                                                        [1294.5743365501949, -1348.2093440288106],
                                                        [1332.4247265889576, -1301.9276702408454],
                                                        [1368.281024526163, -1255.1206110951225],
                                                        [1404.1374810367593, -1208.5523634756532],
                                                        [1438.0008286008201, -1162.9393619611988],
                                                        [1469.870750071564, -1117.8039834992514],
                                                        [1501.7399579620496, -1071.5939531691627],
                                                        [1533.6079170870848, -1023.5032820698261],
                                                        [1563.4839169226661, -978.0992406408436],
                                                        [1591.3661480331402, -932.6568187879989],
                                                        [1619.2475636233194, -885.9862233715637],
                                                        [1647.127755933057, -837.4733676097428],
                                                        [1675.0067249623535, -787.1182515025365],
                                                        [1700.8925963717843, -737.7354394665059],
                                                        [1724.7865934417919, -691.1671918470365],
                                                        [1748.6789254911987, -642.0914232019035],
                                                        [1772.5703061002619, -591.5827853992483],
                                                        [1794.4695609178111, -543.5100251643807],
                                                        [1814.3767375158636, -497.94478595517694],
                                                        [1834.28334324971, -451.5198252514598],
                                                        [1854.1899489835564, -405.09486454774265],
                                                        [1874.0945566926807, -355.6608786132291],
                                                        [1894.000020698115, -307.5164749204855],
                                                        [1913.9046284072397, -258.08248898597196],
                                                        [1933.8095215484673, -209.07836379871497],
                                                        [1953.7138438254888, -159.21451711694482],
                                                        [1973.6190223988197, -110.6402526769445],
                                                        [1993.5233446758416, -60.77640599517429],
                                                        [2013.4316630023063, -16.93060977499701]])

                self.f_att = scipy.interpolate.interp1d(self.zatt_data,  self.att_data, bounds_error=False, fill_value='extrapolate') #should set to zero above surface.  Could try and fit. 
                y = numpy.arange(-3000,500)
                fitparams = numpy.polyfit(y,self.f_att(y),7)
                self.f_att_poly = numpy.poly1d(fitparams)

        elif numpy.isin(self.ice_model, ['greenland_arthern','parametric_greenland','empirical_greenland','greenland_avva']):
            '''
            Johnsen, Sigfus J., et al. "Greenland palaeotemperatures derived from GRIP bore hole temperature and ice core isotope profiles." Tellus B: Chemical and Physical Meteorology 47.5 (1995): 624-629.
            '''

            self.t_data, self.z_data = zip(*[[-31.753821080914417, 0.],
                                   [-31.753821080914417, -96.02695678985128],
                                   [-31.687750517552747, -185.0240056380212],
                                   [-31.55983790688455, -261.3222922080781],
                                   [-31.43086816720257, -363.0445315597058],
                                   [-31.301634145267144, -471.1227591067259],
                                   [-31.296877064705104, -585.5305466237946],
                                   [-31.354490595956477, -699.9251200281897],
                                   [-31.41316125622164, -788.8957406510153],
                                   [-31.533938246046777, -884.2091353565606],
                                   [-31.714707307404307, -1036.7132097079682],
                                   [-31.897326344535962, -1144.7253666916267],
                                   [-32.14020173545346, -1303.572215125754],
                                   [-32.261243007532045, -1392.5296216359068],
                                   [-32.25727877373034, -1487.869444566797],
                                   [-32.191736774875565, -1564.1545170241816],
                                   [-32.000396423380174, -1665.8899704884818],
                                   [-31.68510769501828, -1748.5838875919483],
                                   [-31.245077743029555, -1831.3042329207594],
                                   [-30.116592520812226, -1971.3738272475002],
                                   [-29.239704003876138, -2060.5426595604104],
                                   [-28.301237721886974, -2130.6567413998146],
                                   [-27.175131039950664, -2213.5224419680217],
                                   [-25.049244593225566, -2341.0914857067346],
                                   [-23.36153812271506, -2430.4321014843845],
                                   [-19.612165793067, -2602.8366295203277],
                                   [-15.3638285689116, -2775.3468704576485],
                                   [-10.928379509315953, -2947.8967537329863],
                                   [-8.741972426551555, -3030.9870942166235]])

            #self.z_data = -1. * numpy.array(self.z_data)[::-1]                                                                                                                
            self.z_data = numpy.array(self.z_data)[::-1]
            self.t_data = numpy.array(self.t_data)[::-1]
            self.f_temperature = scipy.interpolate.interp1d(self.z_data, self.t_data, bounds_error=False, fill_value=-50.)

            if numpy.isin(self.ice_model,['greenland_arthern']):
                #I want to add a scale factor to each of these such that they match the curves given in the papers
                #at the respective frequencies those papers present. 
                #The lines were used (not here but with proper setup) to calculate the number below.
                #z_scale = numpy.arange(-2000,0)
                #scale_greenland = numpy.mean(numpy.divide(ice_greenland_avva.attenuationLength(z_scale, frequency=0.075), 0.5*ice_greenland.attenuationLength(z_scale, frequency=0.075) ))

                self.scale_greenland = 1.427441 #This was calculated such that the curve at 75MHz matches the greenland_75MHz model.
                if self.scale_greenland != 1.0:
                    print('WARNING! The attenuation is being scaled from the original mnewcomb websites formula to make it match with other works.')
                    print('The attenuation is scaled by a factor of %f'%self.scale_greenland)

            if numpy.isin(self.ice_model, ['greenland_avva']):
                '''
                Below is attenuation lengths taken from Figure 7 of: 
                An in situ measurement of the radio-frequency attenuation in ice at Summit Station, Greenland - Avva et al.
                https://arxiv.org/pdf/1409.5413.pdf

                I extrapolate it and fit with an order 7 polynomial which seems to do a really good job of capturing the features
                and extending the range to the full 0-3000m depths.  

                This curve is then shifted using the equation described by the paper:
                att_new = att_old -0.55*(freq_new_MHz - 75MHz).
                '''
                self.att_data, self.zatt_data = zip(*[  [458.6614500957296, -3013.0294996264656],
                                                            [472.8849302144898, -2987.519494049282],
                                                            [487.10841033325005, -2960.4470205952925],
                                                            [501.33189045201027, -2934.3130457617704],
                                                            [515.5553705707705, -2907.07058394594],
                                                            [529.7788506895307, -2882.7199600701765],
                                                            [544.0023308082909, -2855.426807114111],
                                                            [558.2258109270512, -2825.8024724437387],
                                                            [572.4492910458114, -2806.9930620062464],
                                                            [586.6727711645717, -2782.260727737126],
                                                            [600.8962512833318, -2758.743759360452],
                                                            [615.1197314020922, -2737.1347322139295],
                                                            [629.3432115208523, -2712.09153300046],
                                                            [643.5666916396126, -2690.928954330006],
                                                            [657.7901717583728, -2667.9176758824497],
                                                            [672.013651877133, -2644.2541307147276],
                                                            [686.2371319958933, -2625.2883717001164],
                                                            [700.4606121146535, -2601.2370087727436],
                                                            [714.6840922334137, -2584.902913894086],
                                                            [728.9075723521739, -2563.900958957156],
                                                            [743.1310524709342, -2542.6107363311644],
                                                            [757.3545325896944, -2513.8414329419106],
                                                            [771.5780127084547, -2495.552370112641],
                                                            [785.8014928272148, -2465.826042425165],
                                                            [800.0249729459752, -2447.972434812636],
                                                            [814.2484530647353, -2424.3369835298645],
                                                            [828.4719331834956, -2403.8248393696895],
                                                            [842.6954133022557, -2385.308582514294],
                                                            [856.918893421016, -2353.6504948681218],
                                                            [871.1423735397761, -2338.5006183138125],
                                                            [885.3658536585365, -2316.3944515509825],
                                                            [899.5893337772966, -2291.4239299964083],
                                                            [913.8128138960569, -2274.168233544033],
                                                            [928.0362940148173, -2258.869337252157],
                                                            [942.2597741335774, -2237.4655176131023],
                                                            [956.4832542523377, -2213.2229941207365],
                                                            [970.7067343710978, -2193.5524450358334],
                                                            [984.9302144898581, -2161.993296723619],
                                                            [999.0459409713549, -2133.7939828241733],
                                                            [1013.0539138155884, -2105.1260881072244],
                                                            [1027.6006548461387, -2076.5343104706817],
                                                            [1041.824134964899, -2050.085806273033],
                                                            [1056.0476150836591, -2021.3287176163667],
                                                            [1070.2710952024195, -1988.2109694260007],
                                                            [1084.4945753211796, -1955.8627493886406],
                                                            [1098.71805543994, -1920.1799073549219],
                                                            [1112.9415355587003, -1879.3668776344978],
                                                            [1127.1650156774604, -1834.7062071490443],
                                                            [1132.9837120896805, -517.0915330150211],
                                                            [1132.0139293543104, -613.9684888209472],
                                                            [1134.4060601015565, -685.6324953878086],
                                                            [1136.2163212075804, -409.8642381785312],
                                                            [1135.5697993840006, -750.1038351541164],
                                                            [1141.3884957962207, -1782.8632739022032],
                                                            [1139.5782346901965, -344.2083093426047],
                                                            [1139.4489303254807, -822.2822044412554],
                                                            [1144.6211049141207, -270.41344433098516],
                                                            [1144.6211049141207, -893.2823568431203],
                                                            [1150.4398013263408, -201.5078289231156],
                                                            [1149.7932795027607, -962.4014404265263],
                                                            [1151.0863231499206, -51.72644981647322],
                                                            [1154.3189322678209, -1725.0091695185731],
                                                            [1152.3793667970808, -119.44237840724847],
                                                            [1154.9654540914007, -1024.466515940712],
                                                            [1160.137628680041, -1083.0045874202879],
                                                            [1164.663281445101, -1664.3467028097884],
                                                            [1165.956325092261, -1149.4418735005215],
                                                            [1171.128499680901, -1212.071269660245],
                                                            [1171.7750215044812, -1607.0944470997017],
                                                            [1175.0076306223812, -1273.1983596389393],
                                                            [1177.5937179167013, -1339.030670537518],
                                                            [1177.5937179167013, -1544.5374389541305],
                                                            [1180.8263270346013, -1410.2697798522966],
                                                            [1180.5030661228113, -1478.694295342166],
                                                            [1134.1463414634145, -466.9139506297472],
                                                            [1132.1138211382113, -564.4933563469268]])
                self.f_att = scipy.interpolate.interp1d(self.zatt_data,  self.att_data, bounds_error=False, fill_value='extrapolate') #should set to zero above surface.  Could try and fit. 
                y = numpy.arange(-3000,500)
                fitparams = numpy.polyfit(y,self.f_att(y),7)
                self.f_att_poly = numpy.poly1d(fitparams)

    def density(self,z):
        '''
        Returns the density (nuclei m^-3) for a particular depth z (m).
        Source: http://www.iceandclimate.nbi.ku.dk/research/flowofice/densification/

        Parameters
        ----------
        z : numpy.ndarray of floats
            Depths to return densities for.

        Returns
        -------
        density : float
            The density in nuclei/m^3.
        '''
        
        if self.ice_model == 'parametric_greenland':
            density_surface = 50. # kg m^-3
            self.density_deep = 917. # kg m^-3
            scale_depth = 30. # m
        
            # Require that neutrino interacts in upper 3 km of ice
            density = (z <= 0.) * (z >= -3000.) \
                      * (self.density_deep - (self.density_deep - density_surface) * numpy.exp(z / scale_depth)) # kg m^-3
            return density / gnosim.utils.constants.mass_proton # convert from kg m^-3 to nucleons m^-3
        elif self.ice_model == 'empirical_greenland':
            return self.f_density(z) # nucleons m^-3
        elif self.ice_model == 'ross':
            return (z >= -500.) * (z <= 0.) * self.f_density(z) # nucleons m^-3
        elif self.ice_model == 'greenland_arthern':
            '''
            Note that this supposedly uses Arthern's density model
            https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/jgrf.20089
            Robert J. Arthern, Hugh F. J. Corr, Fabien Gillet-Chaulet, Robert L. Hawley, nd Elizabeth M. Morris
            '''
            density_surface = 280. # kg m^-3
            density_critical = 550. # kg m^-3
            self.density_deep = 917. # kg m^-3 
            scale_1 = 27. # m
            scale_2 = 42. # m
            z_critical = -1. * scale_1 * numpy.log((self.density_deep - density_surface) / (self.density_deep - density_critical)) # m

            density = (z <= 0.) * (z >= z_critical) \
                      * (self.density_deep - (self.density_deep - density_surface) * numpy.exp(z / scale_1)) \
                      + (z < z_critical) * (z >= -3000.) \
                      * (self.density_deep - (self.density_deep - density_critical) * numpy.exp((z - z_critical) / scale_2)) # kg m^-3
            return density / gnosim.utils.constants.mass_proton # convert from kg m^-3 to nucleons m^-3
        elif self.ice_model == 'greenland_avva':
            '''
            Note that this supposedly uses Arthern's density model
            https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/jgrf.20089
            Robert J. Arthern, Hugh F. J. Corr, Fabien Gillet-Chaulet, Robert L. Hawley, nd Elizabeth M. Morris
            '''
            density_surface = 280. # kg m^-3
            density_critical = 550. # kg m^-3
            self.density_deep = 917. # kg m^-3 
            scale_1 = 27. # m
            scale_2 = 42. # m
            z_critical = -1. * scale_1 * numpy.log((self.density_deep - density_surface) / (self.density_deep - density_critical)) # m

            density = (z <= 0.) * (z >= z_critical) \
                      * (self.density_deep - (self.density_deep - density_surface) * numpy.exp(z / scale_1)) \
                      + (z < z_critical) * (z >= -3000.) \
                      * (self.density_deep - (self.density_deep - density_critical) * numpy.exp((z - z_critical) / scale_2)) # kg m^-3
            return density / gnosim.utils.constants.mass_proton # convert from kg m^-3 to nucleons m^-3
        elif self.ice_model == 'antarctica_newcomb':
            return self.f_density(z) # nucleons m^-3
        elif self.ice_model == 'antarctica_allison':
            return self.f_density(z) # nucleons m^-3
        else:
            print ('WARNING Current icemodel does not match any density functions')
            return -999

    def indexOfRefraction(self,z):
        '''
        Returns the index of refraction for a particular depth z (m).  We use the Arcone formula for this calculation.
        I couldn't find the exact source fror which we got our values, but here is a paper that has similar:
        The in-situ dielectric constant of polar firn revisited - Austin Kovacs, Anthony J. Gow, Rexford M. Morey 
        https://doi.org/10.1016/0165-232X(94)00016-Q

        Parameters
        ----------
        z : numpy.ndarray of floats or float
            Depths to return index of refraction for.

        Returns
        -------
        index of refraction : float
            The index of refraction.
        '''

        if numpy.isscalar(z):
            scalar = True
            z = numpy.array([z])
        else:
            scalar = False

        if self.ice_model == 'parametric_greenland':
            n_infinity = 1.831 # Index of refraction deep in the ice
            n_0 = 1.29 # Index of refraction at surface
            n_air = 1.000293
            a = 10 # m

            value = numpy.zeros(z.shape)
            cut_1 = z <= 0.
            cut_2 = z > 0.
            value[cut_1] = (n_0 - (n_infinity * z[cut_1] / a)) / (1. - (z[cut_1] / a))
            value[cut_2] = n_air
        elif self.ice_model == 'empirical_greenland':
            n_air = 1.000293
            k = 0.86 * 1.e-3 # kg^-1 m^3                                                                                                                   
            
            value = numpy.zeros(z.shape)
            cut_1 = z <= 0.
            cut_2 = z > 0.
            value[cut_1] = 1. + (k * self.f_density(z[cut_1]) * gnosim.utils.constants.mass_proton)
            value[cut_2] = n_air
        elif self.ice_model == 'ross':
            n_air = 1.000293
            n_water = 1.33
            k = 0.86 * 1.e-3 # kg^-1 m^3

            value = numpy.zeros(z.shape)
            cut_1 = z < -500.
            cut_2 = numpy.logical_and(z >= -500., z <= 0.)
            cut_3 = z > 0.
            value[cut_1] = n_water
            value[cut_2] = 1. + (k * self.f_density(z[cut_2]) * gnosim.utils.constants.mass_proton)
            value[cut_3] = n_air
        elif self.ice_model == 'greenland_arthern':
            n_air = 1.000293
            n_water = 1.33
            k = 0.845 * 1.e-3 # kg^-1 m^3

            value = numpy.zeros(z.shape)
            cut_1 = z < -3000.
            cut_2 = numpy.logical_and(z >= -3000., z <= 0.)
            cut_3 = z > 0.
            value[cut_1] = n_water
            value[cut_2] = 1. + (k * self.density(z[cut_2]) * gnosim.utils.constants.mass_proton)
            value[cut_3] = n_air
        elif self.ice_model == 'greenland_avva':
            n_air = 1.000293
            n_water = 1.33
            k = 0.845 * 1.e-3 # kg^-1 m^3

            value = numpy.zeros(z.shape)
            cut_1 = z < -3000.
            cut_2 = numpy.logical_and(z >= -3000., z <= 0.)
            cut_3 = z > 0.
            value[cut_1] = n_water
            value[cut_2] = 1. + (k * self.density(z[cut_2]) * gnosim.utils.constants.mass_proton)
            value[cut_3] = n_air
        elif self.ice_model == 'antarctica_newcomb':
            n_air = 1.000293
            n_water = 1.33
            k = 0.845 * 1.e-3 # kg^-1 m^3

            value = numpy.zeros(z.shape)
            cut_1 = z < -3000.
            cut_2 = numpy.logical_and(z >= -3000., z <= 0.)
            cut_3 = z > 0.
            value[cut_1] = n_water
            value[cut_2] = 1.325 + 0.463 * (1.0-numpy.exp(0.0140*z[cut_2]))
            value[cut_3] = n_air
        elif self.ice_model == 'antarctica_allison':
            n_air = 1.000293
            n_water = 1.33
            k = 0.845 * 1.e-3 # kg^-1 m^3

            value = numpy.zeros(z.shape)
            cut_1 = z < -3000.
            cut_2 = numpy.logical_and(z >= -3000., z <= 0.)
            cut_3 = z > 0.
            value[cut_1] = n_water
            value[cut_2] = 1. + (k * self.density(z[cut_2]) * gnosim.utils.constants.mass_proton)
            value[cut_3] = n_air
        else:
            print ('WARNING Current icemodel does not match any index of refraction functions')
            return -999

        if scalar:
            return value[0]
        else:
            return value

    ############################################################

    def deltaIndexOfRefraction(self,z, delta_z=2.):
        '''
        Returns the change in index of refraction for a particular depth z (m) over a particular delta_z.

        Parameters
        ----------
        z : numpy.ndarray of floats or float
            Depths to return index of refraction for.
        delta_z : float, optional
            The delta_z for which to calculate the change in index of refraction. (Default is 2.0 m).

        Returns
        -------
        d_n : float
            The change in index of refraction.
        '''
        d_n = numpy.max([numpy.fabs(self.indexOfRefraction(z) - self.indexOfRefraction(z - delta_z)),
                          numpy.fabs(self.indexOfRefraction(z) - self.indexOfRefraction(z + delta_z))], axis=0) / delta_z
        return d_n

    ############################################################
    def temperature(self,z):
        '''
        Returns the temperature for a particular depth z (m) over a particular delta_z.

        Parameters
        ----------
        z : numpy.ndarray of floats or float
            Depths to return temperature for.

        Returns
        -------
        temps : numpy.ndarray of float
            The temperatures (C).
        '''
        temps = self.f_temperature(z) 
        return temps

    ############################################################
        
    def attenuationLengthBase(self,temperature, frequency):
        '''
        Returns attenuation length (m) in the ice at a particular frequency and temperature.
        Source: http://icecube.wisc.edu/~mnewcomb/radio/ -> https://icecube.wisc.edu/~araproject/radio/

        Parameters
        ----------
        temperature : numpy.ndarray of floats or float
            The temperature (C) for which to calculate the attenuation length.
        frequency : float
            The frequency (GHz) for which to calculate the attenuation length.

        Returns
        -------
        attenuation_length : numpy.ndarray of float
            The attenuation length (m).
        '''
        a = 5.03097 * numpy.exp(0.134806 * temperature)
        b = (0.172082 * temperature) + 10.629
        c = (-0.00199175 * temperature) - 0.703323
        #attenuation_length = 2. * 1.e3 * 1.701 / (a + (b * frequency**(c + 1.))) # WARNING: The factor 2 is a kludge! do we need it??  (DS: Likely related to power v.s. amplitude?  Site doesn't mention it...)
        attenuation_length = 1.e3 * 1.701 / (a + (b * frequency**(c + 1.)))
        return attenuation_length
    ########################################################################################################

    def attenuationLength(self,z, frequency):
        '''
        Returns attenuation length (m) in the ice at a particular frequency and depth.  i.e., distance at which electric field is reduced by e (m).
        Accounts for depths that are above ground, returning very large values.
        Source: ARA insturment paper Allison et al. 2011, arXiv:1105.2854
        
        Parameters
        ----------
        z : numpy.ndarray of floats or float
            The depth (m) for which to calculate the attenuation length.
        frequency : float
            The frequency (GHz) for which to calculate the attenuation length.

        Returns
        -------
        attenuation_length : numpy.ndarray of float
            The attenuation length (m).
        '''
        if self.ice_model in ['empirical_greenland']:
            attenuation_length = 1000. # m
        elif self.ice_model in ['ross']:
            attenuation_length = 400. # m
        elif self.ice_model in ['greenland_arthern']:
            attenuation_length = self.scale_greenland*self.attenuationLengthBase(self.temperature(z), frequency) # m
        elif self.ice_model in ['greenland_avva']:
            attenuation_length = self.f_att(z) -0.55*(frequency*1.e3 - 75.0)
            if numpy.isscalar(attenuation_length):
                if attenuation_length < 0:
                    attenuation_length = 0
            else:
                attenuation_length[attenuation_length < 0] = 0
        elif self.ice_model in ['antarctica_newcomb']:
            attenuation_length = self.scale_antarctica*self.attenuationLengthBase(self.temperature(z), frequency) # m
        elif self.ice_model in ['antarctica_allison']:
            '''
            This uses attenuation lengths taken from Figure 23 of: 
            Design and Initial Performance of the Askaryan Radio Array Prototype EeV Neutrino Detector at the South Pole - P Allison et al.
            https://arxiv.org/pdf/1105.2854.pdf

            I extrapolate it and fit with an order 7 polynomial which seems to do a really good job of capturing the features
            and extending the range to the full 0-3000m depths.  

            This is given for only 300 MHz, and cannot be reliably used for other frequencies.
            '''
            if frequency != 0.3:
                print('WARNING! The antarctica_allison model uses data for only 300MHz signals and may lose accuracy for other frequencies.')
                print('Given %f'%frequency)
                print('There is currently no attempt to shift the attenuations in the model to account for different frequencies.')
            attenuation_length = self.f_att_poly(z)
        else:
            attenuation_length = 1000. # m
            print ('WARNING! Auto setting attenuation length to %0.2f'%attenuation_length)

        if numpy.isscalar(z):
            return (z <= 0.) * attenuation_length + (z > 0.) * 1.e20 # m
        else:
            return (z <= 0.) * attenuation_length * numpy.ones(len(z)) + (z > 0.) * 1.e20 * numpy.ones(len(z)) # m

def tanToAtt(tan,frequency,n,delta_z):
    '''
    This aims to convert loss tangents from Bogorodsky's textbook to specific absorption, which seems to be
    the portion of signal attenuation caused by ice.  This is not used anywhere but I already coded it so
    am keeping it here.

    Parameters
    ----------
    tan : float
        Unitless loss tangent number given from Bogorodsky
    frequency : float
        Should be the corresponding frequency to the measurement of the loss tangent.  Given
        in GHz.
    n : float
        The index of refractions of the ice at the particular depth.
    delta_z : float
        The thickness of the ice section being discussed.  Given in m.

    Returns
    -------
    Na: float
        Attenuation (I think) in dB, Are these the units I want?

    '''
    c = gnosim.utils.constants.speed_light# = 0.299792 # m ns^-1

    Na = 8.68 * ( frequency / (2.0 * c / n) ) * delta_z**2 * tan

    return Na

    ############################################################

if __name__ == "__main__":
    pylab.close('all')
    z = numpy.linspace(-3000., 100., 10000) # Array of elevations (m)
    ice_antarctica_newcomb = Ice('antarctica_newcomb')
    ice_antarctica_allison = Ice('antarctica_allison')
    index_of_refraction_at_neutrino = ice_antarctica_newcomb.indexOfRefraction(-173.0)
    cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./index_of_refraction_at_neutrino))
    print(cherenkov_angle_deg)
    ice_ross = Ice('ross')
    ice_greenland_arthern = Ice('greenland_arthern')
    ice_greenland_avva = Ice('greenland_avva')
    ice_parametric = Ice('parametric_greenland')
    ice_empirical = Ice('empirical_greenland')


    # Density
    # -------

    pylab.figure()
    pylab.title('Density')
    pylab.plot(z, ice_greenland_avva.density(z) * gnosim.utils.constants.mass_proton, label='greenland_avva density from Arthern') # Convert from nucleons m^-3 to kg m^-3
    pylab.plot(z, ice_antarctica_allison.density(z) * gnosim.utils.constants.mass_proton, label='antarctica_allison density from Spicecore') # Convert from nucleons m^-3 to kg m^-3
    pylab.xlabel('Elevation (m)')
    pylab.ylabel(r'Density (kg m$^{-3}$)')
    pylab.legend(loc='lower center')
    pylab.xlim([-3100., 100.])
    pylab.legend()
    pylab.minorticks_on()
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    pylab.figure()
    pylab.title('Density')
    pylab.plot(-z, ice_greenland_avva.density(z) * gnosim.utils.constants.mass_proton / 1.e3, label='greenland_avva density from Arthern') # Convert from nucleons m^-3 to kg m^-3
    pylab.plot(-z, ice_antarctica_allison.density(z) * gnosim.utils.constants.mass_proton / 1.e3, label='antarctica_allison density from Spicecore') # Convert from nucleons m^-3 to kg m^-3
    pylab.xlabel('Depth (m)')
    pylab.ylabel(r'Density (Mg m$^{-3}$)')
    pylab.legend(loc='lower center')
    pylab.xlim([0,600])
    pylab.ylim([0.3,1.0])
    pylab.minorticks_on()
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


    # Index of Refraction
    # -------------------

    pylab.figure()
    pylab.title('Index of Refraction')
    pylab.plot(z, ice_greenland_avva.indexOfRefraction(z), label='greenland_avva')
    pylab.plot(z, ice_antarctica_allison.indexOfRefraction(z), label='antarctica_allison')
    pylab.xlim([-300,0.0])
    pylab.xlabel('Elevation (m)')
    pylab.ylabel('Index of Refraction')
    pylab.legend(loc='lower center')
    pylab.legend()
    pylab.minorticks_on()
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    pylab.figure()
    pylab.title('Index of Refraction')
    pylab.plot(z, ice_greenland_avva.indexOfRefraction(z), label='greenland_avva')
    pylab.plot(z, ice_antarctica_allison.indexOfRefraction(z), label='antarctica_allison')
    pylab.xlim([-3000,0.0])
    pylab.xlabel('Elevation (m)')
    pylab.ylabel('Index of Refraction')
    pylab.legend(loc='lower center')
    pylab.legend()
    pylab.minorticks_on()
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    
    
    # Attenuation
    # -----------

    pylab.figure()
    pylab.title('Attenuation')
    pylab.plot(ice_greenland_avva.attenuationLength(z, frequency=0.3),z, label='ice_greenland_avva Attenuation from Avva')
    pylab.plot(ice_antarctica_allison.attenuationLength(z, frequency=0.3),z, label='ice_antarctica_allison Attenuation from Allison')

    pylab.ylabel('Elevation (m)',fontsize=16)
    pylab.xlabel('Electric Field Attenuation Length (m)',fontsize=16)
    pylab.ylim([-3000., 0.])
    pylab.xlim([0., 3000.])
    pylab.legend(fontsize=16)
    pylab.minorticks_on()
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    
    pylab.figure()
    pylab.title('Attenuation')
    pylab.plot(ice_greenland_avva.attenuationLength(z, frequency=0.3),z, label='ice_greenland_avva  Attenuation from Avva - 300MHz')
    pylab.plot(ice_antarctica_allison.attenuationLength(z, frequency=0.3),z, label='ice_antarctica_allison  Attenuation from Allison - 300MHz')
    pylab.plot(ice_greenland_avva.attenuationLength(z, frequency=0.075),z,linestyle = '-.', label='ice_greenland_avva  Attenuation from Avva - 75MHz')

    pylab.ylabel('Elevation (m)')
    pylab.xlabel('Electric Field Attenuation Length (m)')
    pylab.ylim([-3000., 0.])
    pylab.xlim([400., 2000.])
    pylab.legend()
    pylab.minorticks_on()
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    pylab.figure()
    pylab.title('Attenuation')
    pylab.plot(ice_greenland_avva.attenuationLength(z, frequency=0.3),z, label='ice_greenland_avva  Attenuation from Avva - 300MHz')
    pylab.plot(ice_antarctica_allison.attenuationLength(z, frequency=0.3),z, label='ice_antarctica_allison  Attenuation from Allison - 300MHz')
    pylab.plot(ice_greenland_avva.attenuationLength(z, frequency=0.075),z,linestyle = '-.', label='ice_greenland_avva  Attenuation from Avva - 75MHz')

    pylab.ylabel('Elevation (m)')
    pylab.xlabel('Electric Field Attenuation Length (m)')
    pylab.ylim([-3000., 0.])
    pylab.xlim([0., 2100.])
    pylab.legend()
    pylab.minorticks_on()
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


    # Temperature
    # -----------

    pylab.figure()
    pylab.title('Temperature')
    pylab.plot(ice_greenland_avva.f_temperature(z),z,label='ice_greenland_avva Temperature from Johnsen')
    pylab.plot(ice_antarctica_allison.f_temperature(z),z,label='ice_antarctica_allison Temperature from Newcomb')

    pylab.ylabel('Elevation (m)')
    pylab.xlabel('Temperature (C)')
    pylab.ylim([-3000., 0.])
    #pylab.xlim([0., 3000.])
    pylab.legend()
    pylab.minorticks_on()
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    pylab.figure()
    pylab.title('Temperature')
    pylab.plot(-z,ice_greenland_avva.f_temperature(z),label='greenland avva Temperature from Johnsen')
    pylab.plot(-z,ice_antarctica_allison.f_temperature(z),label='antarctica allison Temperature from Newcomb')

    pylab.xlabel('Elevation (m)')

    pylab.ylabel('Temperature (C)')
    pylab.xlim([0.,2500.])
    pylab.ylim([-55.,-15.])
    pylab.legend()
    pylab.minorticks_on()
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
