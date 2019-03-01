#!/usr/bin/env python
'''
This file contains the properties of any ice model (Greenland v.s. Antarctic ice)
As well as functions for using these models.

This combines what were previously rhe antarctic.py and greenland.py files.
'''
import sys
sys.path.append("/home/dsouthall/Projects/GNOSim/")
import numpy
import pylab
import scipy.interpolate
import gnosim.utils.constants

pylab.ion()

############################################################

def checkIceModel(ice_model):
    '''
    Checks if the current ice_model is supported.
    Returns the ice model if it is, otherwise it will
    either replace the ice_model give with it's current
    sudonym (i.e. arthern -> greenland), or will return
    the default value if no sudonym found. 
    '''
    possible_models = ['antarctica', 'greenland','parametric_greenland','empirical_greenland','ross','polar','arthern']
    ice_model_default = possible_models[0]
    if ~numpy.isin(ice_model, possible_models):
        print("Requested ice model %s not in available ice model list, defaulting to %s"%(ice_model,ice_model_default))
        ice_model = ice_model_default
    elif ice_model == 'polar':
        ice_model = 'antarctica' #For some backwards compatibility
    elif ice_model == 'arthern':
        ice_model = 'greenland'  #For some backwards compatibility
    return ice_model

class Ice:
    def __init__(self, ice_model, suppress_fun = False):
        self.ice_model = checkIceModel(ice_model)

        images =    {
                    'antarctica':"\n                ++++++++++              \n                ++++++++++++++          \n               ++++++++++++++++++++     \n              +++++++ANTARCTICA++++     \n +            ++++++++++++++++++++++    \n ++++++        +++++++++++++++++++++    \n   ++ ++     + +++++++++++++++++++      \n    + +++   + +++++++ANTARCTICA+++++    \n     +++ +  ++++++++++++++++++++++++++  \n     ++++++++++++++++++++++++++++++++++ \n     ++++++++++++++++++++++++++++++++++ \n    +++++++++++++++++ANTARCTICA++++++++ \n       ++++++++++ +++++++++++++++++++++ \n       ++++++++++   ++++++++++++++++++  \n       +++++++++      ++++++++++++++++  \n          +++++ +     +++++++++++++++   \n                      ++++++++++++++    \n                       ++++++++++++     \n                       +++++++++++      \n                      ++++++++++",
                    'ross':"\n                ++++++++++              \n                ++++++++++++++          \n               ++++++++++++++++++++     \n              +++++++ANTARCTICA++++     \n +            ++++++++++++++++++++++    \n ++++++        +++++++++++++++++++++    \n   ++ ++     + +++++++++++++++++++      \n    + +++   + +++++++ANTARCTICA+++++    \n     +++ +  ++++++++++++++++++++++++++  \n     ++++++++++++++++++++++++++++++++++ \n     ++++++++++++++++++++++++++++++++++ \n    +++++++++++++++++ANTARCTICA++++++++ \n       ++++++++++ +++++++++++++++++++++ \n       ++++++++++   ++++++++++++++++++  \n       +++++++++      ++++++++++++++++  \n          +++++ +     +++++++++++++++   \n                      ++++++++++++++    \n                       ++++++++++++     \n                       +++++++++++      \n                      ++++++++++",
                    'greenland':"                          +++++         \n                       +++ ++++++       \n           + +++  +  +++++++++++++      \n         + ++++ ++++ ++++++++  +        \n       +++++++++++++++++++++++  ++  +   \n       +++++++++++++++++++++++++ ++++++ \n      ++++++++++++++++++++++++ ++++++   \n ++++++++++++++GREENLAND++++++++++      \n  ++++++++++++++++++++++++++++++++      \n ++ ++++++++++++++++++++++++++++++      \n  +++++++++++++++++++++++++++++++       \n + + ++++++++++GREENLAND+++++++++       \n       +++++++++++++++++++++++++++      \n       ++++++++++++++++++++++++         \n        +++++++++++++++++++++++++       \n        +++++++GREENLAND++++++++        \n        ++++++++++++++++++++++          \n       ++++++++++++++++++++             \n       + ++++++++++++++++++             \n       + ++++++GREENLAND+++++           \n      +I++++++++++++++++++  ++          \n        ++++++++++++++++++++++          \n        +++++++++++++++++++             \n       +++++++++++++++                  \n     + +++++++++++++                    \n      +++++++++++++                     \n       ++++++++                         \n      ++++++++                          \n       ++++++                           \n       +++++                            \n         +++                            \n                                        ",
                    'parametric_greenland':"                          +++++         \n                       +++ ++++++       \n           + +++  +  +++++++++++++      \n         + ++++ ++++ ++++++++  +        \n       +++++++++++++++++++++++  ++  +   \n       +++++++++++++++++++++++++ ++++++ \n      ++++++++++++++++++++++++ ++++++   \n ++++++++++++++GREENLAND++++++++++      \n  ++++++++++++++++++++++++++++++++      \n ++ ++++++++++++++++++++++++++++++      \n  +++++++++++++++++++++++++++++++       \n + + ++++++++++GREENLAND+++++++++       \n       +++++++++++++++++++++++++++      \n       ++++++++++++++++++++++++         \n        +++++++++++++++++++++++++       \n        +++++++GREENLAND++++++++        \n        ++++++++++++++++++++++          \n       ++++++++++++++++++++             \n       + ++++++++++++++++++             \n       + ++++++GREENLAND+++++           \n      +I++++++++++++++++++  ++          \n        ++++++++++++++++++++++          \n        +++++++++++++++++++             \n       +++++++++++++++                  \n     + +++++++++++++                    \n      +++++++++++++                     \n       ++++++++                         \n      ++++++++                          \n       ++++++                           \n       +++++                            \n         +++                            \n                                        ",
                    'empirical_greenland':"                          +++++         \n                       +++ ++++++       \n           + +++  +  +++++++++++++      \n         + ++++ ++++ ++++++++  +        \n       +++++++++++++++++++++++  ++  +   \n       +++++++++++++++++++++++++ ++++++ \n      ++++++++++++++++++++++++ ++++++   \n ++++++++++++++GREENLAND++++++++++      \n  ++++++++++++++++++++++++++++++++      \n ++ ++++++++++++++++++++++++++++++      \n  +++++++++++++++++++++++++++++++       \n + + ++++++++++GREENLAND+++++++++       \n       +++++++++++++++++++++++++++      \n       ++++++++++++++++++++++++         \n        +++++++++++++++++++++++++       \n        +++++++GREENLAND++++++++        \n        ++++++++++++++++++++++          \n       ++++++++++++++++++++             \n       + ++++++++++++++++++             \n       + ++++++GREENLAND+++++           \n      +I++++++++++++++++++  ++          \n        ++++++++++++++++++++++          \n        +++++++++++++++++++             \n       +++++++++++++++                  \n     + +++++++++++++                    \n      +++++++++++++                     \n       ++++++++                         \n      ++++++++                          \n       ++++++                           \n       +++++                            \n         +++                            \n                                        "
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
        if numpy.isin(self.ice_model, ['antarctica','ross']):
            self.ice_thickness = 3000. #m
        elif numpy.isin(self.ice_model, ['greenland','parametric_greenland','empirical_greenland']):
            self.ice_thickness = 3000. #m
        else:
            self.ice_thickness = 3000. #m
        ###
        #Defining density profiles
        ###

        if numpy.isin(self.ice_model, ['antarctica','ross']):
            # From South Pole Ice Characteristics, http://icecube.wisc.edu/~mnewcomb/radio/#icerefraction
            # Density (Mg m^-3)
            # Elevation, negative is down (m)
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

        if numpy.isin(self.ice_model, ['antarctica','ross']):
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
        elif numpy.isin(self.ice_model, ['greenland','parametric_greenland','empirical_greenland']):
            # Johnsen et al. 1995
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

    def density(self,z):
        """
        z = elevation (m)
        
        Returns:
        density (nuclei m^-3)
        Source: http://www.iceandclimate.nbi.ku.dk/research/flowofice/densification/
        """
        
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
        elif self.ice_model == 'greenland':
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
        elif self.ice_model == 'antarctica':
            return self.f_density(z) # nucleons m^-3
        else:
            print ('WARNING Current icemodel does not match any density functions')
            return -999

    def indexOfRefraction(self,z):
        """
        z = elevation (m)

        Returns:
        index of refraction

       
        """
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
        elif self.ice_model == 'greenland':
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
        elif self.ice_model == 'antarctica':
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
        else:
            print ('WARNING Current icemodel does not match any index of refraction functions')
            return -999

        if scalar:
            return value[0]
        else:
            return value

    ############################################################

    def deltaIndexOfRefraction(self,z, delta_z=2.):
        """
        Change in index of refraction
        """
        return numpy.max([numpy.fabs(self.indexOfRefraction(z) - self.indexOfRefraction(z - delta_z)),
                          numpy.fabs(self.indexOfRefraction(z) - self.indexOfRefraction(z + delta_z))], axis=0) / delta_z

    ############################################################
    def temperature(self,z):
        """
        z = elevation (m)

        Returns temperature (C)
        """
        return self.f_temperature(z) 

    ############################################################
        
    def attenuationLengthBase(self,temperature, frequency):
        """
        http://icecube.wisc.edu/~mnewcomb/radio/
        
        temperature = Temperature (C)
        frequency = Frequency (GHz)

        Returns attenuation length (m)
        """
        a = 5.03097 * numpy.exp(0.134806 * temperature)
        b = (0.172082 * temperature) + 10.629
        c = (-0.00199175 * temperature) - 0.703323
        return 2. * 1.e3 * 1.701 / (a + (b * frequency**(c + 1.))) # WARNING: The factor 2 is a kludge! do we need it??
    ########################################################################################################

    def attenuationLength(self,z, frequency):
        """
        z = elevation (m)
        frequency = radio frequency (GHz)

        Returns:
        attenuation length, i.e., distance at which electric field is reduced by e (m)

        Source: ARA insturment paper Allison et al. 2011, arXiv:1105.2854

        Question: What is radio attenuation lenth in air??
        """
        # Treat as constant with respect to depth and frequency until add real data points
        frequency_0 = 0.3 # GHz

        if self.ice_model in ['empirical_greenland']:
            attenuation_length = 1000. # m
        elif self.ice_model in ['ross']:
            attenuation_length = 400. # m
        elif self.ice_model in ['greenland']:
            attenuation_length = self.attenuationLengthBase(self.f_temperature(z), frequency) # m
        elif self.ice_model in ['antarctica']:
            attenuation_length = self.attenuationLengthBase(self.f_temperature(z), frequency) # m
        else:
            attenuation_length = 1000. # m
            print ('WARNING, auto setting attenuation length to %0.2f'%attenuation_length)

        if numpy.isscalar(z):
            return (z <= 0.) * attenuation_length + (z > 0.) * 1.e20 # m
        else:
            return (z <= 0.) * attenuation_length * numpy.ones(len(z)) + (z > 0.) * 1.e20 * numpy.ones(len(z)) # m

    ############################################################

if __name__ == "__main__":
    z = numpy.linspace(-3000., 100., 10000) # Array of elevations (m)
    ice_antarctica = Ice('antarctica')
    ice_ross = Ice('ross')
    ice_greenland = Ice('greenland')
    ice_parametric = Ice('parametric_greenland')
    ice_empirical = Ice('empirical_greenland')

    pylab.figure()
    pylab.plot(z, ice_parametric.density(z) * gnosim.utils.constants.mass_proton, label='Parametric') # Convert from nucleons m^-3 to kg m^-3
    pylab.plot(z, ice_empirical.density(z) * gnosim.utils.constants.mass_proton, label='Empirical') # Convert from nucleons m^-3 to kg m^-3
    pylab.plot(z, ice_ross.density(z) * gnosim.utils.constants.mass_proton, label='Ross') # Convert from nucleons m^-3 to kg m^-3
    pylab.plot(z, ice_greenland.density(z) * gnosim.utils.constants.mass_proton, label='greenland') # Convert from nucleons m^-3 to kg m^-3
    pylab.plot(z, ice_antarctica.density(z) * gnosim.utils.constants.mass_proton, label='antarctica') # Convert from nucleons m^-3 to kg m^-3
    pylab.xlabel('Elevation (m)')
    pylab.ylabel(r'Density (kg m$^{-3}$)')
    pylab.legend(loc='lower center')
    pylab.scatter(ice_empirical.elevation_firn[1:-1], ice_empirical.density_firn[1:-1] * gnosim.utils.constants.mass_proton, c='red')
    pylab.scatter(ice_empirical.elevation_deep[1:-1], ice_empirical.density_deep[1:-1] * gnosim.utils.constants.mass_proton, c='black')
    #pylab.xlim([-600., 100.])
    pylab.xlim([-3100., 100.])

    z = numpy.linspace(-3100., 100., 2000) # Array of elevations (m)

    pylab.figure()
    #pylab.plot(z, ice_parametric.indexOfRefraction(z), label='Parametric')
    pylab.plot(z, ice_empirical.indexOfRefraction(z), label='Empirical')
    #pylab.plot(z, ice_ross.indexOfRefraction(z), label='Ross')
    pylab.plot(z, ice_antarctica.indexOfRefraction(z), label='antarctica')

    pylab.xlabel('Elevation (m)')
    pylab.ylabel('Index of Refraction')
    pylab.legend(loc='lower center')
    #k = 0.86 * 1.e-3 # kg^-1 m^3
    #pylab.plot(z, 1. + (k * ice_antarctica.f_density(z) * gnosim.utils.constants.mass_proton), c='green')
    pylab.xlim([-3100., 100.])
    
    z = numpy.linspace(-3000., 0., 2000) # Array of elevations (m)
    pylab.figure()
    frequency = 0.3 # GHz
    pylab.plot(z, ice_antarctica.attenuationLength(z, frequency=frequency))
    pylab.title('Frequency = %i MHz'%(frequency / gnosim.utils.constants.MHz_to_GHz))
    pylab.xlabel('Elevation (m)')
    pylab.ylabel('Electric Field Attenuation Length (m)')
    pylab.xlim([-3000., 0.])
    
    #input("press any key to exit")
############################################################

