import numpy
import h5py

import gnosim.utils.constants
import gnosim.utils.bayesian_efficiency
import gnosim.earth.earth
import gnosim.earth.greenland

############################################################

def acceptance(infile, cos_theta_bins=None, electric_field_threshold=1.e-4, earth=True, mode_reflections='all'):
    """
    Return volumetric acceptance (km^3 sr) and acceptance (m^2 sr)
    """
    reader = h5py.File(infile, 'r')

    energy_neutrino = reader['energy_neutrino'][0] # GeV

    interaction_length, interaction_length_anti \
        = gnosim.earth.earth.interactionLength(gnosim.utils.constants.density_water * gnosim.utils.constants.mass_proton,
                                               energy_neutrino)
    interaction_length = numpy.sqrt(interaction_length * interaction_length_anti) # m

    if cos_theta_bins is not None:
        n_bins = len(cos_theta_bins) - 1
    else:
        cos_theta_bins = numpy.array([-1., 1.])
        n_bins = 1

    volumetric_acceptance = numpy.zeros(n_bins)
    volumetric_acceptance_error_low = numpy.zeros(n_bins)
    volumetric_acceptance_error_high = numpy.zeros(n_bins)
    acceptance = numpy.zeros(n_bins)
    acceptance_error_low = numpy.zeros(n_bins)
    acceptance_error_high = numpy.zeros(n_bins)
    n_volume = numpy.zeros(n_bins)

    ice_model = reader.attrs['ice_model']

    for ii in range(0, n_bins):

        theta_min = numpy.degrees(numpy.arccos(cos_theta_bins[ii]))
        theta_max = numpy.degrees(numpy.arccos(cos_theta_bins[ii + 1]))

        if theta_min > theta_max:
            theta_temp = theta_max
            theta_max = theta_min
            theta_min = theta_temp

        print ('  Theta %.3f -- %.3f deg'%(theta_min, theta_max))

        n_total = numpy.sum(numpy.all([reader['theta_0'][...] >= theta_min,
                                       reader['theta_0'][...] <= theta_max],
                                      axis=0))

        if mode_reflections == 'all':
            cut_mode_reflections = numpy.ones(len(reader['solution'][...]))
            n_pass = int(numpy.sum(reader['p_earth'][...] \
                                   * (reader['theta_0'][...] >= theta_min) \
                                   * (reader['theta_0'][...] <= theta_max) \
                                   * (reader['electric_field'][...] > electric_field_threshold)))
            print ('all')
            
        elif mode_reflections == 'direct':
            cut_mode_reflections = numpy.logical_and(reader['solution'][...] >= 0, reader['solution'][...] <= 2)
            n_pass = int(numpy.sum(reader['p_earth'][...] \
                                   * (reader['solution'][...] >= 0) \
                                   * (reader['solution'][...] <= 2) \
                                   * (reader['theta_0'][...] >= theta_min) \
                                   * (reader['theta_0'][...] <= theta_max) \
                                   * (reader['electric_field'][...] > electric_field_threshold)))
            print ('direct')
        elif mode_reflections == 'reflect':
            cut_mode_reflections = numpy.logical_and(reader['solution'][...] >= 3, reader['solution'][...] <= 5)
            n_pass = int(numpy.sum(reader['p_earth'][...] \
                                   * (reader['solution'][...] >= 3) \
                                   * (reader['solution'][...] <= 5) \
                                   * (reader['theta_0'][...] >= theta_min) \
                                   * (reader['theta_0'][...] <= theta_max) \
                                   * (reader['electric_field'][...] > electric_field_threshold)))
            print ('reflect')
        else:
            print ('WARNING: mode reflections %s not recognized'%(mode_reflections))

        print (n_total, n_pass)

        """
        #if allow_reflections:
        if True:
            n_pass = numpy.sum(numpy.all([reader['theta_0'][...] >= theta_min,
                                          reader['theta_0'][...] <= theta_max,
                                          reader['electric_field'][...] > electric_field_threshold],
                                         axis=0))
            print (n_pass)
        #else:
        if False:
            #print 'NO REFLECTIONS'
            n_pass = numpy.sum(numpy.all([reader['theta_0'][...] >= theta_min,
                                          reader['theta_0'][...] <= theta_max,
                                          reader['electric_field'][...] > electric_field_threshold,
                                          reader['solution'][...] >= 0,
                                          reader['solution'][...] <= 2],
                                         axis=0))
            print (n_pass)
        """

        #print (n_total, n_pass, float(n_pass) / n_total, reader.attrs['geometric_factor'])

        #n_volume = numpy.sum(reader['p_earth'][...] \
        #                     * reader['p_detect'][...] \
        #                     * (reader['theta_0'][...] >= theta_min) \
        #                     * (reader['theta_0'][...] <= theta_max))

        try:
            efficiency, (efficiency_low, efficiency_high) = gnosim.utils.bayesian_efficiency.confidenceInterval(n_total, n_pass)
        except:
            efficiency, efficiency_low, efficiency_high = float(n_pass) / n_total, 0., 0.

        if earth:
            # Including effect of Earth attenuation

            print ('test', len(reader['p_earth'][...]))
            print ('test', len(cut_mode_reflections))

            volumetric_acceptance[ii] = numpy.sum(reader['p_earth'][...] \
                                                  * cut_mode_reflections \
                                                  * (reader['theta_0'][...] >= theta_min) \
                                                  * (reader['theta_0'][...] <= theta_max) \
                                                  * (gnosim.earth.greenland.density(reader['z_0'][...], ice_model=ice_model) / gnosim.utils.constants.density_water) \
                                                  * (reader['electric_field'][...] > electric_field_threshold)) \
                * (reader.attrs['geometric_factor'] / (n_total * n_bins)) * gnosim.utils.constants.km_to_m**-3 # km^3 sr
        else:
            # Ignore Effect of Earth attenuation
            volumetric_acceptance[ii] = numpy.sum(cut_mode_reflections \
                                                  * (reader['theta_0'][...] >= theta_min) \
                                                  * (reader['theta_0'][...] <= theta_max) \
                                                  * (gnosim.earth.greenland.density(reader['z_0'][...], ice_model=ice_model) / gnosim.utils.constants.density_water) \
                                                  * (reader['electric_field'][...] > electric_field_threshold)) \
                    * (reader.attrs['geometric_factor'] / (n_total * n_bins)) * gnosim.utils.constants.km_to_m**-3 # km^3 sr

        if n_pass > 0:
            volumetric_acceptance_error_low[ii] = volumetric_acceptance[ii] * ((efficiency - efficiency_low) / efficiency)
            volumetric_acceptance_error_high[ii] = volumetric_acceptance[ii] * ((efficiency_high - efficiency) / efficiency)
        else:
            volumetric_acceptance_error_low[ii] = 0.
            volumetric_acceptance_error_high[ii] = 0.

        acceptance[ii] = volumetric_acceptance[ii] * gnosim.utils.constants.km_to_m**3 / interaction_length # m^2 sr 
        acceptance_error_low[ii] = volumetric_acceptance_error_low[ii] * gnosim.utils.constants.km_to_m**3 / interaction_length
        acceptance_error_high[ii] = volumetric_acceptance_error_high[ii] * gnosim.utils.constants.km_to_m**3 / interaction_length

    reader.close()
    
    if n_bins > 1:
        return volumetric_acceptance, volumetric_acceptance_error_low, volumetric_acceptance_error_high, \
            acceptance, acceptance_error_low, acceptance_error_high
    else:
        return volumetric_acceptance[0], volumetric_acceptance_error_low[0], volumetric_acceptance_error_high[0], \
            acceptance[0], acceptance_error_low[0], acceptance_error_high[0]

############################################################

def acceptanceToVolumetricAcceptance(acceptance, energy_neutrino):
    """
    Given the acceptance (m^2 sr) and neutrino energy (GeV), return the water-equivalent volumetric acceptance (km^3 sr)
    """
    interaction_length, interaction_length_anti \
        = gnosim.earth.earth.interactionLength(gnosim.utils.constants.density_water * gnosim.utils.constants.mass_proton,
                                               energy_neutrino)
    interaction_length = numpy.sqrt(interaction_length * interaction_length_anti) # m
    volumetric_acceptance = acceptance * gnosim.utils.constants.km_to_m**(-3) * interaction_length # km^3 sr 
    return volumetric_acceptance

############################################################

def volumetricAcceptanceToAcceptance(volumetric_acceptance, energy_neutrino):
    """
    Given the water-equivalent volumetric acceptance (km^3 sr) and neutrino energy (GeV), return the acceptance (m^2 sr)
    """
    interaction_length, interaction_length_anti \
        = gnosim.earth.earth.interactionLength(gnosim.utils.constants.density_water * gnosim.utils.constants.mass_proton,
                                               energy_neutrino)
    interaction_length = numpy.sqrt(interaction_length * interaction_length_anti) # m
    acceptance = volumetric_acceptance * gnosim.utils.constants.km_to_m**3 / interaction_length # m^2 sr
    return acceptance

############################################################
