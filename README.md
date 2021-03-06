# Greenland Neutrino Observatory Simulator (GNOSim) 

## People 
### Past Contributors:

- Keith Bechtol
bechtol@kicp.uchicago.edu

- Shoshana Chipman 
chipmansb@uchicago.edu

 - Dan Smith
danielsmith@uchicago.edu

### Current Contributor:

 - Dan Southall (contact if there are problems)
dsouthall@uchicago.edu




# OVERVIEW

GNOSim is a MC neutrino simulation designed to support time-domain Askaryan signals, allowing for phased-array triggering and more realistic signal waveforms.  The simulation currently has several in-built ice models for easy simulations in either Antarctica or Greenland. The below section discusses the usage of the simulation code:

# Table of Contents

0.0.0 ............. [Preface and Quick Guide](000-preface--quick-guide)

1.0.0 ............. [General Preparations](#100-general-preparations)

&nbsp;&nbsp;&nbsp;&nbsp;1.1.0 ......... [Usage Information](#110-usage-information)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.1 ..... [Dependencies](#111-dependencies)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.2 ..... [Running Scripts](#112-running-scripts)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.3 ..... [Paths](#113-paths)

&nbsp;&nbsp;&nbsp;&nbsp;1.2.0 ......... [Creating a Ray Tracing Library for the Simulation](#120-creating-a-ray-tracing-library-for-the-simulation)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.2.1 ..... [What is a Ray Tracing Library?](#121-what-is-a-ray-tracing-library)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.2.2 ..... [Creating a Ray Tracing Library](#122-creating-a-ray-tracing-library)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.2.3 ..... [Concerning the Envelope and Concave Hull](#123-concerning-the-envelope-and-concave-hull)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.2.4 ..... [Making Many Libraries with Farm Script](#124-making-many-libraries-with-farm-script)

&nbsp;&nbsp;&nbsp;&nbsp;1.3.0 ......... [Preparing a Station Configuration File](#130-preparing-a-station-configuration-file)

&nbsp;&nbsp;&nbsp;&nbsp;1.4.0 ......... [Preparing a Simulation Configuration File](#140-preparing-a-simulation-configuration-file)

2.0.0 ............. [Running the Simulation](#200-running-the-simulation)

&nbsp;&nbsp;&nbsp;&nbsp;2.1.0 ......... [Running a Single Instance of GNOSim](#210-running-a-single-instance-of-gnosim)

&nbsp;&nbsp;&nbsp;&nbsp;2.2.0 ......... [Memory Allocation for GNOSim and Running on Midway](#220-memory-allocation-for-gnosim-and-running-on-midway)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2.1 ..... [Running with Batch](#221-running-with-batch)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2.2 ..... [Running in an Interactive Terminal](#222-running-in-an-interactive-terminal)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2.3 ..... [More Info](#223-more-info)

&nbsp;&nbsp;&nbsp;&nbsp;2.3.0 ......... [Running Many Simulations with Farm Script](2.3.0-running-many-simulations-with-farm-script)

&nbsp;&nbsp;&nbsp;&nbsp;2.4.0 ......... [Stitching Together Simulations](#240-stitching-together-simulations)

3.0.0 ............. [Working with Simulation Data](#300-working-with-simulation-data)

&nbsp;&nbsp;&nbsp;&nbsp;3.1.0 ......... [Information About h5py](3.1.0#-information-about-h5py)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.1 ..... [Saving and Loading h5py Files](#311-saving-and-loading-h5py-files)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.2 ..... [Attributes and Datasets in h5py Files](#312-attributes-and-datasets-in-h5py-files)

&nbsp;&nbsp;&nbsp;&nbsp;3.2.0 ......... [Formatting of Output Files](#320-formatting-of-output-files)

&nbsp;&nbsp;&nbsp;&nbsp;3.3.0 ......... [Working with info_dtype Objects for Meta-data](#330-working-with-info_dtype-objects-for-meta-data)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3.1 ..... [Info Content Descriptions](#331-info-content-descriptions)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3.2 ..... [Working with the Info Dataset](#332-working-with-the-info-dataset)

&nbsp;&nbsp;&nbsp;&nbsp;3.4.0 ......... [Analysis Scripts](#340-analysis-scripts)

&nbsp;&nbsp;&nbsp;&nbsp;3.5.0 ......... [Utils Scripts](#350-utils-scripts)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.5.1 ..... [Generate Event Orientations](#351-generate-event-orientations)

&nbsp;&nbsp;&nbsp;&nbsp;3.6.0 ......... [Converting Signals to Data-Like Format](#360-converting-signals-to-data-like-format)

4.0.0 ............. [Information For Contributors](#400-information-for-contributors)


&nbsp;&nbsp;&nbsp;&nbsp;4.1.0 ......... [Code Repository](#410-code-repository)

&nbsp;&nbsp;&nbsp;&nbsp;4.2.0 ......... [Conventions](#420-conventions)

&nbsp;&nbsp;&nbsp;&nbsp;4.3.0 ......... [TODOs](#430-todos)



# 0.0.0 Preface / Quick Guide
------------------------------
  
  This README has been written to ideally be quite detailed.  This can be extremely helpful when questions arise, but off putting if you just want to quickly try something and ignore the details.  If you wish to know the minimum at first and come back as questions arise, then the sections outlined below are a quick and easy way to start.  Note that if you change things in configuration files you should consult the README to see if there are special considerations for those parameters.  Otherwise for this Quick Guide it is assumed you are running with essentially default parameters.

  Quick Guide
  -----------
  - Read [Section 1.1.1](#111-dependencies) to ensure you have the required software and knowledge of use.
  - Follow [Section 1.2.2](#122-creating-a-ray-tracing-library) to create the necessary ray tracing look-up libraries.
  - Reference [Section 1.3.0](#130-preparing-a-station-configuration-file) and [Section 1.4.0](#140-preparing-a-simulation-configuration-file) for preparing the antenna locations, ice models, and all other simulation parameters.
  - Read the appropriate subsections of [Section 2.0.0](#200-running-the-simulation) depending on how many simulations you want to run on your first attempt.  If you intend on running many jobs (thus potentially using a lot of memory) it is recommended that you read [Sections 2.2.0](#220-memory-allocation-for-gnosim-and-running-on-midway), otherwise for a relatively low event count (~10000 - ~100000) you can run with just the info provided in [Section 2.1.0](#210-running-a-single-instance-of-gnosim).
  - At this point you should have run the simulation and gotten output data.  To learn how to work with this data reference the appropriate portions of [Section 3.0.0](#300-working-with-simulation-data).
  - [Section 3.4.0](#340-analysis-scripts) discusses previously developed analysis scripts that you should in theory be able to run on your data to start working with it.  The code contained in these also shows how the data can be worked with so you can form your own analysis scripts as necessary.
  -   If in early testing you see no events triggering, double check that you are running at reasonable energies.  Energies less than 1e7 GeV will have every small trigger rates.  

# 1.0.0 General Preparations

This section discusses the preparations required to run the simulation.

## 1.1.0 Usage Information

### 1.1.1 Dependencies

The code was developed using compatible builds of Python 3.7.1 and ROOT 6.16.00.  A module has been built on Midway2 specifically for this project and can be loaded using the command:

    . gnosim/loadmodules

This will unload the current ROOT and Python modules and load the recommended versions.

Python packages:

numpy - http://www.numpy.org/

scipy - http://www.scipy.org/

h5py - http://www.h5py.org/

matplotlib / pylab - http://matplotlib.org/

Certain portions of the code require large amounts of memory.  If the code is breaking this may be something to check.  See [Section 2.2.0](#220-memory-allocation-for-gnosim-and-running-on-midway) for more on memory concerns.

### 1.1.2 Running Scripts

Many of the modules can be run as independent files for testing purposes, where commands listed below: 

    if __name__ == '__main__':

will be executed.  To run these interactively, open a python terminal from the folder containing gnosim (such that paths like import gnosim.sim work) and enter the command:

    exec(open('./gnosim/MODULE_PATH/MODULE_NAME.py').read())

Or in an ipython terminal you can use the magic command %run:

    %run ./gnosim/MODULE_PATH/MODULE_NAME.py

# 1.1.3 Paths

In order for GNOSim to reliably find files as intended you must set and export the following system variable in your .bashrc file:

    GNOSIM_DIR="/PATH_TO_GNOSIM"
    export GNOSIM_DIR

Where PATH_TO_GNOSIM is the path that the folder where the git directory is located.  I.e the folder containing gnosim (which should be one directory up from the location of this README).

It may also be useful to add a path to the root location of where data is stored (for instance if you store a lot of data on scratch).

    GNOSIM_DATA="/PATH_TO_GNOSIM_DATA"
    export GNOSIM_DATA

Paths may be used that refer to an environment variable such as:

    image_path = '${GNOSIM_DATA}/figures'

However when this is done whatever us being used to evaluate/interpret this path should use:

    path = os.path.expandvars(path)

Ideally this is done in the code where, but if it is missed and you get an error this may be the problem.

## 1.2.0 Creating a Ray Tracing Library for the Simulation

### 1.2.1 What is a Ray Tracing Library?

GNOSim calculates many of the values per neutrino event by interpolating the values from previously generated ray tracing libraries.  These libraries contain information such as the the paths of rays connecting the antenna to the rest of the detector volume, the time it takes for a ray to travel, distance traveled, attenuation information, reflection information, etc.  They are created in advance once per antenna depth.  The directory containing a particular antenna depth is then listed under that antenna's library directory option in the station configuration file (see [Section 1.3.0](#130-preparing-a-station-configuration-file)).

### 1.2.2 Creating a Ray Tracing Library

#### PREPARATION

Open the gnosim/trace/refraction_library.py file in a text editor, and scroll to:

    if __name__ == '__main__':

The code written below this will generate the libraries for you and split them up by solution type (for quicker loading later).  To create the ray tracing libraries you should adjust the parameters listed here as you wish.  Below are example settings and descriptions:

    make_library = True     #If True, will compute the libraries and save them.  Otherwise just loads from previously saved libraries if available (False is useful for plotting previously generated libraries).
    split_library = True    #If True, will split the libraries by solution type and save them.
    plot_library = False    #If True, will plot the ray tracing libraries as they are created (or loaded if make_library == False).
    save_envelope = True    #If True, will calculate the envelope for the ray tracing library and save it.  Advisable to do in advance when ray tracing library is created.
    plot_envelope = False   #If True, will plot the envelope.
    z_array = numpy.array([-173.0,-174.0,-175.0,-176.0,-177.0,-179.0,-181.0]) #The list of depths for which to throw rays (or load libraries if make_library == False).
    n_rays = 180            #The number of rays to be thrown per depth.
    r_limit = None          #Note if this is NOT None, then all thrown rays will quit once they read this particular radius.  Use with care.  
                            #If you want a simulation with r = 6300 m, it might be advisable to make r_limit = 7000 so the boundaries of hulls are still well defined
    ice_model = 'antarctica' #The ice model to use when throwing rays.  To see available options see gnosim.earth.ice.getAcceptedIceModels().
    plot_solution_list = numpy.array(['direct','cross','reflect','direct_2','cross_2','reflect_2']) #The list of solutions types to plot.  To see options see gnosim.trace.refraction_library.getAcceptedSolutions(). 

The above settings are pretty standard.  This would created and prepare libraries for each of the depths in z_array, which each library containing 180 rays. The more rays, the higher to computing time and RAM requirements but the more accurate the interpolation.  

#### RUNNING

Once the script is prepare, the script can be run by opening a python command line terminal in the folder containing gnosim and running

    exec(open('./gnosim/refraction_library.py').read())

#### GENERAL

Following the above steps in creating a library will create a folder containing multiple traces through the ice corresponding to different elevation angles at the antenna. The RefractionLibrary class manages the output trace files.   The user will need to regenerate these ray trace libraries for different antenna elevations, index of refraction profiles, attenuation lengths, etc.

If you are having troubles running you may need to include a line before the first called gnosim import functions to include the path gnosim is located: sys.path.append('<where-gnosim-is>')

*NOTE:  The current algorithm used for calculated the concave hulls is not ideal, and thus requires some attention from the user.  See [Section 1.2.3](#123-concerning-the-envelope-and-concave-hull) for a description of how this is used, how it is calculated, what to be careful about.*

### 1.2.3 Concerning the Envelope and Concave Hull

The set of points created by the ray tracing library will ultimately populate the r-z plane of the ice (flattened cylindrical coordinates) in such a way as to have a concave hull (i.e. the polygon that would surround all points is concave).  The hull is characterized be an envelope function which is used to determine which portion of the r-z plane has possible ray tracing solutions for each of the solution types.  e.g. does a point at *(r,z)* have any reflected solutions connecting the point to the antenna.  The accuracy of the hull is important for maximizing the image-able volume, without falsely classifying points resulting in poor interpolation.

It turns out that generating a concave hull is significantly harder than calculating a convex hull.  Efforts were made in 2018 to update the hull algorithms (previously only concave hulls).  The problem was solved sufficiently, and prioritization did not enable further development of how the hull is calculated.  Because of this the hull calculation is not quite as automated as would be preferable, and may require slight adjustments if the number of rays thrown in a library vary significantly from the values used in development (often 180 rays).  More on this later.

The information describing the concave hull is calculated and stored in the dictionary concave_hull.  Calculations with the hull use a combination of two limits:
  - Upper and lower depth bounds (denoted as *z_min*, *z_max* in concave_hull)
  - A set of radial boundary functions, which interpolate values stored in the dictionary.  *'z_inner_r_bound'* & *'r_inner_r_bound'* described in inner boundary (low values of *r*), and *'z_outer_r_bound'* & *'r_outer_r_bound'* describe the out boundary (high values of *r*).   For a solution to be identified as in a particular hull, the depth must be within the depth range defined by *z_min* and *z_max*, and within the range specified by the inner and outer radial bounding interpolation functions evaluated at that depth. 

The part that requires some attention is how these values for *'z_inner_r_bound'*, *'r_inner_r_bound'*, *'z_outer_r_bound'*, and *'r_outer_r_bound'* are obtained.  Essentially the total depth of the simulation is broken up into bins (with the number of bins being specified in the dictionary as discussed below).  For each bin, the minimum and maximum radii are stored in *'r_inner_r_bound'* and 'r_outer_r_bound', while depth values are stored as *'z_inner_r_bound'* and *'z_outer_r_bound'*. 

The size of these bins defines how well approximates the concave hull is.  If the bins are too fine then the hull can appear jagged at the boundary, jutting inward to points that would clearly have counterparts at further radii if more rays were thrown, if the bins are too coarse then the hull becomes inaccurate/convex, and can include regions such as the shadow that should not be included in the hull.  The number of bins needed for preferable accuracy depends on the solution type, and thus in practice is included in the definition of the concave_hull dictionary in the refraction_library.py module: See [gnosim.trace.refraction_library.getConcaveHullStarter()](https://github.com/djsouthall/gnosim/blob/master/trace/refraction_library.py).

Below is how the hull is defined presently, with the number of bins being set to values that worked well for 180 rays throw in a volume of 3000 m depth and a radius of 6300 m.

    concave_hull = {'direct':{'n_bins':1000},'cross':{'n_bins':1500},'reflect':{'n_bins':2000},'direct_2':{'n_bins':500},'cross_2':{'n_bins':400},'reflect_2':{'n_bins':2000}} # worked for 120 rays

If significant changes are made to the number of rays thrown, or the dimensions of the ice, then please consult this line of code and adjust it as required.  The hulls/envelopes can be plotted and referenced to guide your adjustment of the number of bins.

### 1.2.4 Making Many Libraries with Farm Script

There is a [farm_lib.py](https://github.com/djsouthall/gnosim/blob/master/trace/farm_lib.py) script that can be used for submitting multiple refraction library generation scripts to midway.  The majority of settings are still contained within the refraction library script (which is called by the farm script).  The farm can currently only vary the depths and submit a different depth per batch job, so to vary things like ice model you will need to change the model as defined in [refraction_library.py](https://github.com/djsouthall/gnosim/blob/master/trace/refraction_library.py) and run the farm again.  


## 1.3.0 Preparing a Station Configuration File

Station configuration files are used in the construction of the station in the simulation.  It contains a lot of information and options for customizing the station.  Multiple stations can be supported, each must simply be a different dictionary listed in the *'stations'* section.  The key given to each station should be a unique station label, (for instance, *'ARA5'*).  This label will then be used as the key in dictionaries and labels for plots, etc.

Station configuration files should be located in gnosim/detector/station_config.  Below is a description of the keys to include in the configuration file.  It is recommended that you copy and modify existing configuration files locally, rather than creating your own from scratch.

*Note that certain fields of the station configuration files will be paths to other files.  To use relative paths you should put ${GNOSIM_DIR} in the paths.  These paths are interpreted using os.expandvars which will replace portions of strings with ${VAR} with the corresponding system variable VAR.*

*For example:*

    'system_response':'${GNOSIM_DIR}/gnosim/sim/response/ara_system_response_vpol.npy'

#### Station Configuration Parameters

- **detector_volume : dict**

A dictionary containing information about the ice to be loaded in the simulation.

  - **ice_model : str**

    The desired ice model (label) to be loaded when the ice is initiated.  This will be checked with gnosim.earth.ice.getAcceptedIceModels() to ensure it is an acceptable ice model.
  - **radius : float**

    The radius of the 'detector' (ice to be used as detector/populated with neutrinos).  Given in m.  Small edge effects can be present for neutrinos near the edge of a ray tracing library due to interpolation at a boundary, however signals at these boundaries are often to weak to be detected anyways.  With that in mind, if using a single station then it may be beneficial to make this radius just under the radius used for creating the ray tracing library such that neutrinos aren't populated near edges.  
  For use of multiple stations one should keep in mind that this specifies the region of ice for which neutrinos are being populated.  Thus if you have two stations each with an effective detectable radius (governed by the ray tracing libraries), then this radius should be large enough to place neutrinos in the visible volume of both stations effective radii (accounting for the stations physical separation).  
  
  - **depth : float**

    The depth of the 'detector' (ice to be used as detector/populated with neutrinos).  Given in m.  This should not exceed that used when making the ray tracing libraries defined in the configuration file.  Making it slightly lower than that used in the ray tracing library construction can help avoid edge effects from the interpolation.

- **stations : dict**

  Contains the dictionaries of each station.  Each station dictionary should include the following:
  - **DAQ : dict**

    A dictionary containing relevant quantities for calculations pertaining to digitization, the fpga, and beamforming for this station.

    - **sampling_rate_GHz : float**

        This is the sampling rate of the fpga digitizer.  It will sample from the electric fields at this rate, returning digitized signals.
    Given in GHz.
    - **sampling_bits: : int**

        This sets the number of voltage bins for a digitized signal.  Signals will be digitized asymmetrically about 0 to this bit size with value ranging from -2**(sampling_bits-1)+1 to 2**(sampling_bits-1).
    - **scale_noise_to : int**

        This scales the calculated 'analog' Askaryan calculations (in V) during digitization such that the noise_rms value is scale_noise_to adu.  
    The common use case is to set noise_rms to 3 adu.
    - **n_beams : int**

        The number of beams to be formed when creating a beam forming dictionary. 
    - **n_baselines : int**

        This sets the number of baselines to be considered when creating the beam forming dictionary.  Currently this will automatically select the n_baselines that are smallest (in m).  I.e. if you had 8 antennas separated evenly by 1 m, then n_baselines = 2 would result in both the 1 m and 2 m baselines being used for subbeams.  If they 8 antennas were separated evenly by 2 m, then n_baselines = 2 would result in both the 2 m and 4 m baselines being used for subbeams.  A subbeam is created for each baseline for a given beam (assuming at least 2 antennas are separated by the baseline).  Thus this parameter selects the number of subbeams to be used per beam.  Specified in the configuration file. Currently the minimum time shift is assigned to the smallest baseline.  Thus every other time shift resulting from larger baselines must be a multiple of the minimum baseline. i.e. all subbeam baselines must be in integer multiples of  the minimum baseline.  Currently requires all  other baselines to be an integer multiple of the minimum baseline.
    - **power_calculation_sum_length : int**

        This sets the width (in bins of voltage data points) of a particular power sum window.  This is used in the beamforming calculation during the power sum.
    - **power_calculation_interval : int**

        This sets the number of bins between successive sums in the power sum.  If this is less than power_calculation_sum_length then successive bins will overlap.  The common use case when this was written was to have the interval set to half of the sum length, so successive sums contain half of their values in common with the previous sum.  This is used in the beamforming calculation during the power sum. 
    - **beamforming_power_sum_bit_cap : int**

        This sets number of bits to cap the power sum calculation (which will have units of adu^2).  This is used in gnosim.detector.fpga.fpgaBeamForming.

  - **position : list**

      A 3 element list containing the x,y,z coordinates of the stations origin.  Antenna positions will be specified with respect to these values.  

  - **phased_antennas : dict**

      A dictionary containing the detectors that will be used in the triggering process.  If using beamforming triggering these phased array antennas should all be positions in a single vertical string, otherwise beamforming calculations and preparations may be false.  If you want multiple phased strings then you are currently constrained to treating these as seperate stations.  Each antenna dictionary should be formatted to contain the following:

    - **antenna_definitions : dict**

        A dictionary containing each antennas definition dictionary.  

    - **antenna_key : dict**

        antenna_key is an identifier given to each antenna should be a unique antenna label (within the station), for instance, 'PA00' for the first phased array antenna, or 'RA00' for the first reconstruction array antenna.  This label will then be used as the key in dictionaries and labels for plots, etc.  Ideally these labels should allow someone to distinguish between phased array v.s. reconstruction array antennas.  The corresponding dictionary contains the info relevant for that antenna:

    - **antenna_type : str**

        This is the label of a particular antenna type defined in the code.  This will select how the antenna behaves, i.e. beam patterns, polarization sensitivity, etc.  To see current supported antenna types  try gnosim.detector.detector.getAcceptedAntennaTypes().
    - **lib : str**

        The location of the ray tracing library corresponding to an antenna at this location.
    - **position : list of floats**

        The Cartesian coordinates of the antenna relative to the stations origin.   i.e. if a station location is specified as [0., 0., -173.], and an antenna within that station has position in the config file of [0., 0., -1.], then the absolute ice frame coordinates of the antenna will be [0., 0., -174.].  Given in m.
    - **orientation : list of floats**

        Specifies the Euler angle coordinates for choosing the orientation of the antenna in the ice frame: [alpha_deg, beta_deg, gamma_deg].  Given in degrees.  These angles will be used with a rotation matrix to orient the axis of the antenna using a z-x-z extrinsic rotation.   For additional information see the [gnosim/info/EulerAngleDefiniton.pdf](https://github.com/djsouthall/gnosim/blob/master/info/EulerAngleDefinition.pdf) file.
    - **system_response : str, optional**

        The directory/file containing the system response. See gnosim.detector.detector.Antenna.addTimingInfo() to see how the system response is loaded and handled.  If a new system response is loaded it should have am equally sampled antenna response.  The sampling period for each should be sufficiently small that the calculations of Askaryan radiation do not lose necessary resolution (A sampling period of ~0.1 ns).  It may be necessary to up sample the responses to accommodate this.  Note:  I am aware that upsampling the system/antenna response does NOT increase information about the responses, but the times used to calculate Askaryan radiation are chosen such that they easily match up with the frequencies of the responses.  So information can be lost in the Askaryan calculation if the responses are not sampled at high enough rate because the Askaryan radiation is calculated at that time step.
    - **antenna_response : str**

        The directory/file containing the antenna response. See gnosim.detector.detector.Antenna.addTimingInfo() to see how the antenna response is loaded and handled.  If a new antenna response is loaded it should have am equally sampled system response.  The sampling period for each should be sufficiently small that the calculations of Askaryan radiation do not lose necessary resolution (A sampling period of ~0.1 ns).  It may be necessary to up sample the responses to accommodate this.  Note:  I am aware that upsampling the system/antenna response does NOT increase information about the responses, but the times used to calculate Askaryan radiation are chosen such that they easily match up with the frequencies of the responses.  So information can be lost in the Askaryan calculation if the responses are not sampled at high enough rate because the Askaryan radiation is calculated at that time step.
    - **noise_temperature : float**

        The temperature to be used in the noise calculation.  Given in K.  Note that the noise is also processed by the system response, which may be scaled to obtain a particular noise level for a certain temperature.
    - **resistance : float**

        The resistance to be used in the noise calculation.  Given in Ohms. Note that the noise is also processed by the system response, which may be scaled to obtain a particular noise level for a certain temperature.
    - **frequency_low : float, optional**

        The lower frequency bound for the old and unsupported frequency domain calculation of the Askaryan radiation.  Only used if the selected time domain for the calculation is 'freq'.  electric_field_domain is specified in the simulation configuration file.
    - **frequency_high : float, optional**

        The upper frequency bound for the old and unsupported frequency domain calculation of the Askaryan radiation.  Only used if the selected time domain for the calculation is 'freq'.  electric_field_domain is specified in the simulation configuration file.

  - **reconstruction_antennas : dict**

    A dictionary containing the detectors that will be used for reconstruction.  These antennas are only used in the event that the phased array triggers.  Unlike phased array antennas, these do not need to be on a single string and can be placed throughout the ice as desired.  Each antenna should be formatted in the same way as described above for the phased antennas.  Keep in mind that the antenna labels are station-wide, NOT per group.  So if you label a phased antenna 'dipole0' and a reconstruction antenna 'dipole0', there WILL be issues.  It is best to keep the labels clear and separate.  For instance 'PA00' v.s. 'RA00'.

      

## 1.4.0 Preparing a Simulation Configuration File
  
Simulation configuration files are used to hold the selected settings for running the simulation.  This includes reference to the station configuration files  to be used, as well general options for running.                       

Below is a description of the keys to include in the simulation configuration file.  It is recommended that you copy and modify existing configuration files locally, rather than creating your own from scratch.

- **outfile_dir : str**

  Where the data files will be saved.
- **outfile_name_root : str**

  This will be the first part of the outfiles name.  It will be followed by information about the simulation such as station_config_file_name and the number of neutrino events.  Common choices for this are the date, i.e. 'results_2019_Mar', to distinguish sets of simulations.
- **station_config_file : str**

  The address of the station configuration file.
- **solutions : list of str**

  The solutions to work with in the simulation.  To see the list of accepted solution types see gnosim.trace.refraction_library.getAcceptedSolutions().
  Often either all accepted solution types, or the same list omitting types containing _2, which represent libraries of solution that reflect of the bottom of the ice.  These are often omitted because they double compute time and are assumed to have a small effect on the overall measured results.
- **image_extension : str**

  The image extension for any saved images.  Examples: 'svg', 'png', etc.
- **image_path_root : str**

  The location of the directory for which a new sub-folder with the name of the simulation will be created for output images.
- **electric_field_domain : str**

  Selects the Askaryan calculation type.  Either 'time' (highly recommended) or 'freq'.  'freq' is not fully supported and outdated, will use the frequency domain calculation that was in early builds of GNOSim rather than the new time domain calculation.
- **do_beamforming : bool**

  Enables a the performance of beamforming trigger on digitized signals.  Meta data will be stored in the info section of the outfile pertaining to
  this.  Beams will only be triggered on if trigger_threshold_units is 'fpga'. 
- **include_noise : bool**

  Enables the inclusion of noise in the signals.  If this is False no noise will be in the signals, but digitized signals will still be scaled as if noise were present (i.e. a noise signal will be calculated to get the noise_rms voltage, while will be scaled as specified in the station configuration file).
- **summed_signals : bool**

  If True, then signals resulting from different solution types are combined into a single waveform per antenna.  Otherwise only the waveform of the solution type with the maximum signal per antenna will be used.
- **pre_split : bool**

  Determines whether to attempt to load from pre split libraries.  If true (and the pre split libraries are calculated and saved appropriately) this avoids lengthy calculations which separate the rays into the different solution types. 
- **plot_geometry : bool**

  Enables plotting of the neutrino location, rays, and antennas.  Only plots for trigger events.
- **plot_signals : bool**

  Enables plotting of the waveforms, as well as of the beam forming plots and some meta data.  Only plots for trigger events.
- **trigger_threshold : float**

  The trigger threshold to be applied on the signal or set of signals.  This should correspond to trigger_threshold_units.
- **trigger_threshold_units : str**

  This selects to units used for triggering.  To see the options try getAcceptedTriggerUnits().  If this is 'fpga' then do_beamforming must be True.
- **use_interp_threading : bool**

  Enables multi-threading during the interpolation portion of the code.  Significantly speeds up users perception of start up time (computationally this obviously still takes the same amount of time).  Note that if this is enabled then live plotting is disabled, due to matplotlib currently not being thread safe.  Plots can be generated after the fact. (See gnosim.analysis.testing_single_event.py).
- **use_event_threading : bool**

  Enables multi-threading during the interpolation portion of the code.  Significantly speeds up users perception of total run time (computationally this obviously still takes the same amount of time).   Note that if this is enabled then live plotting is disabled, due to matplotlib currently not being thread safe.  Plots can be generated after the fact. (See gnosim.analysis.testing_single_event.py).
- **output_all_solutions : bool**

  Enables all solution types to be output, otherwise only the solution type with the maximum signal per antenna is output.
- **save_signals : bool**

  Enables saving the waveforms for triggered events.  Waveforms are saved in the output file as a dictionary under the 'signals' header.  This dictionary is organized at the top level by event, then branching to station label.  The signals from a station are in the form of a numpy.ndarray which contains the signals for all antennas in the station.  Each antenna in each station is stored as row in the numpy.ndarray corresponding to that station.  The final row of this ndarray is the times corresponding to the signals in every other row.  Signals are given in units of adu and times in units of ns.

- **pre_trigger_angle : float or list of floats or None, optional**

  If given, then a pre trigger will be applied to the event such that calculations of Askaryan radiation/electric field will only be conducted if ANY of the possible solution types (for all antennas and stations) have an observation angle within pre_trigger_angle number of degrees to the Cherenkov angle.  Essentially the Cherenkov cone must be observable within the pre trigger angular window from at least one antenna in the array in order for the calculations to proceed.

  pre_trigger_angle can be given as a single float (i.e. 10.0), or as a list no more than 2 floats i.e.([5.0,10.0]).  In order for 2 values to be interpretted correctly by yaml they must be given as a list, NOT a tuple or array in the config file.  If a single number is given then it will be used for both the upper an lower (relative) bounds of acceptable angles.  If two values are given the the first value will act as the lower (relative) bound, and the second will be used as the upper (relative) bound.  In this case relative mean that the given angle(s) are relative to the cherenkov angle, and will be added or subtracted from it depending on the bound.

  i.e. if pre_trigger_angle is 10.0 degrees then signals will only be calculated if one of the solutions was emitted (observed on cone) at an angle: theta_c - 10.0 deg < theta_obs < theta_c + 10.0 deg.

  If [5.0,10.0] is given then signals will only be calculated if one of the solutions was emitted (observed on cone) at an angle: theta_c - 5.0 deg < theta_obs < theta_c + 10.0 deg.

  If ANY of the solution types of ANY of the antennas in the entire array satisfies the pre trigger, then all calculations for that event proceed, not just the solution types that independently satisfied the pre trigger threshold.  

  *NOTE:  Setting a pre trigger can speed up the code significantly, but it must be done with care or else information may be lost. It is highly recommended that you first run a simulation with you desired settings and pre_trigger_angle set to None.  Then with this preliminary simulation analyse the losses you expect by not computing these angles (using gnosim/analysis/pre_trigger_set_tool.py for example). Then with this knowledge set the pre trigger for future simulations as you see fit.  This script currently only supports single values pre trigger angles.*

  (Default is None).
- **output_fields : list of str**

  Much of the meta-data for each event is contained within a single structured array.  Internally (while the simulation is running) this array will have the dtype info_dtype, which is specified withinin antarcticsim.  This data type is reduced for ouput, with some fields being required and staying, and others only being present if specified.  This parameter (output_fields) specifies which of the optional fields to include in the output file.

  Below are the lists of required and optional fields.  The required fields are deemed so because they allow for testing_single_event to run without doing a full grid interpolation, saving considerable time when trying to examine just a few events after a larger simulation has been run.  The rest of the fields can be recalculated within testing_single_event using just these fields, and thus are optional.  It is considerably easier to have an optional field enabled (if it is valuable) then to recalculte it with testing_single_event, so it is recommended that you enable the fields you require.  Note that the more fields that are enabled, the more RAM that will be needed and the larger the output file.  Full descriptions of the data corresponding to each of the following fields can befound in [Section 3.3.1](#331-saving-and-loading-h5py-files).

  **Required Fields**
  These fields are already included in the output dtype and do not need to be listed in output_fields.

  - *'eventid'*
  - *'station'*
  - *'antenna'*
  - *'has_solution'*
  - *'triggered'*
  - *'solution'*
  - *'time'*
  - *'distance'*
  - *'theta_ant'*
  - *'theta_ray'*
  - *'a_s'*
  - *'a_p'*
  - *'seed'*

  **Optional Fields**
  These fields are not included on output by default, and must be listed as utes in output_fields for them to be saved in the info dataset.  If you wish to have none of these then you either put an empty list, or you can set output_fields to None, 'None', or 'none'.  If you wish to have all of the fields then you can simply list them all or set output_fields to 'All' or 'all'. 

  - *'pre_triggered'*
  - *'observation_angle'*
  - *'electric_field'*
  - *'electric_field_digitized'*
  - *'fpga_max'*
  - *'dominant_freq'*
  - *'SNR'*
  - *'signal_reduction_factor'*
  - *'polarization_dot_factor'*
  - *'beam_pattern_factor'*
  - *'attenuation_factor'*
  - *'pol_dot_angle'*
  - *'neutrino_travel_dir_vector'*
  - *'emission_wave_vector'*
  - *'detection_wave_vector'*
  - *'emission_polarization_vector'*
  - *'detection_polarization_vector'*
  
- **coords : str or None**

  This is the location of a csv file containing the coordinates of neutrinos to be thrown.  If this is given then the number of events thrown will be overidden to match the number of specified coordinates in this file.  To see an example of how such a csv file can be created see gnosim/utils/generate_event_orientations.py.  The columns are expected to be: *x_0*, *y_0*, *z_0*, *phi_0*, *theta_0*.  Definitions of these can be found in the documentation for [gnosim.sim.antarcticsim.sim.throw()](https://github.com/djsouthall/gnosim/sim/antarcticsim.py).


# 2.0.0 Running the Simulation

This section discusses the actual running of the simulation, once preparations have been made.

## 2.1.0 Running a Single Instance of GNOSim

An instance of GNOSim is launched using the following command in a Linux terminal from the folder containing gnosim:

    ./gnosim/sim/antarcticsim.py PATH_T0_SIMULATION_CONFIG ENERGY_NEUTRINO N_EVENTS INDEX [SEED]

This command will be referenced as LAUNCH_COMMAND in future sections.  The contents of the above line are discussed below.

#### Command Parameters
------------------

- **[./gnosim/sim/antarcticsim.py](https://github.com/djsouthall/gnosim/sim/antarcticsim.py) : executable file**

  This is the antarcticsim file.  The __main__ script will be run by placing this as the command in the command line.  The following parameters will be used as input parameters to this script, and must be ordered correctly.
- **PATH_T0_SIMULATION_CONFIG : path**

  This should be the address of the simulation file.  For example: [./gnosim/sim/sim_settings.py](https://github.com/djsouthall/gnosim/sim/sim_settings.py)
- **ENERGY_NEUTRINO : float**

  This is the energy of the neutrinos to be thrown.  Given in GeV. *Note that in practice within the simulation the energy that actually goes into the calculation of the Askaryan radiation is a reduced energy that accounts for inelasticity of the interaction in the ice.  
- **N_EVENTS : int**

  The number of neutrinos to throw for this simulation.
- **INDEX : int**

  The index of this simulation.  This is useful for instance if several simulations are being run with the same settings to be stitched back together in analysis (running a large number of events can reduce the required memory per job). 
- **SEED : int, optional**

  This is an integer which will seed the entire simulation.  Allows for the simulation to be reproducible.  Be careful to make this different for each job if you are intending on running multiple jobs with the same settings and stitching them back together, but don't want duplicate events.  If this is not given then either a default seed will be used (if altered within the antarcticsim to be a value other than None), or if the default seed is None (Default), then no seed is used.

  This seed (or lack thereof) is used to calculate an individual seed for each event in the simulation using the command: 
  
      numpy.random.randint(numpy.iinfo(numpy.uint32).max,size=self.n_events)
  This allows for reproducibility of individual events through scripts like gnosim.analysis.test_single_event, which can be used to learn more about a particular event after the simulation has run, but to reproduce the entire simulation these event seeds and other rand properties are dependent on SEED.

## 2.2.0 Memory Allocation for GNOSim and Running on Midway

Due to the interpolation being performed using large ray tracing libraries, the simulation is generally very memory intensive.  Because of this it is recommended that you use a machine capable of allocating a lot of memory (such as the Midway cluster at UChicago).  It does not matter how this memory is available, but if multi-threading is enabled then it is in general best to request as many nodes as possible, each with as little memory as needed such that *n_cpu* \* *mem_per_cpu* is larger than the memory needed to run the simulation.  Determining the ideal configuration for your simulation may require some trial and error.  

### 2.2.1 Running with Batch

For UChicago/Midway it was common when running simulations with about 1 million events to run with the resources specified in the following command:

    sbatch --cpus-per-task=16 --mem-per-cpu=1500 --job-name=gnosim LAUNCH_COMMAND

For information on LAUNCH_COMMAND please see [Section 2.1.0](#210-running-a-single-instance-of-gnosim).  When sending a command to Slurm using these flags, we request 30 GB of memory per job, with the ability to use up to 16 threads per job.  The amount of memory requested here may be more than is necessary depending on the application.

The status of this job can be monitored using the command:

    squeue --user=USERID

While running a slurm.out file will be made in the current directory.  This should be updated as print statements are made to the command line in the simulation (barring some issues with Pythons print buffering), and thus can be monitored to see where the simulation is.  This file can be monitored live used the command:

    tail -f slurm.out

Which will continue to show the end of the slurm file as it updates.  

This method of running jobs is used in the farm.py script which automates the running of many simulations (see [Section 2.3.0](#230-running-many-simulations-with-farm-script)).

### 2.2.2 Running in an Interactive Terminal

The simulation can also be run in an interactive terminal by launching the terminal using the command:

    sinteractive --cpus-per-task=16 --mem-per-cpu=1500 --job-name=gnosim

Within this terminal the LAUNCH_COMMAND can be run from the folder containing gnosim.  Here the print statements of the simulation will appear in the terminal as normal, so no special monitoring is required.

### 2.2.3 More Info
Please see https://rcc.uchicago.edu/docs/using-midway/index.html for information on using Midway.  Additionally it may be helpful to look up common Slurm commands.

## 2.3.0 Running Many Simulations with the gnosim/sim/farm.py Script

It is often desirable to initiate/queue many simulations at once.  For this one can use (modify locally as needed) the gnosim/sim/farm.py script.  This script contains a sample of the infrastructure that can be used to queue many jobs with a range of energies or simulation configuration files.  The parameters for running this script are mostly self explanatory.  

The status of each job can be monitored using the command:

    squeue --user=USERID

To see the command line print statements of a specific job, locate the slurm file (formatted slurm-JOBID.out) with the appropriate JOBID from the squeue command.  Then run the command:

    tail -f slurm.out

Which will continue to show the end of the slurm file as it updates.  

## 2.4.0 Stitching Together Simulations

Sometimes when running simulations with large numbers of events it is desirable to split the events across multiple instances of the simulation, such that individual jobs require less memory, or potentially so the events can be run faster in user time (obviously computation time does not significantly change as mostly the same number of calculations are performed).  If this is done then a different index should be used for each simulation instance.

To recombine the meta-data and outfiles of these simulations one can use (or modify to their needs locally) the [gnosim/analysis/quickmerge.py](https://github.com/djsouthall/gnosim/analysis/quickmerge.py) script, which provides a general infrastructure for combining the output files.  There is no claim that this is the best way to recombine, but it is currently how that is done.  

Recombining requires significant memory, so it is recommended to run this in a batch command or from an interactive terminal with access to a lot of memory.  It is currently designed to be run in a particular directory of the output files.  It will try to combine files unless a file exists already with the expected  *'merged'* output name.  The output file is generated for a particular set of simulations first, so this script can be run multiple times in a directory and each instance will only work on files that aren't already being worked on (given that the instances are run with enough time separation to allow for the output file to be made by one script to be noticed by the other).  This is not the most elegant of solutions, so be careful when using.

The output files will have events appended into the data files with new event numbers.  The attributes are crudely copied and appended with the index of the particular file for which it came from.

# 3.0.0 Working with Simulation Data

This section contains information about working with the output data from the simulation. 
  
## 3.1.0 Information About h5py

The data is stored in h5 (HDF5) files produced using the h5py python package (http://www.h5py.org/).  For full details on working with this type of data it is recommended that the user reference the website or the textbook *'Python and HDF5 - Unlocking Scientific Data' by Andrew Collette (O'Reilly Publishing)*.  A basic overview is discussed below.

To work with h5py files the package must be loaded in python using the command:

    import h5py

### 3.1.1 Saving and Loading h5py Files

In order to create or save an h5py file it must be opened in 'write' (below as 'w'):

    file = h5py.File(OUTFILE_NAME, 'w')

where OUTFILE_NAME is a string of the name of the output file, including extension - e.g. './results.h5'.

If loading a file and writing is not required then you can load the file in 'read' mode:

    file = h5py.File(INFILE_NAME, 'r')

After the file is no longer needed it should be closed:

    file.close()

### 3.1.2 Attributes and Datasets in h5py Files

The h5py files used in this simulation store meta information as either 'attrs' (attributes) or datasets.  To quickly see the contents of an h5py file, open it using python and the following commands:

    import h5py
    reader = h5py.File('/PATH/TO/FILE.h5' , 'r')

Then you can list the available attrs and data sets using the following:

    print(list(reader.attrs)) #Prints attribute labels

    print(list(reader.keys())) #Prints name of datasets available
    
    print(reader['info'].dtype) #Prints the fields available in the info dataset

Attributes and datasets are described broadly below, with more details in the following sections.

#### Attrs

Attributes are often one-off meta-data like information about the simulation.  The attributes are stored in the file essentially in a
dictionary like structure.  For instance the name of the station configuration file is stored in output files using the command:

    file.attrs['config'] = config_file #Where config_file is a string containing the location of the station configuration file

And can later be accessed from a loaded file using the command:

    station_configuration_file = station_file.attrs['config'] 

#### Datasets

Datasets are the central feature of HDF5. You can think of them as NumPy arrays that live on disk. Every dataset in HDF5 has a name, 
a type, and a shape, and supports random access. Unlike the built-in np.save and friends, there?s no need to read and write the entire 
array as a block; you can use the standard NumPy syntax for slicing to read and write just the parts you want.' - Chapter 3 of 
*'Python and HDF5 - Unlocking Scientific Data' by Andrew Collette*.

Datasets can be added to a file using a command such as:

    file.create_dataset('DATASET_NAME', (LENGTH,), dtype='DTYPE', compression='gzip', compression_opts=9, shuffle=True)

As discussed in Chapter 7 of *'Python and HDF5 - Unlocking Scientific Data'*, HDF5 supports many data types, however the main ones used
in the simulation are integer, float, strings, and, compound.  Compound dtypes utilize the numpy's structured data types, which are used
in the simulation for the 'info' data set and info_dtype objects (More in [Section 3.2.0](#320-formatting-of-output-files)).  

Then with this dataset present, it can be populated either in complete blocks or using numpy-like slices/indexing:

    file['DATASET_NAME'][...] = numpy.ones(LENGTH) #Example of saving entire block

    file['DATASET_NAME'][INDEX : INDEX + N_ANTENNA] = numpy.ones(N_ANTENNA) #An example of filling just a slice.

Reading from the data set works similarly.  They can be loaded completely in blocks, or with slices, or if the dtype supports it,
with slices on keys as well  (See [Section 3.2.0](#320-formatting-of-output-files) for slicing on keys).  Examples of loading are below:
    
    ALL_VALUES = file['DATASET_NAME'][...]

    SOME_VALUES = file['DATASET_NAME'][INDEX : INDEX + N_ANTENNA]   

## 3.2.0 Formatting of Output Files

Below are descriptions of the many attributes and datasets stored within a gnosim output file.

#### Attrs
- **seed : int or str**

  The seed used for the simulation.  Will be the str 'None' if no string was used.
- **geometric_factor : float**

This is the 'geometric_factor' calculated for the simulation, which is often used as a pre factor in volumetric acceptance calculations.  Given in m^3 sr.
- **config : float**

  This is the location of the station configuration file used for the simulation.
- **ice_model : str**

  The used ice model (label).
- **trigger_mode : str**

  The units/mode of trigger used.  Corresponds to trigger_threshold_units from the simulation configuration file.
- **trigger_threshold : float**

  The threshold set for triggering.  The bool signifying if a particular event triggered or not is stored in the 'info' dataset.
- **pre_trigger_angle : float or list of floats or None, optional**

  This is the pre trigger angle(s) used.  If None was used then this will be 'None'.


#### Datasets


- **'event_seeds' : int**

  The seeds that set the states of the random objects to be used internally for each event, allowing it to be reproducable regardless of the number of times random calls were made externally.  This is calculated using:

      numpy.random.randint(numpy.iinfo(numpy.uint32).max,size=self.n_events)

  where the seed of this random call is set by the simulation seed.

- **'energy_neutrino' : float**

  Energy of each event.  Given in GeV.  Length is n_events.  This is the full energy before inelasticity is accounted for.
- **'inelasticity' : float**

  Inelasticity of each event.  Length is n_events.
- **'x_0' : float**

  x coordinate of the neutrino interaction for each event site in the ice frame.  Given in m.  Length is n_events.
- **'y_0' : float**

  y coordinate of the neutrino interaction for each event site in the ice frame.  Given in m.  Length is n_events.
- **'z_0' : float**

  z coordinate of the neutrino interaction for each event site in the ice frame.  Given in m.  Length is n_events.
- **'theta_0' : float**

  The polar spherical coordinates for the directions each neutrino came from.  Given in degrees.  Length is n_events.
- **'phi_0' : float**

  The azimuthal spherical coordinates for the directions each neutrino came from.  Given in degrees.  Length is n_events.
- **'p_interact' : float**

  The probability that the neutrino interacts in a sphere containing a cubic meter of ice for each event.  This may be used for weighting in some analysis scripts, however currently it's exact definition and relevance for use in weight is not entirely clear.  See gnosim.earth.earth.probInteract().  Length is n_events.
- **'p_earth' : float**

  The probability of survival for the neutrino passing through the earth.  These are used on volumetric acceptance calculations.  See [gnosim.earth.earth.probSurvival()](https://github.com/djsouthall/gnosim/earth/earth.py).  Length is n_events.
- **'p_detect' : bool**

  A bool for each event signifying if the event is observable (meaning there is at least one ray connecting the neutrino interaction site an antenna in the array).  True if the event is observable at all (i.e. has at least one solution type visible across the array).  Length is n_events.
- **'random_time_offsets' : float**

  A small random jitter in timing to ensure that no systematic error is introduced from perfect timing in the MC simulation.  Given in ns.  Length is n_events.
- **'info' : info_dtype**

  Contains the meta data for the event for each antenna and solution type (if output_all_solutions == True, which is defined in the simulation configuration file).  For more information about the contents of this see [Section 3.3.0](#330-working-with-info_dtype-objects-for-meta-data).  Length is n_events*n_antennas if output_all_solutions == False, otherwise it is n_events*n_antennas*n_solution_types.  Two info_dtypes will exists within the simulation, the main: info_dtype which contains ALL meta-data fields, and the output: out_info_dtype, which contains only the required fields for reconstruction offline without interpolation, plus those fields selected in the 'output_fields' section of the simulation configuration file.  See [Section 1.4.0](#140-preparing-a-simulation-configuration-file) for more information on specifying output fields.

## 3.3.0 Working with info_dtype Objects for Meta-data

Most of the information and meta-data about the individual events is stored in the 'info' dataset of the output files.  The data within this dataset is stored using the out_info_dtype numpy structured dtype.  [Section 3.3.1](#331-info-content-descriptions) discusses the contents.  [Section 3.3.2](#332-working-with-the-info-dataset) discusses how to work with this data.  

### 3.3.1 Info Content Descriptions

info_dtype is defined within antarcticsim.  Search for it in that file to see the definition used.  Below are descriptions of the keys/contents in the data type.  The info dataset often as an 'entry' for every solution type of every antenna in the array.  Thus below I will be using the terminology of 'per entry' rather then 'per event', because multiple entries will correspond to a single event, but describe different solutions/antennas.


- **eventid : int**

  The identifier for the event within the simulation.  Corresponds to the index in most other datasets.
- **station : int**

  The index of the station for each entry.
- **antenna : int**

  The index of the antenna for each entry.
- **solution : S10**

  The solution type for this particular entry.  Stored as a fixed with string.  This may need to be decoded for use, as it will appear as a b-str.  i.e. this might look like *b'direct'*, rather than *'direct'*.  So you may need to do *solution.decode()*.
- **has_solution : bool**

  True if there is a ray connection the neutrino interaction location to the antenna that is corresponds to this entries solution type (defined in solution).
- **pre_triggered : bool**

  Whether this particular solution satisfied the pre trigger.  It passed the pre trigger if True.
- **triggered : bool**

  True for all entries of a particular event if the event as a whole was triggered on.
- **time : float**

  The time of flight for this particular entry.  Given in ns.
- **distance : float**

  The distance traveled for this entry.  Given in m.
- **theta_ant : float**

  The spherical polar coordinate of the momentum vector of the ray pointing along the ray towards the neutrino event in the ice frame at the antenna.  In other words: The spherical polar angle of the direction of the ray connecting the neutrino and the antenna as interpolated to be at the antenna location.  This points in the opposite direction of the pulse travel direction (i.e. opposite to 'towards the antenna from the neutrino').  Given for this particular entry.  Given in degrees. 
- **theta_ray : float**

  The spherical polar coordinate of the momentum vector of the ray pointing along the ray towards the neutrino event in the ice frame at the neutrino.  In other words: The spherical polar angle of the direction of the ray connecting the neutrino and the antenna as interpolated to be at the neutrino interaction location.  This points in the opposite direction of the pulse travel direction (i.e. opposite to 'towards the antenna from the neutrino').  Given for this particular entry.  Given in degrees. 
- **observation_angle : float**

  The observation angle relative to the shower axis.  i.e. The angle 'on-cone'.  This is the angle used in the pre trigger.  Given in degrees.
- **electric_field : float**

  The maximum value of the electric field for this entry.  Taken after noise was added (if noise was added), thus for small signals may only be the noise max.  Given in V.
- **electric_field_digitized : float**

  The maximum value of the electric field for this entry.  Taken after noise was added (if noise was added), thus for small signals may only be the noise max.  Given in adu.
- **fpga_max : int**

  The maximum measured beamforming value for this event.  Given in adu^2.  If do_beamforming = False then this will be -999.0.
- **dominant_freq : float**

  This is the frequency of the bin corresponding to the maximum power in the frequency spectrum of the Askaryan radiation.  Taken after noise was added (if noise was added).
- **a_p : float**

  *p* polarization attenuation factor.  Includes effects from general attenuation in ice due to attenuation length (currently only for 300 MHz), as well as reduction in signal resulting from Fresnel coefficients (as well as sign flips resulting from Fresnel amplitudes).  Corresponds to this entry.
- **a_s : float**

  *s* polarization attenuation factor.  Includes effects from general attenuation in ice due to attenuation length (currently only for 300 MHz), as well as reduction in signal resulting from Fresnel coefficients (as well as sign flips resulting from Fresnel amplitudes).  Corresponds to this entry.
- **SNR : float**

  SNR is calculated as the ratio of the 'peak to peak'/2 over rms(noise), squared (ratio of powers).  The 'peak to peak' is calculated using the NOISELESS signal, allowing for the SNR to be accurate far below noise level (a privilege of simulation land). 
- **signal_reduction_factor : float**

  The calculation method for this is specific to the detector type selected for the antenna of this entry (defined in the station configuration file).  See [gnosim.detector.detector.Antenna.getAntennaResponseFactor](https://github.com/djsouthall/gnosim/detector/detector.py). This is the reduction factor that was multiplied with the antenna response for this entries Askaryan calculation.
- **beam_pattern_factor : float**

  The beam pattern componant of the signal_reduction factor.  I.e. the reduction in signal strength due to the approach angles of the wave vector at the antenna.  This componant may be calculated depending on the type of antenna chosen.  If it is not calculated for a particular antenna type then a value of -999.0 is returned.  To see how this factor was specifically calculated for your antenna type see the code below.
- **attenuation_factor : float**

  The attenuation componant of the signal_reduction factor.  The contains the affects of a_s and a_p (the attennuations for each polarization of light).  Depending on the type of detector chosen this could entail simply returning one of those values, or accounting for the appropriate vectorized proportion of each componant on the final signal.  This componant may be calculated depending on the type of antenna chosen. If it is not calculated for a particular antenna type then a value of -999.0 is returned.  To see how this factor was specifically calculated for your antenna type see the code below.
- **polarization_dot_factor : float**

  The polarization componant of the signal_reduction factor (i.e. the factor resulting from the dot product between the polarization vector at the antenna and the polarization sensitivity vector set for that antenna).  This componant may be calculated depending on the type of antenna chosen. If it is not calculated for a particular antenna type then a value of -999.0 is returned.  To see how this factor was specifically calculated for your antenna type see the appropriate section of gnosim.detector.antenna.getAntennaResponseFactor().
- **pol_dot_angle : float**

  The angle from the dot product of the polarization vector and the polarization sensitivity vector.  Only works for detectors sensitive to polarization, will return -999.0 otherwise. 
- **seed : int**

  This is the event specific seed.  
- **neutrino_travel_dir_vector : numpy.ndarray of floats**

  The unit vector for the direction the shower is propogating.  This is returned in ice-frame cartesian coordinates.
- **emission_wave_vector : numpy.ndarray of floats**

  The unit vector for the vector directed towards the antenna along the observation ray.  This is returned in ice-frame cartesian coordinates.  This should be the wave vector as it was emitted from the neutrino.
- **detection_wave_vector : numpy.ndarray of floats**

  The unit vector for the vector directed towards the antenna along the observation ray at the antenna.  Given in the ice-frame cartesian coordinates.
- **emission_polarization_vector : numpy.ndarray of floats**

  The unit vector for the polarization as it is just after emission at the neutrino. This is a unit vector.  This is returned in ice-frame cartesian coordinates.
- **detection_polarization_vector : numpy.ndarray of floats**

  The unit vector for the polarization as it is just before interacting with the antenna. This is a unit vector, magnitudes represent how the *s* and *p* polarizations have been reduced during ray propogation.  The magnitude of this vector originally contained information about *a_s* and *a_p*, however the magnitude has been split from the direction and is stored in attenuation_factor output value. This vector is returned in ice-frame cartesian coordinates.


### 3.3.2 Working with the Info Dataset

The info data set can be extracted as a whole using the command:

    info = file['info'][...]

However individual portions of the info object can be loaded from the h5py file without loading the entire object into memory.  This can be done to obtain a single list if information, or to select on the components required for a particular calculation.  It is common to load in one set of information to make a cut on to use subsequently for loading other information.  For instance, if I only wanted to look at triggered events I could do the following:

    bool_triggered = file['info']['triggered'] 
    
This will be an array of bools.  If the event triggered then those entries will have True at their index.

    info_triggered = file['info'][numpy.where(bool_triggered)[0]]

This will load only entries corresponding to events that were triggered on.

    triggered_event_ids = numpy.unique(info_triggered['eventids'])
    
This will get a list of events that triggered, which can be used to index other datasets.

The above list of triggered event ids could alternatively be calculated as seen below.

    my_info = numpy.unique(reader['info']['eventid','triggered'])
    
An array containing the event id and trigger bool for each entry, Because eventid and triggered are the same for each entry in a given event, this unique function will cause my_info to consist of only one entry per event.  Thus below I do not need a second unique call. Doing it here makes the below line faster as the cut is operating on less elements.

    triggered_event_ids = my_info['eventid'][ my_info['triggered'] == True ] 
    
This returns the 'eventid' section but sliced by  the cut on the 'triggered' section.

Note that when working with these compound data types, you should first select the key and then apply the cut.
It will work both ways with reading the information, but if you wish to set values to a certain cut, the cut
must be applied second.  i.e.:

WILL WORK: 

    my_info['eventid'][ my_info['triggered'] == False ] = -999 #will set the eventid for all non-triggered events to -999

WON'T WORK

    my_info[ my_info['triggered'] == False ]['eventid'] = -999 #This will not actually set the values to -999 in my experience.


## 3.4.0 Analysis Scripts

As a full analysis has not been done with the simulation code, only some analysis files exist.  Those that exist can be used as examples for future development.  Ideally as the simulation is worked with, interesting analysis scripts will be developed and added to the git repository to allow for general use.  The sub sections below discuss some existing analysis scripts.  Note that these are not as maintained or thought out/generalized as some things in the main portion of the simulation, and may require some tinkering to ensure they work as you desire.  Analysis file should stay within the gnosim/analysis folder for organization purposes.

### 3.4.1 Volumetric Acceptance

The volumetric acceptance is a common thing to calculate for a particular set of simulations.  During development the script gnosim/analysis/volumetric_acceptance.py was used for these calculations. 

### 3.4.2 Neutrino Approach Test

This script was written for a long gone build of the code, and was supposed to plot the approach angles of the neutrinos (potentially with some weighting) as a check on if it made sense.  The code is still there in case it will some day be adapted for the recent code build. 

### 3.4.3 Arrival Times Testing

This script was written for a long gone build of the code, and was supposed to plot times of arrival across the different antennas and compare this to the expected values for different angles.  This code was used while the hull was being improved but has not been touched since.  The code is still there in case it will some day be adapted for the recent code build.

## 3.5.0 Utils Scripts

Sometimes a script will be made and added to the utils section rather than analysis, as it is just for general use or is useful before the simulation is run and not necessarily as part of the analysis.  Here are descriptions of some of those scripts.

### 3.5.1 Generate Event Orientations

Script: [https://github.com/djsouthall/gnosim/utils/generate_event_orientations.py](gnosim/utils/generate_event_orientations.py)
This is a small script which can be used to create a csv file containing coordinates of on-cone neutrino events surrounding
chosen antenna locations at a defined radius (either spherical or cylindrical radius).  These location of the csv containing 
these coordinates can thn be used as the input parameter 'coords' in sim configuration files.

### 3.6.0 Converting Signals to Data-Like Format

TODO: Create a script that can do this.

# 4.0.0 Information For Contributors

This section will contain information relevant for people intending on contributing to the development of the simulation.

## 4.1.0 Code Repository

        [Midway Repo]
        https://github.com/djsouthall/gnosim/


## 4.2.0 Conventions

The following conventions are ideally followed:

#### Naming

    package_name
    module_name.py
    ClassName
    functionName
    variable_name

#### Documentation

Each function and module should be properly documented with both contributors and users in mind.  The formatting should follow the NumPy Style for Python Docstrings.  It is recommended that you look at some of the main modules to see examples of how the documentation is intended to look. Typically single quotes are used over double quotes. Eventually it would be good to use a tool like sphinx and the sphinx.ext.napoleon parser to turn these docstrings into a navigable html structure.  


#### Coordinates

    x = east (currently arbitrary)
    y = north
    z = elevation (negative values are below surface)
    theta = zenith angle
    phi = azimuth angle

#### Units

    SI units are used throughout for all electricity and magnetism calculations.
    distance (m)
    time (ns)
    frequency (GHz)
    mass (kg)
    See gnosim/utils/constants.py for unit conversions.

## 4.3.0 TODOs

If you are coding and find something that should be changed (and you do not intend to change it right away), please put a comment near the desired change location with the following formatting: 

    #TODO : This section should be changed to this different way because of this reason. 

This way those fleeting thoughts of improvement are not lost, and can be easily found.  


