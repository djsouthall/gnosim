#ARATotal,ARAVPol,NuPhaseOn/OffBeam, and 1SigThresh all come from the Design and Perfomance paper.
#Ben Hokanson-Fasig supplied the Pyrex data.  
#x-axis is in PeV
{ 
'ARATotal':     {
                'x':[10,31.49255419,100,312.3371592,991.7809693,3149.255419,10000,31492.55419,100000,317535.3749],
                'y':[0.01632925,0.099823306,0.445892309,1.676913322,4.561671741,12.03789725,24.66534102,42.12177618,62.42863334,86.19692011],
                'style':'ro-',
                'label':'ARA total'
                },
'ARAVPol':      {
                'x':[10,31.49255419,100,314.9255419,1000,3149.255419,10000,31753.53749,100000,317535.3749],
                'y':[0.0136096,0.074431955,0.379227926,1.546471216,4.249627469,11.56027631,24.41695596,40.45016708,59.95168723,85.3288994],
                'style':'ro--',
                'label':'ARA Vpol'
                },
'NuPhaseOnBeam':{
                'x':[10,31.75353749,99.17809693,317.5353749,1000,3149.255419,10082.87143,31492.55419,100000,314925.5419],
                'y':[0.036326118,0.215423931,0.942993383,3.440255117,8.122122623,16.14444648,30.19918854,44.75899253,65.00791621,87.07456059],
                'style':'ko-',
                'label':'NuPhase on-beam (1.6$\sigma$)'
                },
'NuPhaseOffBeam':{
                'x':[10,31.49255419,100,314.9255419,1000,3149.255419,10000,31753.53749,100000,317535.3749],
                'y':[0.023507584,0.143705606,0.770180678,2.43861948,6.500791621,13.59245441,25.42575966,42.12139417,59.95168723,79.49263869],
                'style':'ko--',
                'label':'NuPhase off-beam (2.1$\sigma$)'
                },
'1SigThresh':   {
                'x':[10,31.75353749,99.17809693,317.5353749,1000,3149.255419,10000,31492.55419,100000,317535.3749],
                'y':[0.065337731,0.360968887,1.532842935,4.756089189,10.25105215,19.96782914,35.15064335,52.09719832,73.40294726,93.46662986],
                'style':'bo:',
                'label':'1.0$\sigma$ Threshold'
                },
'PyRex200m':    {
                'x':[3.16227766e+00, 1.00000000e+01, 3.16227766e+01, 1.00000000e+02, 3.16227766e+02, 1.00000000e+03, 3.16227766e+03, 1.00000000e+04, 3.16227766e+04, 1.00000000e+05],
                'y':[6.042e-3,1.099e-1,6.822e-1,2.769e+0,8.447e+0,1.999e+1,3.955e+1,7.090e+1,1.479e+2,2.263e+2],
                'style':'mo:',
                'label':'PyRex 200m 1.5$\sigma$ Threshold'
                }
}






