'''
    SCRIPT TO GENERATE PLOT OF MULTISCALE EVOLUTION OF METRICS IN THE REF & PRT ENSEMBLES
    Variables examined: ps, bottommost layer q, bottomost layer t


    Script structure:
    -----------------
    1) User settings
            Hard-coded user settings that controls operation of the script. These settings
            include dictionaries that indicate information about (i) template paths of pickle
            files containing various statistical metrics, (ii) better names for variables and
            metrics, (iii) flags relating to the aggregation of metrics.
    
    2) Function: load_data_on_multiscale_evolution_of_metric
            Loads and aggregates metrics for plotting. Note that the function relies on the 
            variables specified in the user settings.

    3) Function: plot_data_on_multiscale_evolution_of_metric 
            Plots the data loaded by function load_data_on_multiscale_evolution_of_metric
    
    4) Main program
            Calls the two functions to generate the plots.


    Authorship history (in chronological order):
    --------------------------------------------
    - SS & JC wrote python code to decompose SPEED ensemble into wavebands
    - MA & AR wrote python code to generate pkl files of multiscale metrics
    - JC wrote slurm job submission codes to run MA's & AR's pkl file generator
    - MA & AR drafted code to visualize data in pkl files
    - JC synthesized MA's & AR's visualization codes into this script.

    JC = Joseph Chan, SS = Shuvo Saurav, MA = Max Albrecht, AR = Aiden Ridgeway
'''


'''
    IMPORT PACKAGES
'''
import numpy as np
import pickle as pkl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from copy import deepcopy
from gc import collect as gc_collect




'''
    USER SETTINGS
'''
# Dictionary containing some meta data about variables examined
vinfo_dict = {
    'ps': {'pretty': 'Psfc', 'unit': 'hPa'},
    't':  {'pretty': 'Temp', 'unit': 'K'},
    'q':  {'pretty': 'Qvap', 'unit': 'kg/kg'},
}

# Dictionary containing some meta data about waveband examined
wbinfo_dict = {
    "data_raw": 'All Scales', 
    "L_scale":  'Lrg Scales',
    "M_scale":  'Med Scales',
    "S_scale":  'Sml Scales'
}

# Dictionary containing meta data about ensembles examiend
ensinfo_dict = {
    "reference_ens": 'Ref Ens', 
    "perturbed_ens": 'Prt Ens'
}


# Dictionary containing meta data about metrics examined
# ENAME will be overwritten with ensemble name, VNAME will be overwritten with variable name, etc.
minfo_dict = {
    'variances': {
        'filepath': "diagnostic_pkl_files/diagnostics_max/ENAME/WNAME/VNAME/VNAME_ENAME_DATE_variances.pkl",
        'sq flag': False,
        'fullname': 'RMS Std Dev',
        'rms flag': True
    },
    'skewness': {
        'filepath': "diagnostic_pkl_files/diagnostics_aiden/ENAME/WNAME/VNAME/VNAME_ENAME_WNAME_DATE_skewness.pkl",
        'sq flag': True,
        'fullname': 'RMS Skewness',
        'rms flag': True
    },
    'kurtosis': {
        'filepath': "diagnostic_pkl_files/diagnostics_aiden/ENAME/WNAME/VNAME/VNAME_ENAME_WNAME_DATE_kurtosis.pkl",
        'sq flag': True,
        'fullname': 'RMS Kurtosis',
        'rms flag': True
    },
    'p_values':{
        'filepath': "diagnostic_pkl_files/diagnostics_max/ENAME/WNAME/VNAME/VNAME_ENAME_DATE_p_values.pkl",
        'sq flag': False,
        'threshold': 0.05,
        'fullname': 'Gaussianity',
        'rms flag': False
    }
}

# List of latitudes
lat_list = np.linspace(-87.21645, 87.21645, 48)

# List of all dates
date_st = datetime( year=2011, month=1, day=1 )
date_ed = datetime( year=2011, month=3, day=1 )
date_interval = timedelta( days=1 )
date_list = []
date_nw = deepcopy( date_st )
while date_nw <= date_ed:
    date_list.append(date_nw)
    date_nw += date_interval














'''
    FUNCTION TO LOAD DATA FOR A SPECIFIC METRIC

    Inputs:
    1) mname -- metric name (variances, skewness, kurtosis, sw_pval)
    2) date_list -- list of datetime objects that correspond to dates of interest

    Output:
    1) metric_data_dict -- Dictionary containing longitudinally averaged metrics
                      Layers: ensemble, waveband, variable
'''
def load_data_on_multiscale_evolution_of_metric( mname, date_list ):

    # Dictionary to hold metric data after loading.
    metric_data_dict = {}

    # number of dates
    num_dates = len( date_list )


    # Loop over all combinations of ensemble, waveband, and variable
    for ensemble_name in ensinfo_dict.keys():
        metric_data_dict[ensemble_name] = {}

        for waveband_name in wbinfo_dict.keys():
            metric_data_dict[ensemble_name][waveband_name] = {}

            for variable_name in vinfo_dict.keys():
    
                # Init temporary list to hold metric
                metric_data_dict[ensemble_name][waveband_name][variable_name] = [None for dd in range(num_dates)]

                # Loop over all dates
                for dd, date_nw in enumerate( date_list ):

                    # Generate pkl file path
                    fname = deepcopy( minfo_dict[mname]['filepath'] )
                    fname = fname.replace( 'ENAME', ensemble_name )
                    fname = fname.replace( 'VNAME', variable_name )
                    fname = fname.replace( 'WNAME', waveband_name )
                    fname = fname.replace( 'DATE', date_nw.strftime('%Y%m%d%H%M') )

                    # Load metric from pickle file
                    with open( fname, 'rb' ) as fhandle:
                        metric_arr = (pkl.load( fhandle ))[mname]

                    # Kill vertical dimension, if present
                    if len( metric_arr.shape ) == 3 and metric_arr.shape[0] == 8:
                        metric_arr = metric_arr[-1]  # Select only bottommost dimension
                    elif len( metric_arr.shape ) > 2:
                        print('Error: loaded metric data has too many dimensions')
                        print( ensemble_name, variable_name, waveband_name, date_nw, mname )
                        print( metric_arr.shape )
                        quit()
                    # --- end of vertical dimension slaughtering.

                    # Square variable if necessary
                    if minfo_dict[mname]['sq flag']:
                        metric_arr = metric_arr **2

                    # Perform thresholding if needed -- used for Shapiro-Wilks
                    if 'threshold' in minfo_dict[mname].keys():
                        metric_arr = metric_arr < 0.05
                    
                    # Average over longitude dimension & aggregate
                    metric_arr = np.mean( metric_arr, axis=-1)

                    # Is this an RMS variable?
                    if minfo_dict[mname]['rms flag']:
                        metric_arr = np.sqrt( metric_arr )

                    metric_data_dict[ensemble_name][waveband_name][variable_name][dd] = metric_arr
                
                # --- End of loop over dates

                # Convert data list into array
                metric_data_dict[ensemble_name][waveband_name][variable_name] = np.array(
                    metric_data_dict[ensemble_name][waveband_name][variable_name]
                )
            
            # --- End of loop over variables
        # --- End of loop over wavebands
    # --- End of loop over ensemble names

    return metric_data_dict















'''
    FUNCTION TO PLOT METRIC BASED ON LOADED DATA
    
    Inputs:
    1) metric_data_dict -- Dictionary containing longitudinally averaged metrics
                           Layers: ensemble, waveband, variable
    2) mname -- metric name (variances, skewness, kurtosis, sw_pval)
    3) date_list -- list of datetime objects that correspond to dates of interest 

    Outputs a figure handle.

    Figure structure: 6 rows, 4 cols 
    Each column is a spatial scale
    Row 1: reference PS,     Row 2: Pert PS / Ref PS
    Row 3: reference T,      Row 4: Pert T / Ref T
    Row 5: reference Q,      Row 6: Pert Q / Ref Q
'''
def plot_multiscale_evolution_of_metric( metric_data_dict, mname, date_list, lat_list ):

    # Number of variables & wavebands
    num_variables = len( vinfo_dict.keys() )
    num_waveband = len( wbinfo_dict.keys() )


    # Init figure
    fig, axs = plt.subplots( nrows= num_variables*2, ncols= num_waveband, 
                             figsize=( 3 * num_waveband, 2 * num_variables * 2 ) )

    # Loop over all columns (i.e., wavebands)
    for icol, wbname in enumerate( ['data_raw', 'L_scale', 'M_scale', 'S_scale'] ):
        
        # Loop over variables considered
        for ivar, vname in enumerate( vinfo_dict.keys() ):

            # Generate plot for reference ens
            ax = axs[ivar*2, icol]
            ref_metric_arr = metric_data_dict['reference_ens'][wbname][vname]
            cnf = ax.contourf( 
                np.arange(len(date_list)), lat_list, ref_metric_arr.T, 11, 
                #date_list, lat_list, ref_metric_arr.T, 11, 
                cmap = 'inferno' )
            plt.colorbar(cnf, ax=ax, orientation='vertical')
            ax.set_title( 
                'Ref Ens %s %s' 
                % ( vinfo_dict[vname]['pretty'], wbinfo_dict[wbname])
            )
            
            # Generate ratio to compare perturbed and reference ensembles
            prt_metric_arr = metric_data_dict['perturbed_ens'][wbname][vname]
            ratio = prt_metric_arr / ref_metric_arr

            # Killing off small denominators
            if mname == 'p_values':
                ratio[ np.abs(ref_metric_arr) < 0.05  ] = np.nan
            else:
                ratio[ np.abs(ref_metric_arr) < 1e-9  ] = np.nan

            # Generate plot of relative differences based on the metrics
            ax = axs[ivar*2+1, icol]
            cnf = ax.contourf( 
                np.arange(len(date_list)), lat_list, ratio.T-1,
                # date_list, lat_list, ratio.T-1,
                np.linspace(-0.99,0.99,10),
                # norm = mcolors.LogNorm(), 
                cmap = 'RdBu_r', extend='both'
            )
            # cnf = ax.contourf( 
            #     date_list, lat_list, np.abs(ratio.T-1),
            #     np.power(10,np.linspace(-2,0,5)),
            #     norm = mcolors.LogNorm(), 
            #     cmap = 'viridis', extend='both'
            # )
            # hth = ax.contourf( 
            #     date_list, lat_list, ratio.T-1,
            #     [-10,0],
            #     hatches=['xx'],
            #     colors = 'none'
            # )
            plt.colorbar(cnf, ax=ax, orientation='vertical')
            ax.set_title( 
                'Rel Diff %s %s' 
                % ( vinfo_dict[vname]['pretty'], wbinfo_dict[wbname])
            )

        # --- end of loop over variables considered
    # --- end of loop over wavebands


    # Generate figure title
    plt.suptitle( 'Multiscale Evolution of %s' % minfo_dict[mname]['fullname'])
    plt.tight_layout()

    return fig











'''
    MAIN 'PROGRAM' WITHIN SCRIPT
'''

# Loop over all metric names
for mname in minfo_dict.keys():

    # Load metric data
    metric_data_dict = load_data_on_multiscale_evolution_of_metric(mname, date_list )
    print( 'loaded all data for %s' % mname)

    # Plot metric data
    fig = plot_multiscale_evolution_of_metric( metric_data_dict, mname, date_list, lat_list )
    print( 'plotted all data for %s' % mname)

    # Save figure
    plt.savefig( 'manuscript_fig_%s.pdf' % mname)
    plt.close()

    # Clearing memory
    gc_collect()


# --- end of loop over metrics
