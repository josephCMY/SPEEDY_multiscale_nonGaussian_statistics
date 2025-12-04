'''
    SCRIPT TO GENERATE PLOTS FOR GRAPHICAL ABSTRACT
    Variables examined: ps

    Written by: Joseph Chan
'''

'''
    IMPORT PACKAGES
'''
import numpy as np
import pickle as pkl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates 
from copy import deepcopy
from gc import collect as gc_collect
from scipy.stats import kstest
from os.path import isfile


'''
    USER SETTINGS
'''
# Dictionary containing some meta data about variables examined
vinfo_dict = {
    'ps': {'pretty': 'PSFC', 'unit': 'hPa'},
    # 't':  {'pretty': 'TEMP', 'unit': 'K'},
    # 'q':  {'pretty': 'QVAP', 'unit': 'kg/kg'},
}

# Dictionary containing some meta data about waveband examined
wbinfo_dict = {
    "data_raw": 'Unfiltered', 
    # "L_scale":  'L Scale',
    # "M_scale":  'M Scale',
    # "S_scale":  'S Scale'
}

# Dictionary containing meta data about ensembles examiend
ensinfo_dict = {
    "reference_ens": 'Ref.', 
    "perturbed_ens": 'Prt.'
}

# Information about where KS test data is stored
pkl_rootpath='/fs/scratch/PAS2635/chan1063/SPEEDY_Multiscale_NonGaussianity/diagnostics/jchan/ks_tests'

# Dictionary containing information about the KS test stiatistics examined
# ENAME will be overwritten with ensemble name, VNAME will be overwritten with variable name, etc.
control_dict = {
    'ks': {
        'filepath': pkl_rootpath+"/WNAME/VNAME/perturbed_ens_VNAME_WNAME_DATE.pkl",
        'fullname': 'KS Test'
    }
}


# Dictionary containing meta data about metrics examined
# ENAME will be overwritten with ensemble name, VNAME will be overwritten with variable name, etc.
minfo_dict = {
    'variances': {
        'filepath': "diagnostic_pkl_files/diagnostics_max/ENAME/WNAME/VNAME/VNAME_ENAME_DATE_variances.pkl",
        'sq flag': False,
        'fullname': 'MSS',
        'rms flag': False
    },
    # 'skewness': {
    #     'filepath': "diagnostic_pkl_files/diagnostics_aiden/ENAME/WNAME/VNAME/VNAME_ENAME_WNAME_DATE_skewness.pkl",
    #     'sq flag': True,
    #     'fullname': 'MSW',
    #     'rms flag': False
    # },
    # 'kurtosis': {
    #     'filepath': "diagnostic_pkl_files/diagnostics_aiden/ENAME/WNAME/VNAME/VNAME_ENAME_WNAME_DATE_kurtosis.pkl",
    #     'sq flag': True,
    #     'fullname': 'MSK',
    #     'rms flag': False
    # },
    'p_values':{
        'filepath': "diagnostic_pkl_files/diagnostics_max/ENAME/WNAME/VNAME/VNAME_ENAME_DATE_p_values.pkl",
        'sq flag': False,
        'threshold': 0.05,
        'fullname': '$f_{NG}$',
        'rms flag': False
    }
}


# List of latitudes
lat_list = np.linspace(-87.21645, 87.21645, 48)


# List of all dates
date_st = datetime( year=2011, month=1, day=1 )
date_ed = datetime( year=2011, month=12, day=31 )
date_interval = timedelta( days=1 )
date_list = []
date_nw = deepcopy( date_st )
while date_nw <= date_ed:
    date_list.append(date_nw)
    date_nw += date_interval



# Weights used for equal area averaging
area_weights = np.zeros( (96,48), dtype='f8' )
# area_weights += np.sin( np.arange(48) * 3.1415 / 48)
area_weights += np.cos( lat_list * 3.1415 / 180)
area_weights = area_weights.T
area_norm = np.mean( area_weights )




'''
    FUNCTION TO LOAD KS TEST METRICS AND VARIANCES COMPARING REFERENCE AND PERTURBED


    Inputs:
    1) date_list -- list of datetime objects that correspond to dates of interest

    Output:
    1) conv_dict -- Dictionary containing measures of convergence
'''
def load_data_on_multiscale_statistical_convergence( date_list, lat_list, pval=0.05 ):

    # Dictionary to hold convergence stuff data after loading.
    conv_dict = {}
    conv_dict['ks'] = {}
    conv_dict['variances'] = {}

    # number of dates
    num_dates = len( date_list )

    # ----- Load KS data
    mname='ks'
    for waveband_name in wbinfo_dict.keys():
        conv_dict[mname][waveband_name] = {}
        for variable_name in vinfo_dict.keys():
            # Init temporary list to hold metric
            conv_dict[mname][waveband_name][variable_name] = [None for dd in range(num_dates)]
            # Loop over all dates
            for dd, date_nw in enumerate( date_list ):

                # Generate pkl file path for
                fname = deepcopy( control_dict[mname]['filepath'] )
                fname = fname.replace( 'VNAME', variable_name )
                fname = fname.replace( 'WNAME', waveband_name )
                fname = fname.replace( 'DATE', date_nw.strftime('%Y%m%d%H%M') )

                # Load metric from pickle file
                if isfile( fname ):
                    with open( fname, 'rb' ) as fhandle:
                        metric_arr = (pkl.load( fhandle ))['ks pval'] > pval
                else:
                    metric_arr = np.zeros( (48,96) ) + np.nan

                # Kill vertical dimension, if present
                if len( metric_arr.shape ) == 3 and metric_arr.shape[0] == 8:
                    metric_arr = metric_arr[-1]  # Select only bottommost dimension
                elif len( metric_arr.shape ) > 2:
                    print('Error: loaded metric data has too many dimensions')
                    print( variable_name, waveband_name, date_nw, 'ks' )
                    print( metric_arr.shape )
                    quit()
                # --- end of vertical dimension slaughtering.

                # Evaluate global equal-area average
                conv_dict[mname][waveband_name][variable_name][dd] = (
                    np.mean( metric_arr*area_weights ) / area_norm
                )

            # --- End of loop over dates

            # Convert data list into array
            conv_dict[mname][waveband_name][variable_name] = np.array(
                conv_dict[mname][waveband_name][variable_name]
            )
            
        # --- End of loop over variables
    # --- End of loop over wavebands
    # ----- Finished loading KS dat


    return conv_dict
    














'''
    FUNCTION TO LOAD DATA FOR A SPECIFIC METRIC

    Inputs:
    1) mname -- metric name (variances, skewness, kurtosis, sw_pval)
    2) date_list -- list of datetime objects that correspond to dates of interest

    Output:
    1) metric_data_dict -- Dictionary containing longitudinally averaged metrics
                      Layers: ensemble, waveband, variable
'''
def load_data_on_multiscale_evolution_of_metric( mname, date_list,lat_list ):

    # Dictionary to hold metric data after loading.
    metric_data_dict = {}

    # number of dates
    num_dates = len( date_list )

    # Loop over all combinations of ensemble, waveband, and variable
    for waveband_name in wbinfo_dict.keys():
        metric_data_dict[waveband_name] = {}

        for variable_name in vinfo_dict.keys():
            metric_data_dict[waveband_name][variable_name] = {}
                
            for ensemble_name in ensinfo_dict.keys():
                metric_data_dict[waveband_name][variable_name][ensemble_name] = {}
    
                # Init temporary list to hold metric
                metric_data_dict[waveband_name][variable_name][ensemble_name] = [None for dd in range(num_dates)]

                # Loop over all dates
                for dd, date_nw in enumerate( date_list ):

                    # Generate pkl file path
                    fname = deepcopy( minfo_dict[mname]['filepath'] )
                    fname = fname.replace( 'ENAME', ensemble_name )
                    fname = fname.replace( 'VNAME', variable_name )
                    fname = fname.replace( 'WNAME', waveband_name )
                    fname = fname.replace( 'DATE', date_nw.strftime('%Y%m%d%H%M') )

                    # Load metric from pickle file
                    if isfile( fname ):
                        with open( fname, 'rb' ) as fhandle:
                            metric_arr = (pkl.load( fhandle ))[mname]
                    else: 
                        metric_arr = np.zeros( (48,96) ) + np.nan

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
                    metric_arr = np.mean( metric_arr * area_weights) / area_norm

                    # Is this an MS variable?
                    if minfo_dict[mname]['rms flag']:
                        metric_arr = np.sqrt( metric_arr )

                    metric_data_dict[waveband_name][variable_name][ensemble_name][dd] = metric_arr
                
                # --- End of loop over dates

                # Convert data list into array
                metric_data_dict[waveband_name][variable_name][ensemble_name] = np.array(
                    metric_data_dict[waveband_name][variable_name][ensemble_name]
                )
            
            # --- End of loop over variables
        # --- End of loop over wavebands
    # --- End of loop over ensemble names

    return metric_data_dict















'''
    FUNCTION TO VISUALIZE FIGURE FOR GRAPHICAL ABSTRACT
    
    Inputs:
    1) conv_dict -- Dictionary containing longitudinally averaged metrics
                           Layers: sim_frac/meta, waveband, variable
    2) date_list -- list of datetime objects that correspond to dates of interest 

    Outputs a figure handle.

'''
def plot_graphical_abstract( conv_dict, date_list ):

    # Number of variables & wavebands
    num_variables = len( vinfo_dict.keys() )
    num_waveband = len( wbinfo_dict.keys() )

    curve_control_dict = {
        'ks':           { 'color': 'r'          , 'linestyle':'-' },
        'variances':    { 'color': 'k'          , 'linestyle':'-' },
        # 'skewness':     { 'color': 'dodgerblue' , 'linestyle':'-' },
        # 'kurtosis':     { 'color': 'r'          , 'linestyle':'--'},
        'p_values':     { 'color': 'dodgerblue'          , 'linestyle':'-'}
    }

    ndates = len(date_list)

    # Init figure
    fig, axs = plt.subplots( nrows=2, ncols=1 , figsize=(5.5,5.5 ) )
    # axs = [axs]
        
    # Loop over variables considered
    # for ivar, vname in ['ps']: #enumerate( vinfo_dict.keys() ):
    vname = 'ps'

    # Loop over all columns (i.e., wavebands)
    for iwb, wbname in enumerate( ['data_raw']): 

        # Plot similarity fractions
        ax = axs[0]
        ax.plot( np.arange(ndates)/70, conv_dict['ks'][wbname][vname], #+1e-6, #date_list, conv_dict['ks'][wbname][vname]+1e-6, 
                 label = 'Sim. Frac.', color=curve_control_dict['ks']['color'],
                 linestyle=curve_control_dict['ks']['linestyle']
        )
        ax.set_yscale('log')
        ax.set_ylim([1e-2,2])
        ax.set_title( 
            'a) Normalized Variance & Similarity Fraction', # \n      (%s)' % (iwb+1, wbinfo_dict[wbname]), #vinfo_dict[vname]['pretty']),
            loc='left', fontsize=14
        )
        # if iwb == 0:
        #     ax.axhline(0.95, color = 'r', zorder=0, linewidth=0.5, linestyle='--', label = 'y = 0.95') 
        # else:
        #     ax.axhline(0.95, color = 'r', zorder=0, linewidth=0.5, linestyle='--')

        # Plot MSS ratios
        mname = 'variances'
        ratio = (
                conv_dict[mname][wbname][vname]['perturbed_ens'] 
                / conv_dict[mname][wbname][vname]['reference_ens'] 
        )
        ax.plot( np.arange(ndates)/70, ratio, 
                label = 'Norm. Var.', 
                color=curve_control_dict[mname]['color'],
                linestyle=curve_control_dict[mname]['linestyle']
            )
        
        ax.axhline(1, color = 'k', zorder=0, linewidth=0.5, linestyle='--') 
        
        # ax.set_yscale('log')
        # ax.set_ylim([1e-4,2])

        # Plot SW p-value
        ax = axs[1]
        for mname in ['p_values']:
            ratio = (
                conv_dict[mname][wbname][vname]['perturbed_ens'] 
                / conv_dict[mname][wbname][vname]['reference_ens'] 
            )
            ratio[ratio > 50] = np.nan
            ax.plot( np.arange(ndates-1)/70, ratio[1:],   #date_list[1:], ratio[1:], 
                    label = 'Non-Gauss. Frac.', 
                    color=curve_control_dict[mname]['color'],
                    linestyle=curve_control_dict[mname]['linestyle']
                )
        if iwb == 0:
            ax.axhline(1, color = 'k', zorder=0, linewidth=0.5, linestyle='--', label = 'y = 1.00') 
        else:
            ax.axhline(1, color = 'k', zorder=0, linewidth=0.5, linestyle='--')

        ax.set_ylim([0,1.5])
        ax.set_title( 
            'b) Non-Gaussian Areal Fraction', # % (iwb+1, wbinfo_dict[wbname]), #vinfo_dict[vname]['pretty']),
            loc='left', fontsize=14
        )

    # --- End of loop over wavebands


    # Prettifying
    for ax in axs:
        ax.set_xlim([0,3])
        ax.set_xlabel('Normalized Model Time', fontsize=10)

    # Indicate saturation times
    axs[0].axvline( 1.6, color='r', linestyle='--', zorder=0)
    axs[0].axvline( 1, color='k',  linestyle='--', zorder=0)
    axs[1].axvline( 1.6, color='r', linestyle='--', zorder=0, label='Equilibration Time')
    axs[1].axvline( 1, color='k',  linestyle='--', zorder=0, label='Var. Sat. Time')

    # Generate space for figure legend
    fig.subplots_adjust(hspace=0.5, bottom=0.2, top=0.95, left=0.1, right=0.95 )

    # Make plot labels
    handles1, labels1 = axs[0].get_legend_handles_labels()
    handles2, labels2 = axs[1].get_legend_handles_labels()
    handles = handles1+handles2
    labels = labels1+labels2
    fig.legend(handles, labels, loc='lower center', ncols=3, fontsize=10)
        

    return fig

































# Loading data into dictionary
# conv_dict = load_data_on_multiscale_statistical_convergence( date_list, lat_list, pval=0.05 )
# for mname in minfo_dict.keys():
#     conv_dict[mname] = load_data_on_multiscale_evolution_of_metric( mname, date_list, lat_list )
# with open('convergence_plot_info.pkl', 'wb') as f:
#     pkl.dump(conv_dict, f)

with open('convergence_plot_info.pkl', 'rb') as f:
    conv_dict = pkl.load(f)

fig = plot_graphical_abstract( conv_dict, date_list )
plt.savefig( 'manuscript_graphical_abstract.pdf')
# plt.savefig( 'manuscript_graphical_abstract.png', dpi=300)
plt.close()


# fig = plot_multiscale_equilibrium( conv_dict, date_list )
# plt.savefig( 'manuscript_fig_equlibrium.pdf')
# plt.close()

# fig = plot_multiscale_evolution_of_average_nonGauss( conv_dict, date_list )
# plt.savefig('manuscript_fig_ratio_of_ratios.pdf')
# plt.close()



# # Average similarity fractions
# date_flags = np.array(date_list) >= datetime( year=2011, month=4, day=20)
# for wbname in ['data_raw', 'L_scale', 'M_scale', 'S_scale']:
#     avg = np.nanmean( conv_dict['ks'][wbname]['ps'][date_flags] )
#     print( 'Phase III similarity fraction for %s is %f' % (wbname, avg))

# '''
# Print statement outcome:
#     Phase III similarity fraction for data_raw is 0.953225
#     Phase III similarity fraction for L_scale is 0.952922
#     Phase III similarity fraction for M_scale is 0.952450
#     Phase III similarity fraction for S_scale is 0.946940
# '''