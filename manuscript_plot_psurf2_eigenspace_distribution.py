
'''
    Load packages
'''
from datetime import datetime, timedelta
import numpy as np
from netCDF4 import Dataset as ncopen
from matplotlib import use as mpl_use
mpl_use('agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib import ticker
from gc import collect as gc_collect
import pickle
from scipy.stats import norm
from matplotlib.cm import get_cmap
from scipy.spatial import ConvexHull


'''
    User settings
'''
# Start and end dates
date_st = datetime(year=2011, month=1, day=1)
date_ed = datetime(year=2011, month=3, day=1)

# Time interval between simulation outputs
time_int = timedelta( days=1 )

# Path to SPEEDY reference ensemble and perturbed ensemble
path_reference_ens_dir = '/fs/ess/PAS2856/SPEEDY_ensemble_data/reference_ens/data_raw'
path_perturbed_ens_dir = '/fs/ess/PAS2856/SPEEDY_ensemble_data/perturbed_ens/data_raw'
path_dict = {'ref': path_reference_ens_dir, 'prt': path_perturbed_ens_dir}

# Typical date formatting
date_fmt = '%Y%m%d%H%M'

# Latitude to make plots for (~40 deg N)
lat_list = np.linspace(-87.21645, 87.21645, 48)
targ_lat_ind=34

# Targetted longitude (at roughly 180 deg E)
targ_lon_ind = 48


# Ensembel size
ens_size = 1000

# flag to do big data load
load_data_flag = False


'''
    Generate information about dates to plot
'''
# Generate list of date
date_list = []
date_nw = deepcopy( date_st )
while ( date_nw <= date_ed ):
    date_list.append(date_nw)
    date_nw += time_int
# --- End of loop over dates

# Number of dates
n_dates = len(date_list)










'''
    Loading coordiante arrays
'''
# Generate path to file
fname = date_st.strftime( '%s/%s.nc' % (path_reference_ens_dir, date_fmt) )

# Load file
ncfile = ncopen( fname, 'r' )

# # Extract first time point's ensemble mean
# ref_ps_state = np.mean( np.squeeze( ncfile.variables['ps'] ), axis=0 )

# Extract lat and lon for completeness
lat1d = np.squeeze( ncfile.variables['lat'] )
lon1d = np.squeeze( ncfile.variables['lon'] )

# Close file for hygiene
ncfile.close()

# Number of lat and lon points
nlon = len(lon1d)
nlat = len(lat1d)





'''
    Generate eigenvectors for reference ensemble at a specified date
'''
def compute_psurf_ref_eigenvec( date_nw = datetime(year=2011, month=2, day=13)  ):

    # LOAD REFERENCE ENSEMBLE
    # ----------------------
    ename = 'ref'

    # Generate path to file
    fname = date_nw.strftime( '%s/%s.nc' % (path_dict[ename], date_fmt) )

    # Load file
    ncfile = ncopen( fname, 'r' )

    # Load surface pressure
    ref_ens = np.squeeze(ncfile.variables['ps'])

    # Close file
    ncfile.close()


    # EIGENVECTOR COMPUTATION
    # ------------------------

    # Compute ensemble perturbations
    ref_ens_avg = np.mean( ref_ens, axis=0 )
    ref_ens_prt = np.empty( [nlon*nlat, ens_size] )   # Perturbations in matrix form
    for ee in range(ens_size):
        ref_ens_prt[:,ee] = (ref_ens[ee] - ref_ens_avg).flatten()
    ref_ens_prt = np.matrix(ref_ens_prt)

    # Compute SVD
    U, S, Vh = np.linalg.svd( ref_ens_prt, compute_uv=True)
    
    # Reshape leading eigenvectors
    ref_eigvec1 = np.array( U[:,0].reshape([nlat, nlon]) )
    ref_eigvec2 = np.array( U[:,1].reshape([nlat, nlon]) )
    ref_eigvec3 = np.array( U[:,3].reshape([nlat, nlon]) )

    return S, ref_eigvec1, ref_eigvec2, ref_eigvec3, ref_ens_avg





'''
    For a specified date, compute first 2 leading eigenvectors of the reference ensemble
    and plot out both ensembles
'''
def project_and_plot_psurf_eigenspace( date_nw, S1, S2, S3, ref_eigvec1, ref_eigvec2, ref_eigvec3, baseline_state ):

    # LOAD ENSEMBLE DATA
    # -------------------

    # Init data holder
    data_dict = {}

    for ename in ['ref','prt']:

        # Generate path to file
        fname = date_nw.strftime( '%s/%s.nc' % (path_dict[ename], date_fmt) )

        # Load file
        ncfile = ncopen( fname, 'r' )

        # Load surface pressure
        data_dict[ename] = np.squeeze(ncfile.variables['ps'])

        # Close file
        ncfile.close()
    
    # -- End of loop over ensemble types


    
    # PROJECT REFERENCE AND PERTURBED ENSEMBLE ONTO LEADING EIGENVECTORS
    # -----------------------------------------------------------------
    eig_dict = {}
    for ename in ['ref','prt']:
        
        eig_dict[ename] = {}

        # Subtract from reference ens avg b'cos looking at reference' space
        ens_offsets = data_dict[ename] - np.mean( data_dict['ref'], axis = 0) #baseline_state

        # Project!
        eig_dict[ename]['eigvec1 coeff'] = np.empty( ens_size )
        eig_dict[ename]['eigvec2 coeff'] = np.empty( ens_size )
        eig_dict[ename]['eigvec3 coeff'] = np.empty( ens_size )
        for ie in range(ens_size):
            eig_dict[ename]['eigvec1 coeff'][ie] = np.sum(ens_offsets[ie] * ref_eigvec1)
            eig_dict[ename]['eigvec2 coeff'][ie] = np.sum(ens_offsets[ie] * ref_eigvec2)
            eig_dict[ename]['eigvec3 coeff'][ie] = np.sum(ens_offsets[ie] * ref_eigvec3)


    # Save eigenvalues and eigenvectors
    eig_dict['eigvecs'] = [ref_eigvec1, ref_eigvec2]
    eig_dict['eigvals'] = np.sqrt([S1, S2, S3]) * np.sqrt(1000-1)
    eig_dict['ctr'] = baseline_state

    return eig_dict






'''
    COMPUTE PCAs
'''
# Compute leading eigenvectors and eigenvalues
S, ref_eigvec1, ref_eigvec2, ref_eigvec3, baseline_state = compute_psurf_ref_eigenvec()
S1, S2, S3 = S[:3]

# Dictionary to hold PCAs
pca_dict = {}
pca_dict[0] = [S1, ref_eigvec1, S1/np.sum(S)]
pca_dict[1] = [S2, ref_eigvec2, S2/np.sum(S)]
pca_dict[2] = [S3, ref_eigvec3, S3/np.sum(S)]






'''
    GENERATE SUBPLOTS OF THE LEADING THREE PCAS
'''

# Figure to hold PCAs and PCA projections at three dates
fig, axs = plt.subplots( nrows=4, ncols=3, figsize=(9,10) )

# Subplot to visualize the leading three PCAs
for i in range(3):

    # Singular vectors
    pca = deepcopy( pca_dict[i][1] )
    singular_frac = deepcopy( pca_dict[i][2] )

    pca /= pca.max()

    ax = axs[0,i]
    cnf = ax.contourf( 
        lon1d, lat1d, pca, np.linspace(-1,1,6),
        extend='both', cmap='RdBu_r'
    )
    cbar = fig.colorbar(cnf, ax=ax, orientation = 'horizontal', pad=0.3)
    ax.set_title('a%d) PC%d (frac: %5.3f)' % (i+1, i+1, singular_frac ), loc='left')
    ax.set_ylim([0,lat1d.max()])
    ax.set_xlabel('Lon (deg E)')
    ax.set_ylabel('')


# --- End of subplots to visualize leading PCAs

# cosmetics
axs[0,0].set_ylabel('Lat (deg N)')


# Inscribing PCA 2 & 3 as unfilled contours
axs[0,1].contour( 
    lon1d, lat1d, ref_eigvec3 / ref_eigvec3.max(),
    [-0.6, 0.6], colors='k'
)
axs[0,2].contour( 
    lon1d, lat1d, ref_eigvec2 / ref_eigvec2.max(),
    [-0.6, 0.6], colors='k'
)

plt.suptitle('Projections onto Surface Pressure Principal Components (PCs)', fontsize=16)









'''
    PLOT OUT STATISTICS GROWTH BEHAVIOR 
'''
for iperiod, day_st in enumerate([4,6,8]):
    # Linear growth date list
    date_list = [datetime(year=2011, month=2, day=day_st) + timedelta(days=iday) for iday in range(6) ]

    # Plotting combinations
    plot_combo_list = [[1,2], [2,3], [1,3]]

    # Dictionary to store trajectories for perturbed members
    traj_dict = {}
    for i in [1,2,3]:
        traj_dict['eigvec%d coeff' % i] = np.empty( [6, ens_size] )


    # Obtain eigenprojections and trajectories
    for idate, date_nw in enumerate(date_list):

        # Generate eigenprojections data
        eig_dict = project_and_plot_psurf_eigenspace( date_nw, S1, S2, S3, ref_eigvec1, ref_eigvec2, ref_eigvec3, baseline_state )

        # Store data on the eigenvector projections
        for i in [1,2,3]:
            traj_dict['eigvec%d coeff' % i][idate,:] = eig_dict['prt']['eigvec%d coeff' % i]
        
    # ---- End of eigenprojection

    # Note: eig_dict contains the coefficients from the final date (Jan 26) in the loop.


    # Plot the eigenprojections for linear phase
    for icomb, comb_pair in enumerate( plot_combo_list):

        # Select axis
        ax = axs[iperiod+1,icomb]

        # pairing
        key1 = 'eigvec%d coeff' % comb_pair[0]
        key2 = 'eigvec%d coeff' % comb_pair[1]

        # Scatter reference ensemble coefficients
        ax.scatter( 
            eig_dict['ref'][key1]/1000, eig_dict['ref'][key2]/1000,
            s = 10, c='lightgray', linewidth=0
        )

        # Plot convex polygon bounding data
        ref_arr = np.empty([ens_size,2])
        ref_arr[:,0] = eig_dict['ref'][key1]/1000
        ref_arr[:,1] = eig_dict['ref'][key2]/1000
        hull = ConvexHull( ref_arr )
        for simplex in hull.simplices:
            ax.plot(ref_arr[simplex, 0], ref_arr[simplex, 1], color='darkgray')

        # Scatter perturbed ensemble coefficients
        ax.scatter( 
            eig_dict['prt'][key1]/1000, eig_dict['prt'][key2]/1000,
            s = 10, alpha=0.2, c='r', linewidth=0
        )

        # "Comet" plot of 5 members
        idate_st = max( 0, idate-6 )
        for imem in range(5):
            ax.plot( 
                traj_dict[key1][idate_st:idate+1, imem]/1000, 
                traj_dict[key2][idate_st:idate+1, imem]/1000,
                color='k', linewidth=0.5
            )
            ax.scatter( 
                traj_dict[key1][idate, imem]/1000, 
                traj_dict[key2][idate, imem]/1000,
                s=2, c='k', marker='o'
            )
        

        # Cosmetics
        ax.set_title(
            '%s%d) PC%d VS PC%d ($t^*=$%4.2f)' % 
            (   
                ['b','c','d'][iperiod], 
                icomb+1, comb_pair[0], comb_pair[1], 
                ((date_list[-1] - datetime(year=2011,month=1,day=1)).total_seconds()/86400)/70
             ), 
            loc='left'
        )
        ax.set_xlabel('PC%d Coef. (kPa)' % comb_pair[0] )
        ax.set_ylabel('PC%d Coef. (kPa)' % comb_pair[1] )








plt.tight_layout()
plt.savefig('manuscript_fig_pca.pdf')

