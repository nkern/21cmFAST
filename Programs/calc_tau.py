"""
calc_tau.py
-----------

Calculate the average optical depth, tau, to the CMB
according to Liu et al. 2016 Phys. Rev. D 93, 043013
utilizing output files from 21cmFAST

Method:
See Equations (6) - (9) from Liu et al. 2016

The code below takes the following steps:
Note: z_min & z_max refer to the redshift range of 21cmFAST box outputs
0. Run 21cmFAST over any redshift range and save density and ionization boxes
1. Tau integral is from z=0 to z=30
2. For low-z
    if x_HII(z_min) > 0.999:
        we assume x_HII = 1.0
    else:
        we fit a quadratic and/or tanh function to x_HII at lowest 3 and/or 2 points in redshift
        and extrapolate down in redshift to x_HII(z)~0.999, below which we assume x_HII = 1.0
3. In the range z_min < z < z_max, we start using the full x_HII and delta_b box outputs from 21cmFAST
    to get spatially averaged <x_HII * (1+delta_b)> and use equations (6) - (9) in Liu et al. 2016
    The tau integral uses a higher redshift resolution than the box outputs of 21cmFAST, so we interpolate
    the value of <x_HII*(1+delta_b)> in between box outputs using a second-order polynomial to
    nearest neighbor points of 21cmFAST output redshifts.
4. For high-z we assume x_HII = 0.0 if x_HII(z_max) < 0.001 or
	we fit a tanh function to x_HII at the highest 2 points in redshift and extrapolate
    up in redshift to x_HII(z)~0.001, above which we assume x_HII = 0.0
5. We model Helium ionization as a step function with x_HeIII(z<3) = 1.0 and x_HeIII(z>3) = 0.0

Results:
1) ../tau.png image
2) ../tau_model.tab text file which is our model for tau and ionized fraction across redshift
3) ../x_HII.tab text file which contains 21cmFAST output of spatially averaged, density-weighted ionized fraction

Dependencies include:
- scipy

This script assumes a cosmological parameter file "param_vals.tab"
is present in the directory ../ and contains five LCDM parameters
sigma8, hlittle, OmBh2, OmCh2, ns
If this doesn't exist, it attempts to extract them from the COSMOLOGY.H file

Nicholas Kern
January, 2017
"""
# Import Modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mp
import numpy as np
import os
import argparse
import sys
import fnmatch
from scipy.optimize import leastsq
from scipy.integrate import trapz
from matplotlib import rc
rc('text', usetex=True)

# Parse options
parser = argparse.ArgumentParser(description='calculate tau')
parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='overwrite existing tau data if passed --overwrite') 
args = parser.parse_args()
overwrite = args.overwrite

if os.path.isfile('../tau_model.tab') == True and overwrite == False:
	print "Existing Tau data exists, not overwriting..."
	sys.exit(0)

# Define constants
sig_T	= 6.6524e-25	# Thomson Cross Section, cgs
G		= 6.674e-8		# Gravitational Const, cgs
c		= 2.99792e10	# Speed of Light, cgs
H0		= 3.24079e-18	# Hubble Constant, 100 km/sec/Mpc conveted to cgs (1/sec)
Yp		= 0.2467		# BBN Helium Number Density Fraction
m_H		= 1.0			# Hydrogen Atomic Weight
m_He	= 4.0			# Helium Atomic Weight
m_p		= 1.6726e-24	# Proton Mass, cgs
mu		= 1+(Yp/4.0)*(m_He/m_H - 1)	# Mean Molecular Weight

# Import Cosmological Parameters
try:
    sigma8, hlittle, OmBh2, OmCh2, ns = np.array(np.loadtxt('../param_vals.tab', dtype=str).T[1], float)[:5]
    H0 = H0*hlittle
    OmB = OmBh2 / hlittle**2
    OmC = OmCh2 / hlittle**2
    OmM = OmB + OmC
    OmL = 1.0 - OmM

except IOError:
    with open("../Parameter_files/COSMOLOGY.H","r") as f:
        lines = f.readlines()
    sigma8  = float(fnmatch.filter(lines, "*#define SIGMA8*")[0].split('(')[2].split(')')[0])
    hlittle = float(fnmatch.filter(lines, "*#define hlittle*")[0].split('(')[2].split(')')[0])
    OmB     = float(fnmatch.filter(lines, "*#define OMb*")[0].split('(')[2].split(')')[0])
    OmC     = float(fnmatch.filter(lines, "*#define OMc*")[0].split('(')[2].split(')')[0])
    ns      = float(fnmatch.filter(lines, "*#define POWER_INDEX*")[0].split('(')[2].split(')')[0]) 
    H0      = hlittle * H0
    OmM     = OmB + OmC
    OmL     = 1.0 - OmM

# Functions
def loadfile(fname, dtype=np.float32, shape=None):
	if shape is None:
		return np.fromfile(fname, dtype=dtype)
	else:
		return np.fromfile(fname, dtype=dtype).reshape(shape)

def dl_dz(z, H0=H0, OmM=OmM, OmL=OmL):
	"""
	Line-of-sight proper distance per unit redshift for a flat universe
	Eqn (9)
	"""
	return (c/H0)/((1+z)*np.sqrt(OmM*(1+z)**3 + OmL))

def nb_avg(z, H0=H0, OmB=OmB, mu=mu):
	"""
	Average Baryon Density
	Eqn (7)
	"""
	return 3*H0**2*OmB*(1+z)**3/(8*np.pi*G*mu*m_p)

def ne_avg(z, avg_x_HII_1_deltab, avg_x_HeIII_1_deltab, nb_avg=nb_avg, Yp=Yp):
	"""
	Average electron density
	Eqn (6)

	Input:
	------
	z : float
		redshift of universe

	avg_x_HII_1_deltab : float or ndarray (vector)
		spatial average of density weighted ionized fraction:
        avg[x_HII * (1+delta_b)]

    avg_x_HeIII_1_deltab : float or ndarray
        spatial average of density weighted helium-1 ionized fraction:
        avg[x_HeIII * (1+delta_b)]

	nb_avg : float
		Average baryon density

	Yp : float
		Helium Mass Fraction

	Output:
	-------
	ne_avg : float or vector

	Notes:
	------
	- ndarrays x_HII and delta_b must have same shape
	- Helium Reionization modeled as a step-function occuring at z_HeReion
	"""
	return nb_avg(z) * (avg_x_HII_1_deltab + 0.25*avg_x_HeIII_1_deltab*Yp)

# Load Box Data
rho_files	= np.array(map(lambda x: '../Boxes/'+x, sorted(fnmatch.filter(os.listdir('../Boxes'), 'updated_smoothed_deltax_z*'))))
ion_files	= np.array(map(lambda x: '../Boxes/'+x, sorted(fnmatch.filter(os.listdir('../Boxes'), 'xH_nohalos_z*'))))

# Extract redshifts
rho_z = np.array(map(lambda x: float(x.split('_')[3][1:]), rho_files))
ion_z = np.array(map(lambda x: float(x.split('_')[2][1:]), ion_files))

# Ensure they are equal in length, if not than N_ion will be greater than N_rho
N_rho = len(rho_z)
N_ion = len(ion_z)
if N_rho < N_ion:
    select = np.ones(N_rho, dtype=bool)
    for i in range(N_rho):
        if rho_z[i] not in ion_z:
            select[i] = False
    rho_files = rho_files[select]
    rho_z = rho_z[select] 

z = rho_z

# Calculate density weighted x_HII
avg_x_HII_1_deltab_output = []
for i in range(len(z)):
	avg_x_HII_1_deltab_output.append(np.mean((1+loadfile(rho_files[i]))*(1-loadfile(ion_files[i]))))
avg_x_HII_1_deltab_output = np.array(avg_x_HII_1_deltab_output)

# Get z_axis for tau integration
z_arr = np.arange(0,29.99,0.05)
dz = z_arr[1]-z_arr[0]

# Assign <x_HeIII(1+deltab)> array
avg_x_HeIII_1_deltab = np.zeros(len(z_arr))
avg_x_HeIII_1_deltab[np.where(z_arr<=3.0)] = 1.0

# Assign <x_HII(1+deltab)> array
avg_x_HII_1_deltab = np.zeros(len(z_arr))

# Define model and residual
model = lambda x, a, b: -0.5*np.tanh(a*(x-b))+0.5
def residual(coeffs, xdata, ydata, model):
	a = coeffs[0]
	b = coeffs[1]
	return (model(xdata,a,b)-ydata)**2

def perform_leastsq(residual, coeffs, args=None):
	"""
	perform least squares but jitter coeffs if result is not well behaved
	"""
	adder = np.array([0.2, 2])
	for i in range(-2,3):
		for j in range(-2,3):
			coeffs2 = coeffs + np.array([adder[0]*i, adder[1]*j])
			out = leastsq(residual, coeffs2, args=(xdata, ydata, model))[0]
			if out[0] > 0 and out[1] > 0: return out
	print "leastsq did not converge!"
	return out

# Check lowest redshift for x_HII
if avg_x_HII_1_deltab_output[0] <= 0.999:
    # If avg_x_HII(z_min) < 0.85 fit a quadratic to 0.85
    if avg_x_HII_1_deltab_output[0] < 0.85:
        # Fit quadratic
        fit = np.polyfit(z[:3], avg_x_HII_1_deltab_output[:3], deg=2)
        # Generate model
        model_y = z_arr**2 * fit[0] + z_arr * fit[1] + fit[2]
        # Find when model = 0.85
        z_low = z_arr[np.where(np.abs(model_y[z_arr<z[0]]-0.85)==np.abs(model_y[z_arr<z[0]]-0.85).min())[0][0]]
        # Assign z_low < z < z_min to model
        intermediate = np.where((z_arr > z_low) & (z_arr <= z[0]))
        avg_x_HII_1_deltab[intermediate] = z_arr[intermediate]**2 * fit[0] + z_arr[intermediate] * fit[1] + fit[2]

        # Extrapolate curve down to z=0 by fitting tanh
        # Perform least squares
        coeffs = np.array([0.5,8.0])
        xdata = np.copy(z_arr[intermediate][:2])
        ydata = np.copy(avg_x_HII_1_deltab[intermediate][:2])
        out = perform_leastsq(residual, 1*coeffs, args=(xdata, ydata, model))

        # Locate when avg_x_HII_1_deltab(z) == 0.999
        model_y = model(z_arr, out[0], out[1])
        z_low = z_arr[np.where(np.abs(model_y-0.999)==np.abs(model_y-0.999).min())[0][0]]

        # Assign to 1.0
        avg_x_HII_1_deltab[np.where(z_arr <= z_low)] = 1.0

        # Assign z_low < z < z_min to model
        intermediate = np.where((z_arr>z_low)&(z_arr<=z_arr[intermediate][0]))
        avg_x_HII_1_deltab[intermediate] = model(z_arr[intermediate], out[0], out[1])

    else:
        # Extrapolate curve down to z=0 by fitting tanh
        # Perform least squares
        coeffs = np.array([0.5,8.0])
        xdata = np.copy(z[:2])
        ydata = np.copy(avg_x_HII_1_deltab_output[:2])
        out = perform_leastsq(residual, 1*coeffs, args=(xdata, ydata, model))

        # Locate when avg_x_HII_1_deltab(z) == 0.999
        model_y = model(z_arr, out[0], out[1])
        z_low = z_arr[np.where(np.abs(model_y-0.999)==np.abs(model_y-0.999).min())[0][0]]

        # Assign to 1.0
        avg_x_HII_1_deltab[np.where(z_arr<=z_low)] = 1.0

        # Assign z_low < z < z_min to model
        intermediate = np.where((z_arr>z_low)&(z_arr<=z[0]))
        avg_x_HII_1_deltab[intermediate] = model(z_arr[intermediate], out[0], out[1])

else:
	avg_x_HII_1_deltab[np.where(z_arr<=z[0])] = 1.0

# Check high redshift end for x_HII
if avg_x_HII_1_deltab_output[-1] >= 0.001:
	coeffs = np.array([0.5, 8.0])
	xdata = np.copy(z[-2:])
	ydata = np.copy(avg_x_HII_1_deltab_output[-2:])
	out = perform_leastsq(residual, 1*coeffs, args=(xdata, ydata, model))

	# Locate when avg_x_HII_1_deltab(z) == 0.999
	model_y = model(z_arr, out[0], out[1])
	z_high = z_arr[np.where(np.abs(model_y-0.001)==np.abs(model_y-0.001).min())[0][0]]

	# Assign to 0.0
	avg_x_HII_1_deltab[np.where(z_arr>z_high)] = 0.0

	# Assign z_max < z < z_high to model
	intermediate = np.where((z_arr<=z_high)&(z_arr>z[-1]))
	avg_x_HII_1_deltab[intermediate] = model(z_arr[intermediate], out[0], out[1])

else:
    avg_x_HII_1_deltab[np.where(z_arr>z[-1])] = 0.0

# Assign intermediate redshifts by interpolating 21cmFAST data using 2nd order polynomial
avg_x_HII_1_deltab[np.where((z_arr>z[0])&(z_arr<=z[-1]))] = np.interp(z_arr[np.where((z_arr>z[0])&(z_arr<=z[-1]))], z, avg_x_HII_1_deltab_output)

# Remove bad predictions
avg_x_HII_1_deltab[np.where(avg_x_HII_1_deltab<0)] = 0.0
avg_x_HII_1_deltab[np.where(avg_x_HII_1_deltab>1)] = 1.0

# Perform Tau Integration
tau = np.zeros(len(z_arr))
avg_ne = ne_avg(z_arr, avg_x_HII_1_deltab, avg_x_HeIII_1_deltab)
dldz = dl_dz(z_arr)
for i in range(1,len(z_arr)):
    tau[i] = sig_T * trapz(avg_ne[:i]*dldz[:i], x=z_arr[:i], dx=dz)

# Interpolate back to z
tau_z = np.interp(z, z_arr, tau)

# Write to file
np.savetxt('../x_HII.tab', np.vstack([z, avg_x_HII_1_deltab_output]).T, fmt='%8.5f', delimiter='\t',
            header='z\t <x_HII*(1+delta_b)>')

# write high-z resolution tau_model
np.savetxt('../tau_model.tab', np.vstack([z_arr, tau, avg_x_HII_1_deltab]).T, fmt='%8.5f', delimiter='\t', header='z\t tau\t <x_HII*(1+delta_b)>')

# Plot
fig=mp.figure(figsize=(5,5))
fig.subplots_adjust(hspace=0.1)

ax1 = fig.add_subplot(211)
ax1.set_xlim(0,35)
ax1.set_ylim(-0.1,1.1)
mp.setp(ax1.get_xticklabels(), visible=False)
ax1.set_ylabel(r'$\langle(x_{HII}(1+\delta_{b})\rangle$',fontsize=18)
ax1.grid(True)
ax1.plot(z_arr,avg_x_HII_1_deltab,'k-', linewidth=1.5, alpha=0.75)
ax1.scatter(z, avg_x_HII_1_deltab_output, edgecolor='', c='dodgerblue', s=25, alpha=0.75)

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.set_xlim(0,35)
ax2.set_ylim(0,0.12)
ax2.set_xlabel(r'$z$',fontsize=18)
ax2.set_ylabel(r'$\tau(z)$', fontsize=18)
ax2.grid(True)
ax2.plot(z_arr, tau, 'g', linewidth=2, alpha=0.75)
ax2.annotate(r'$\tau=%.04f$'%tau[-1], fontsize=18, xy=(0.05,0.8),
                xycoords='axes fraction', bbox=dict(boxstyle='round', fc='w', alpha=0.8))

fig.savefig('../tau.png', dpi=200, bbox_inches='tight')
mp.close()

