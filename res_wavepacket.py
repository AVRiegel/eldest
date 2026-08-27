#!/usr/bin/python3
# This script takes the projections of the wavefunction on the vibronic resonance states as input
# and yields the expanded wavefunctions as a function of R and t in these levels
# as well as the combined wavepacket in the resonance state as output.
# Alexander Riegel, 2024/2025.

import argparse
from contextlib import contextmanager
import numpy as np
import pandas as pd
import os
from pathlib import Path
from pygnuplot import gnuplot   # Module is py-gnuplot
from scipy.integrate import romb, simpson, trapezoid
import subprocess
import sys
sys.path.append('/mnt/home/alexander/eldest')
import warnings
warnings.filterwarnings(action='ignore', category=np.exceptions.ComplexWarning)

import in_out
import sciconv
import wellenfkt as wf

@contextmanager
def silence_print():
    with open(os.devnull, 'w') as dummyout:
        old_stdout = sys.stdout
        sys.stdout = dummyout
        try:
            yield
        finally:
            sys.stdout = old_stdout


# set up argument parser
parser = argparse.ArgumentParser(
        description='''This script calculates the wavepacket in the resonance state
        from the projections onto the vibronic resonance states
        that are the output of res_nuclear_dyn.py (in wp_res.dat).''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Alexander V. Riegel, 2024.')
parser.add_argument('-w', '--wavepacket_infile', default='wp_res.dat',
                    help='File which contains the projections of the wavefunction on the vibronic resonance states.')
parser.add_argument('-s', '--settings_infile', default='photonucl.in',
                    help='File which includes the simulation settings, potential parameters etc.')
parser.add_argument('-r', '--r_lims', nargs=3, default=[5.8, 6.8, 7], metavar=('R_low', 'R_high', 'R_num_exp'),
                    help='Lower and upper limit for R (in a.u.) as well as exponent for number of points with num = 2**R_num_exp + 1.')
parser.add_argument('-t', '--time_lims', nargs=2, default=[-1.2, 100.], metavar=('t_low', 't_up'),
                    help='Iterable with lower and upper limit for time t (in fs).')
parser.add_argument('-l', '--lambda_indiv', default=False,
                    help='Calculate "populations" also for individual vibrational resonance levels.')
args = parser.parse_args()


# read in potentials and set infile/outfile
infile = args.wavepacket_infile
settings = args.settings_infile

if Path(infile).is_file():
    print('Input file for resonance-state projections:', infile)
else:
    sys.exit('Input file for resonance-state projections "%s" does not exist.' % infile)

if Path(settings).is_file():
    print('Input file for simulation and potential settings:', settings)
else:
    sys.exit('Input file for simulation and potential settings "%s" does not exist.' % settings)

with open(os.devnull, 'w') as dummyfile, silence_print():
    (X_ICD, X_RICD, _, _,
     _, _, _, _, _, _, _, _, _,
     Omega_eV, n_X, _, _, X_gauss, _,
     _, _, _, _, _, _, _, _, _,
     _, _, _, Ep_step_eV, _, _, Ep_min_eV, Ep_max_eV,
     _, _, _, _, _, _, _,
     mass1, mass2, _, _,
     _, _, _, _,
     res_a, res_b, res_c, _, res_pot_type,
     _, _, _, _, _
     ) = in_out.read_input(settings, dummyfile)

outfile=f'wf_{infile}'


# prepare array with R values and set time limits
R_low, R_high, R_num_exp = [float(num) for num in args.r_lims]
if (R_num_exp.is_integer() and R_num_exp > 0):
    R_len = 2**int(R_num_exp) + 1
else:
    sys.exit('R_num_exp must be positive integer.')

R_arr = np.linspace(R_low, R_high, R_len)

t_low, t_high = [float(num) for num in args.time_lims]

# choose E_p value
if X_ICD:   # Use centre E_p value
    num_p = int(((Ep_max_eV-Ep_min_eV)//Ep_step_eV)/2)
elif X_RICD:
    num_p = int(0)


##########

with open(infile,'r') as f:
    raw_data = pd.read_csv(f, header=None, sep=r'\s+', engine='python')

# Select only rows with chosen E_p value, then drop E_p column and renumber columns
Ep_eV = raw_data[1].iloc[num_p]
print(f'num_p = {num_p} <=> E_p = {Ep_eV:>8.5f} eV')
data = raw_data.loc[raw_data[1] == Ep_eV].reset_index(drop=True)
data.drop(1,axis=1,inplace=True)
data.columns = range(data.columns.size)

# Sort input by time and quantum number, repeat each line R_len times (each t and lambda evaluated at each R)
data[2] = data[2].astype(complex)
if np.any(np.isnan(data)) or np.any(np.isinf(data)):
    print('!!! Beware, there is at least one invalid number in the input, which will hopefully be filtered out.')
mata = data.sort_values([0,1]).reset_index(drop=True)
N_lambda = int(mata[0].iloc[-1] + 1)
sata = mata.loc[data.index.repeat(R_len)]
sata[3] = np.tile(R_arr, len(data[0]))

# Potential
red_mass = wf.red_mass_au(mass1,mass2)
if (res_pot_type == 'morse'):
    De    = res_a
    alpha = res_b
    Req   = res_c
    factors_res = [1] * N_lambda
elif (res_pot_type == 'hyperbel'):
    res_hyp_a       = res_a
    res_hyp_b       = res_b
    R_hyp_step_res  = res_c
    if(X_gauss):
        Omega_au  = sciconv.ev_to_hartree(Omega_eV)
        sigma     = np.pi * n_X / (Omega_au * np.sqrt(np.log(2)))
        sigma_E   = 1. / sigma
        width_E   = 5 * sigma_E
        EX_max_au = Omega_au + 0.5 * width_E
    R_start_EX_max_res = res_hyp_a / (EX_max_au - res_hyp_b)
    R_starts = np.array([R_start_EX_max_res+n*R_hyp_step_res for n in range(N_lambda)][::-1])
    factors_res = R_hyp_step_res * res_hyp_a / R_starts**2
else:
    sys.exit("The type of resonance-state potential is not supported.")

# At each point, multiply the WF of the vibrational state with the projection of the total WF on it
sata[4] = complex(0)
for n in range(N_lambda):
    s_down, s_up = n*len(data[0])//N_lambda, (n+1)*len(data[0])//N_lambda-1     # select block with the current quantum number n
    if (res_pot_type == 'morse'):
        wavefunction = np.vectorize(wf.mp_psi_n)(sata.loc[s_down:s_up,3], int(data[0][n]), alpha, Req, red_mass, De).astype(complex)
    elif (res_pot_type == 'hyperbel'):
        wavefunction = np.vectorize(wf.mp_psi_hyp)(sata.loc[s_down:s_up,3], res_hyp_a, res_hyp_b, red_mass, R_starts[n]).astype(complex)
    sata.loc[s_down:s_up,4] = sata.loc[s_down:s_up,2] * wavefunction

# Prepare an additional block for the whole resonance wavepacket, indicate by quantum number -1
fata = pd.concat((sata, sata[sata[0] == 0].set_index(sata[sata[0] == 0].index + sata.index[-1] + 1)))
fata.loc[len(data[0]):,0] = -1  # loc with data[0] is equivalent to iloc with sata[0]

# Add up the contributions at each R,t point to the whole resonance wavepacket at this point
aata = np.array(fata[[3,1,0,4]])
l_block_len = len(sata[0])//N_lambda
for x in np.arange(l_block_len):
    aata[N_lambda*l_block_len + x][3] = np.ma.masked_invalid([aata[n*l_block_len + x][3] * factors_res[n] for n in range(N_lambda)]).sum()

# Write out the array
pata = np.array((abs(aata[:,3])**2,)).T
oata = np.hstack((aata[:,:3].astype(float), pata))
np.savetxt(outfile, oata, delimiter='   ', fmt=['%10.7f', '% .7e', '% i', '% .15e'])


##########

# Extract the total resonance-state wavepacket, restructure the file for pm3d and calc population & R expectation value
outfile_pm3d=f'pm3d_{outfile}'
eata = oata[-l_block_len:]
np.savetxt(outfile_pm3d, eata, delimiter='   ', fmt=['%10.7f', '% .7e', '% i', '% .15e'])
subprocess.call(['sed', '-i', f'/^[[:space:]]*{f"{R_high:.7f}"}/G', outfile_pm3d])

popfile=f'pop_{infile}'
pop = pd.DataFrame() 
for t in range(len(eata)//len(R_arr)):
    pop.loc[t, 0] = mata[1][t]
    pop.loc[t, 1] = simpson(eata[t*len(R_arr):(t+1)*len(R_arr)][:,3],dx=R_arr[1]-R_arr[0])
np.savetxt(popfile, pop, delimiter='   ', fmt=['% .7e', '% .15e'])

if args.lambda_indiv:
    for n in range(N_lambda):
        eata_sub = oata[n*l_block_len:(n+1)*l_block_len]
        popfile_sub=f'pop_{n}_{infile}'
        pop_sub = pd.DataFrame() 
        for t in range(len(eata_sub)//len(R_arr)):
            pop_sub.loc[t, 0] = mata[1][t]
            pop_sub.loc[t, 1] = simpson(eata_sub[t*len(R_arr):(t+1)*len(R_arr)][:,3],dx=R_arr[1]-R_arr[0])
        np.savetxt(popfile_sub, pop_sub, delimiter='   ', fmt=['% .7e', '% .15e'])


expectfile=f'expect-R_{infile}'
expect = pd.DataFrame()
#expect.loc[0, 0] = mata[1][0]
#expect.loc[0, 1] = 0.
for t in range(1,len(eata)//len(R_arr)):
    expect.loc[t-1, 0] = mata[1][t]
    expect.loc[t-1, 1] = simpson(eata[t*len(R_arr):(t+1)*len(R_arr)][:,0]*eata[t*len(R_arr):(t+1)*len(R_arr)][:,3],dx=R_arr[1]-R_arr[0])/pop[1][t]
np.savetxt(expectfile, expect, delimiter='   ', fmt=['% .7e', '% .15e'])


# Plot to eps
g = gnuplot.Gnuplot()
g.set(terminal = "postscript enhanced color size 30cm,15cm font 'Helvetica,26' lw 4",
      output = "'gp_outfile.eps'",
      bmargin = "0.5",
      lmargin = "5.0",
      rmargin = "-5.0",
      cbtics = "font ',20'",
      xlabel = "'R (a.u.)'",
      ylabel = "'t (fs)'",
      zlabel = "'P (a.u.)'",
      xrange = f"[{R_low}:{R_high}]",
#      xrange = "[5.8:6.8]",
      yrange = f"[{t_low}:{t_high}]",
      key = None,
      view = "map",
      size = "ratio 0.5 0.8,1")
g.splot(f"'{outfile_pm3d}' u 1:(1e15*$2):4 w pm3d")

# Convert to pdf, crop pdf and clean up (conversion wont work, so perform the commands printed in the end directly in the shell)
#subprocess.run([
#    'gs',
#    '-P-',  # don't look first in current dir for lib files
#    '-dSAFER',
#    '-q',   # quiet
#    '-dNOPAUSE',    # no prompt, no pause at end of pages
#    '-dBATCH',      # exit gs after processing
#    '-sDEVICE=pdfwrite',        # select output device
##    '-dEPSCrop',    # crop to EPS bounding box
#    '-sOutputFile=gp_uncropped.pdf',    # select output file
#    'gp_outfile.eps',            # input file
#], shell=False)
#with open(os.devnull, 'w') as dummyout:
#    subprocess.call(['pdfcrop', 'gp_uncropped.pdf', 'wavefunction_res_combined.pdf'], stdout=dummyout)
#os.remove('gp_outfile.eps')
#os.remove('gp_uncropped.pdf')
print('### Manually perform the following command: ###\nps2pdf gp_outfile.eps gp_uncropped.pdf; pdfcrop gp_uncropped.pdf wavefunction_res_combined.pdf >/dev/null; rm gp_outfile.eps gp_uncropped.pdf\n######')
