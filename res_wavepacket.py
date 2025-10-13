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

#######################################
# Prepare array with R values
R_low = 5.8
R_hig = 6.8
R_len = 2**7 + 1
R_arr = np.linspace(R_low,R_hig,R_len)
#######################################

# set up argument parser
parser = argparse.ArgumentParser(
        description='''This script calculates the wavepacket in the resonance state
        from the projections onto the vibronic resonance states
        that are the output of res_nuclear_dyn.py (in wp_res.dat).''',
        epilog='Alexander V. Riegel, 2024.')
parser.add_argument('-w', '--wavepacket_infile', default='wp_res.dat',
                    help='File which contains the projections of the wavefunction on the vibronic resonance states.')
parser.add_argument('-s', '--settings_infile', default='photonucl.in',
                    help='File which includes the simulation settings, potential parameters etc.')
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
    (_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
     _, _, _, _, _, _, _, _, _, _, _, _, mass1, mass2, _, _, _, _, _, _, De, alpha, Req, _, res_pot_type, _, _, _, _, _
     ) = in_out.read_input(settings, dummyfile)

if not res_pot_type == 'morse':
    sys.exit('Only available for Morse potential as resonance-state potential at the moment.')


outfile=f'wf_{infile}'


##########

with open(infile,'r') as f:
    data = pd.read_csv(f, header=None, sep='   ', engine='python')

# Morse potential
red_mass = wf.red_mass_au(mass1,mass2)
lambda_param_res = np.sqrt(2*red_mass*De) / alpha
N_lambda = int(lambda_param_res - 0.5) + 1

# Sort input by time and quantum number, repeat each line R_len times (each t and lambda evaluated at each R)
data[2] = data[2].astype(complex)
mata = data.sort_values([0,1]).reset_index(drop=True)
sata = mata.loc[data.index.repeat(R_len)]
sata[3] = np.tile(R_arr, len(data[0]))

# At each point, multiply the WF of the vibrational state with the projection of the total WF on it
sata[4] = complex(0)
for n in range(N_lambda):
    s_down, s_up = n*len(data[0])//N_lambda, (n+1)*len(data[0])//N_lambda-1     # select block with the current quantum number n
    sata.loc[s_down:s_up,4] = sata.loc[s_down:s_up,2] * wf.psi_n(sata.loc[s_down:s_up,3], int(data[0][n]), alpha, Req, red_mass, De)

# Prepare an additional block for the whole resonance wavepacket, indicate by quantum number -1
fata = pd.concat((sata, sata[sata[0] == 0].set_index(sata[sata[0] == 0].index + sata.index[-1] + 1)))
fata.loc[len(data[0]):,0] = -1

# Add up the contributions at each R,t point to the whole resonance wavepacket at this point
aata = np.array(fata[[3,1,0,4]])
with np.nditer(np.arange(len(sata[0])//N_lambda)) as it:
    for x in it:
      aata[N_lambda*len(sata[0])//N_lambda + x][3] = sum(aata[n*len(sata[0])//N_lambda + x][3] for n in range(N_lambda))

# Write out the array
pata = np.array((abs(aata[:,3])**2,)).T
oata = np.hstack((aata[:,:3].astype(float), pata))
np.savetxt(outfile, oata, delimiter='   ', fmt=['%10.7f', '% .7e', '% i', '% .15e'])


##########

# Extract the total resonance-state wavepacket, restructure the file for pm3d and calc population & R expectation value
outfile_pm3d=f'pm3d_{outfile}'
eata = oata[-len(sata[0])//N_lambda:]
np.savetxt(outfile_pm3d, eata, delimiter='   ', fmt=['%10.7f', '% .7e', '% i', '% .15e'])
subprocess.call(['sed', '-i', f'/{R_hig:10.7f}/G', outfile_pm3d])

popfile=f'pop_{infile}'
pop = pd.DataFrame() 
for t in range(len(eata)//len(R_arr)):
    pop.loc[t, 0] = mata[1][t]
    pop.loc[t, 1] = simpson(eata[t*len(R_arr):(t+1)*len(R_arr)][:,3]**2,dx=R_arr[1]-R_arr[0])
np.savetxt(popfile, pop, delimiter='   ', fmt=['% .7e', '% .15e'])

expectfile=f'expect-R_{infile}'
expect = pd.DataFrame()
#expect.loc[0, 0] = mata[1][0]
#expect.loc[0, 1] = 0.
for t in range(1,len(eata)//len(R_arr)):
    expect.loc[t-1, 0] = mata[1][t]
    expect.loc[t-1, 1] = simpson(eata[t*len(R_arr):(t+1)*len(R_arr)][:,0]*eata[t*len(R_arr):(t+1)*len(R_arr)][:,3]**2,dx=R_arr[1]-R_arr[0])/pop[1][t]
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
      xrange = f"[{R_low}:{R_hig}]",
#      xrange = "[5.8:6.8]",
      yrange = "[-1.2:100]",
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
