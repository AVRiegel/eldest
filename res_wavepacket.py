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


# set up argument parser
parser = argparse.ArgumentParser(
        description='''This script calculates the wavepacket in the resonance state
        from the projections onto the vibronic resonance states
        that are the output of res_nuclear_dyn.py (in wp_res.dat).''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Alexander V. Riegel, 2024.')
parser.add_argument('-w', '--wavepacket_infile', nargs='+', default=['wp_res.dat'],
                    help='File which contains the projections of the wavefunction on the vibronic resonance states.')
parser.add_argument('-s', '--settings_infile', default='photonucl.in',
                    help='File which includes the simulation settings, potential parameters etc.')
parser.add_argument('-R', '--r_lims', nargs=3, default=[5.8, 6.8, 7], metavar=('R_low', 'R_high', 'R_num_exp'),
                    help='Lower and upper limit for R (in a.u.) as well as exponent for number of points with num = 2**R_num_exp + 1.')
parser.add_argument('-t', '--time_lims', nargs=2, default=[-1.2, 100.], metavar=('t_low', 't_up'),
                    help='Iterable with lower and upper limit for time t (in fs).')
parser.add_argument('-l', '--lambda_indiv', default=False,
                    help='Calculate "populations" also for individual vibrational resonance levels.')
args = parser.parse_args()


# read in potentials and set infile/outfile
infile_list = args.wavepacket_infile
settings = args.settings_infile

for infile in infile_list:
    if Path(infile).is_file():
        print('Input file for resonance-state projections:', infile)
    else:
        sys.exit('Input file for resonance-state projections "%s" does not exist.' % infile)

if Path(settings).is_file():
    print('Input file for simulation and potential settings:', settings)
else:
    sys.exit('Input file for simulation and potential settings "%s" does not exist.' % settings)

with open(os.devnull, 'w') as dummyfile, silence_print():
    (X_ICD, X_RICD, _, _, N_res,
     _, _, _, _, _, _, _,
     _, _, _, _, _, _,
     _, _, _, _, _, _, _, _, _,
     _, _, _, Ep_step_eV, _, _, Ep_min_eV, Ep_max_eV,
     _, _, _, _, _, _, _,
     mass1, mass2, _, _,
     _, _, _, _,
     De_list, alpha_list, Req_list, _,
     _, _, _, _, _
     ) = in_out.read_input(settings, dummyfile)

if len(infile_list) == N_res:
    print('Number of electronic resonance states: N_res =', N_res)
else:
    sys.exit(f'Number of input files for resonance-state projections, {len(infile_list)}, does not equal number of electronic resonance states, {N_res}.')


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

psi_list = []  # Will store the total wavepacket psi(R,t) for each electronic resonance state

for res in range(N_res):
    infile  = infile_list[res]
    outfile = f'wf_{infile}'
    with open(infile,'r') as f:
        raw_data = pd.read_csv(f, header=None, sep=r'\s+', engine='python')

    # Select only rows with chosen E_p value, then drop E_p column and renumber columns
    Ep_eV = raw_data[1].iloc[num_p]
    if res == 0:
        print(f'num_p = {num_p} <=> E_p = {Ep_eV:>8.5f} eV')
    data = raw_data.loc[raw_data[1] == Ep_eV].reset_index(drop=True)
    data.drop(1,axis=1,inplace=True)
    data.columns = range(data.columns.size)

    # Morse potential
    red_mass = wf.red_mass_au(mass1,mass2)
    lambda_param_res = np.sqrt(2*red_mass*De_list[res]) / alpha_list[res]
    N_lambda = int(lambda_param_res - 0.5) + 1

    # Sort input by time and quantum number, repeat each line R_len times (each t and lambda evaluated at each R)
    data[2] = data[2].astype(complex)                       # n, t, amplitude_n(t)
    mata = data.sort_values([0,1]).reset_index(drop=True)   # n, t, amplitude_n(t) (sorted by n and t)
    sata = mata.loc[data.index.repeat(R_len)]
    sata[3] = np.tile(R_arr, len(data[0]))                  # n, t, amplitude_n(t), R

    # At each point, multiply the WF of the vibrational state with the projection of the total WF on it
    sata[4] = complex(0)
    for n in range(N_lambda):
        s_down, s_up = n*len(data[0])//N_lambda, (n+1)*len(data[0])//N_lambda-1     # select block with the current quantum number n
        sata.loc[s_down:s_up,4] = sata.loc[s_down:s_up,2] * wf.psi_n(sata.loc[s_down:s_up,3], int(data[0][n]), alpha_list[res], Req_list[res], red_mass, De_list[res])  # n, t, amplitude_n(t), R, psi_n(R,t) = amplitude_n(t)*chi_n(R)
    step_width = len(sata[0])//N_lambda

    # Prepare an additional block for the whole resonance wavepacket, indicate by quantum number -1
    fata = pd.concat((sata, sata[sata[0] == 0].set_index(sata[sata[0] == 0].index + sata.index[-1] + 1)))
    fata.loc[len(data[0]):,0] = -1      # Again: n, t, amplitude_n(t), R, psi_n(R,t) = amplitude_n(t)*chi_n(R)

    # Add up the contributions at each R,t point to the whole resonance wavepacket at this point
    aata = np.array(fata[[3,1,0,4]])    # R, t, n, psi_n(R,t)
    with np.nditer(np.arange(step_width)) as it:
        for x in it:
            aata[N_lambda*step_width + x][3] = sum(aata[n*step_width + x][3] for n in range(N_lambda))  # R, t, n, psi_n(R,t) and psi(R,t) = Sum_n psi_n(R,t)

    # Write out the array
    psi_list.append(aata[-step_width:])
    pata = abs(aata[:,3])**2             # |psi_n(R,t)|^2 and |psi(R,t)|^2
    oata = np.column_stack((aata[:,:3].astype(float), pata))  # R, t, n, |psi_n(R,t)|^2 and |psi(R,t)|^2
    np.savetxt(outfile, oata, delimiter='   ', fmt=['%10.7f', '% .7e', '% i', '% .15e'])


    ##########

    # Extract the total resonance-state wavepacket, restructure the file for pm3d and calc population & R expectation value
    outfile_pm3d=f'pm3d_{outfile}'
    eata = oata[-step_width:]       # R, t, 'n' = -1, |psi(R,t)|^2
    np.savetxt(outfile_pm3d, eata, delimiter='   ', fmt=['%10.7f', '% .7e', '% i', '% .15e'])
    subprocess.call(['sed', '-i', f'/^[[:space:]]*{R_high:.7f}/G', outfile_pm3d])

    popfile=f'pop_{infile}'
    pop = pd.DataFrame()    # t, pop(t) = Int dR |psi(R,t)|^2
    for t in range(len(eata)//len(R_arr)):
        pop.loc[t, 0] = mata[1][t]
        pop.loc[t, 1] = simpson(eata[t*len(R_arr):(t+1)*len(R_arr)][:,3],dx=R_arr[1]-R_arr[0])
    np.savetxt(popfile, pop, delimiter='   ', fmt=['% .7e', '% .15e'])

    if args.lambda_indiv:
        for n in range(N_lambda):
            eata_sub = oata[n*step_width:(n+1)*step_width]  # R, t, n = const., |psi_n(R,t)|^2
            popfile_sub=f'pop_{n}_{infile}'
            pop_sub = pd.DataFrame()       # t, pop_n(t) = Int dR |psi_n(R,t)|^2 (n = const.)
            for t in range(len(eata_sub)//len(R_arr)):
                pop_sub.loc[t, 0] = mata[1][t]
                pop_sub.loc[t, 1] = simpson(eata_sub[t*len(R_arr):(t+1)*len(R_arr)][:,3],dx=R_arr[1]-R_arr[0])
            np.savetxt(popfile_sub, pop_sub, delimiter='   ', fmt=['% .7e', '% .15e'])


    expectfile=f'expect-R_{infile}'
    expect = pd.DataFrame()     # t, <R>(t) = Int dR R |psi(R,t)|^2
    for t in range(1,len(eata)//len(R_arr)):
        expect.loc[t-1, 0] = mata[1][t]
        expect.loc[t-1, 1] = simpson(eata[t*len(R_arr):(t+1)*len(R_arr)][:,0]*eata[t*len(R_arr):(t+1)*len(R_arr)][:,3],dx=R_arr[1]-R_arr[0])/pop[1][t]
    np.savetxt(expectfile, expect, delimiter='   ', fmt=['% .7e', '% .15e'])


    # Plot to eps
    g = gnuplot.Gnuplot()
    g.set(terminal = "postscript enhanced color size 30cm,15cm font 'Helvetica,26' lw 4",
        output = f"'gp_{outfile.rpartition('.')[0]}.eps'",
        bmargin = "0.5",
        lmargin = "5.0",
        rmargin = "-5.0",
        cbtics = "font ',20'",
        xlabel = "'R (a.u.)'",
        ylabel = "'t (fs)'",
        zlabel = "'P (a.u.)'",
        xrange = f"[{R_low}:{R_high}]",
        yrange = f"[{t_low}:{t_high}]",
        key = None,
        view = "map",
        size = "ratio 0.5 0.8,1")
    g.splot(f"'{outfile_pm3d}' u 1:(1e15*$2):4 w pm3d")

##########

# For the total resonance-states wavepacket Psi(R,t) = sum_res psi_res(R,t)
if N_res > 1:
    Psi = np.sum([psi[:,3] for psi in psi_list], axis=0)
    oata_total = np.column_stack((psi_list[0][:,:3].astype(float), abs(Psi)**2))    # R, t, 'n' = 1, |Psi(R,t)|^2
    np.savetxt('wf_wp_res_total.dat', oata_total, delimiter='   ', fmt=['%10.7f', '% .7e', '% i', '% .15e'])
    with open('pm3d_wf_wp_res_total.dat','w') as pmout:
        subprocess.call(['sed', f'/^[[:space:]]*{R_high:.7f}/G', 'wf_wp_res_total.dat'], stdout=pmout)

    pop_total = pd.DataFrame()      # t, Pop(t) = Int dR |Psi(R,t)|^2
    for t in range(len(oata_total)//len(R_arr)):
        pop_total.loc[t, 0] = mata[1][t]    # Uses mata still in memory from last electronic resonance state
        pop_total.loc[t, 1] = simpson(oata_total[t*len(R_arr):(t+1)*len(R_arr)][:,3],dx=R_arr[1]-R_arr[0])
    np.savetxt('pop_wp_res_total.dat', pop_total, delimiter='   ', fmt=['% .7e', '% .15e'])

    expect_total = pd.DataFrame()   # t, <R>(t) = Int dR R |Psi(R,t)|^2
    for t in range(1,len(oata_total)//len(R_arr)):
        expect_total.loc[t-1, 0] = mata[1][t]
        expect_total.loc[t-1, 1] = simpson(oata_total[t*len(R_arr):(t+1)*len(R_arr)][:,0]*oata_total[t*len(R_arr):(t+1)*len(R_arr)][:,3],dx=R_arr[1]-R_arr[0])/pop_total[1][t]
    np.savetxt('expect-R_wp_res_total.dat', expect_total, delimiter='   ', fmt=['% .7e', '% .15e'])

    g = gnuplot.Gnuplot()
    g.set(terminal = "postscript enhanced color size 30cm,15cm font 'Helvetica,26' lw 4",
        output = "'gp_wf_wp_res_total.eps'",
        bmargin = "0.5",
        lmargin = "5.0",
        rmargin = "-5.0",
        cbtics = "font ',20'",
        xlabel = "'R (a.u.)'",
        ylabel = "'t (fs)'",
        zlabel = "'P (a.u.)'",
        xrange = f"[{R_low}:{R_high}]",
        yrange = f"[{t_low}:{t_high}]",
        key = None,
        view = "map",
        size = "ratio 0.5 0.8,1")
    g.splot("'pm3d_wf_wp_res_total.dat' u 1:(1e15*$2):4 w pm3d")

    infile_list.append('wp_res_total.dat')

print('### Manually perform the following command: ###\n for f in {}; do ps2pdf "gp_wf_"$f".eps" gp_uncropped.pdf; pdfcrop gp_uncropped.pdf "wavefunction_"$f"_combined.pdf" >/dev/null; rm "gp_wf_"$f".eps" gp_uncropped.pdf; done\n ######'.format(
    ' '.join(f"'{file.rpartition('.')[0]}'" for file in infile_list)
))
