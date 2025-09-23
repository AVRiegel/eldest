#!/usr/bin/python

##########################################################################
#                                    ELDEST                              #
#        Investigating Electronic Decay Processes with Streaking         #
##########################################################################
# Purpose:                                                               #
#          - A program to simulate the time-resolved RICD spectroscopy   #
#            including quantum nuclear dynamics.                         #
#                                                                        #
##########################################################################
# written by: Elke Fasshauer November 2020                               #
# extended by: Alexander Riegel from July 2023 onwards                   #
# last change: 2025-06-25 AVR                                            #
##########################################################################

import argparse
from datetime import datetime
import dill
import mpmath as mp
import numpy as np
from os import devnull
import scipy
import scipy.integrate as integrate
from scipy.signal import argrelextrema
import sys
import warnings

import complex_integration as ci
import in_out
import sciconv
import wellenfkt as wf

dt_start = datetime.now()

# set logging outfile
outfile = open("eldest.out", mode='w')

# don't print warnings unless python -W ... is used
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# set up argument parser
parser = argparse.ArgumentParser(
        description='''ELDEST -- nuclear_dyn.py :
        A programme to simulate the time-resolved RICD spectroscopy
        including quantum nuclear dynamics.''',
        epilog='Originally written by Elke Fasshauer, extended by Alexander V. Riegel.')
parser.add_argument('infile', help='Input file for simulation, probably photonucl.in')
parser.add_argument('-f', '--fc', help='''Optional file with pre-calculated "Franck-Condon overlap integrals"
                    (which may or may not include weighting functions inside the integrand).
                    The file is thought to be a copy of eldest.out from a previous calculation.
                    It may be truncated, but at least the Franck-Condon-integrals block must be present
                    (starting with the line "Franck-Condon overlaps between ground and resonance state"
                    and ending with the last final-resonance integral) and structured as in the eldest.out file.
                    +++ This option is incompatible with the -g/--gamma option.''')
parser.add_argument('-g', '--gamma', help='''Optional binary file containing the functional dependence
                    of the decay width Gamma on the internuclear distance R.
                    The information can be stored as a univariate function or an SciPy interpolator.
                    Permissible are, besides user-defined functions, all "np.foo" and "scipy.foo",
                    with full tree beginning at the module, e.g. "np.sqrt";
                    "myfunc" defined as def myfunc(x): return np.polynomial.hermite.Hermite((2,0,8))(x);
                    "scipy.interpolate.PchipInterpolator(xarray,yarray)".
                    The file shall be binary and contain the functional dependence in a pickled form (preferably by dill).
                    +++ This option is incompatible with the -f/--fc option.''')
#parser.add_argument('-p', '--partial', help='''If 'prefactor' or 'pre' is chosen, then the Gamma(R) dependence is incorporated
#                    only into the overlap integrals in the prefactors for the transition amplitude
#                    but not in W_lambda in the exponents. If 'exponent' or 'exp' or 'Wl' is chosen, the reverse is true.
#                    If none is given, the Gamma(R) dependence is incorporated in all relevant places (default).''')
parser.add_argument('-F', '--FC', help='''Same as '-f' and '--fc', but for an additional file with overlap integrals
                    without Gamma(R) dependence in the res-fin integrals. The file structure is the same as before;
                    this also means that only one res-fin block will be recognized - if the file contains both
                    res-fin integrals with and without Gamma(R) dependence, the block with Gamma(R) must be deleted.
                    +++ This option is only available if partial_GamR is not None.''')
#                    +++ This option is only available in combination with the -p/--partial option.''')
#parser.add_argument('-w', '--wavepacket_only', action='store_true', help='''If this flag is given, only the projection
#                    onto the vibrational states of the electronic resonance state (needed to reconstruct
#                    the wavepacket in the resonance state) will be calculated, whereas the calculation of the projections
#                    onto the final state (needed for the spectrum) will be skipped. Also, progress will be written
#                    to eldest.out as usual, but existing full.dat and movie.dat files will not be altered.''')
args = parser.parse_args()

print(str(dt_start))
outfile.write(str(dt_start) + '\n')
outfile.write('Tempora mutantur, nos et mutamur in illis.\n')
outfile.write("The results were obtained with nuclear_dyn.py\n")

infile = args.infile
print(infile)

#-------------------------------------------------------------------------
# set some defaults
Xshape = 'convoluted'

#-------------------------------------------------------------------------
# read inputfile
# (see next section for explanations of most symbols)
# ( * X_sinsq, X_gauss are simply Booleans, created by in_out from X_shape)
# ( * phi is the phase for the IR pulse potential cosine-oscillation, a remnant from PRA 2020)
# ( * integ, integ_outer are integration schemes: [analytic,] quadrature, romberg)
# (currently NOT in use: cdg_au, tau_a_s, tau_b_s interact_eV, Lshape, shift_step_s, phi, grad_delta, R_eq_AA, gs_const, res_const)
# ( * Er_b_eV and E_fin_eV_2 will be converted to au, but these will not be used afterwards)
# ( * tau_s_2 will be converted to au at this to Gamma, but this will not be used afterwards)
# ( * omega_eV will be converted to au, from which TL and A0L are calculated, but other than being used for needless printing and for check_input, they will not be used afterwards)
# ( * n_L and I_L only lead to related qnts like TL, E0L and A0L, for which above holds)
# ( * FWHM_L will be converted to au and this printed, but not be used afterwards)
# ( * fin_d will be used to bind fin_const for Morse final potential, but both will not be used afterwards)

# (q is explicit input, not calced as q = rdg / (cdg pi VEr) = sqrt(2 tau / pi) rdg / cdg )

(rdg_au, cdg_au,
 Er_a_eV, Er_b_eV, tau_a_s, tau_b_s, E_fin_eV, tau_s, E_fin_eV_2, tau_s_2,
 interact_eV,
 Omega_eV, n_X, I_X, X_sinsq, X_gauss, Xshape,
 omega_eV, n_L, I_L, Lshape, delta_t_s, shift_step_s, phi, q, FWHM_L,
 tmax_s, timestep_s, E_step_eV,
 E_min_eV, E_max_eV,
 integ, integ_outer, Gamma_type,
 fc_precalc, partial_GamR, part_fc_pre, wavepac_only,
 mass1, mass2, grad_delta, R_eq_AA,
 gs_de, gs_a, gs_Req, gs_const,
 res_de, res_a, res_Req, res_const,
 fin_a, fin_b, fin_c, fin_d, fin_pot_type
 ) = in_out.read_input(infile, outfile)


#-------------------------------------------------------------------------
# open further output files
pure_out = open('full.dat' if not wavepac_only else devnull, mode='w')
movie_out = open('movie.dat' if not wavepac_only else devnull, mode='w')
#popfile = open("pop.dat", mode='w')
wp_res_out = open('wp_res.dat', mode='w')

if fc_precalc:
    print('The Franck-Condon overlap integrals are read from file: ' + str(args.fc))
    outfile.write('The Franck-Condon overlap integrals are read from file: ' + str(args.fc) + '\n')
else:
    print('The Franck-Condon overlap integrals are calculated from scratch')
    outfile.write('The Franck-Condon overlap integrals are calculated from scratch\n')

if partial_GamR:
    if part_fc_pre:
        print('Additional res-fin overlap integrals without Gamma(R) dependence are read from file: ' + str(args.FC))
        outfile.write('Additional res-fin overlap integrals without Gamma(R) dependence are read from file: ' + str(args.FC) + '\n')
    else:
        print('Additional res-fin overlap integrals without Gamma(R) dependence are calculated from scratch')
        outfile.write('Additional res-fin overlap integrals without Gamma(R) dependence are calculated from scratch\n')

if wavepac_only:
    print('Only the resonance-state projections will be calculated, not the spectrum (final-state projections)')
    outfile.write('Only the resonance-state projections will be calculated, not the spectrum (final-state projections)' + '\n')

if args.gamma:
    print('Gamma(R) dependence is read from file: ' + str(args.gamma))
    outfile.write('Gamma(R) dependence is read from file: ' + str(args.gamma) + '\n')


#-------------------------------------------------------------------------
# Convert input parameters to atomic units
#-------------------------------------------------------------------------
Er_a_au        = sciconv.ev_to_hartree(Er_a_eV)     # resonance E for RICD + AI
#Er_b_au        = sciconv.ev_to_hartree(Er_b_eV)     # resonance E for ICD
Er_au          = Er_a_au        # ? One could delete Er_a_au altogether
E_fin_au       = sciconv.ev_to_hartree(E_fin_eV)    # (same as for Er)
E_fin_au_1     = sciconv.ev_to_hartree(E_fin_eV)    # final E for sRICD

tau_au_1       = sciconv.second_to_atu(tau_s)       # lifetime for sRICD res. st.
tau_au         = tau_au_1                           # (same as for Er)
Gamma_au       = 1. / tau_au
Gamma_eV       = sciconv.hartree_to_ev(Gamma_au)
if Gamma_type == 'const':
    outfile.write('Gamma_eV = ' + str(Gamma_eV) + '\n')

# second final state
#E_fin_au_2       = sciconv.ev_to_hartree(E_fin_eV_2)
#tau_au_2         = sciconv.second_to_atu(tau_s_2)
#Gamma_au_2       = 1. / tau_au_2

# laser parameters
Omega_au      = sciconv.ev_to_hartree(Omega_eV)
if (X_sinsq):
    TX_au     = n_X * 2 * np.pi / Omega_au
elif(X_gauss):
    sigma     = np.pi * n_X / (Omega_au * np.sqrt(np.log(2)))
    FWHM      = 2 * np.sqrt( 2 * np.log(2)) * sigma
    TX_au     = 5 * sigma
    sigma_E   = 1. / (2 * sigma)
    width_E   = 5 * sigma_E
    EX_max_au = Omega_au + 0.5 * width_E
    print('sigma [s] = ', sciconv.atu_to_second(sigma))
    print('FWHM [s] = ', sciconv.atu_to_second(FWHM))
    print('sigma_E [eV] = ', sciconv.hartree_to_ev(sigma_E))
    print('XUV reaches up to {:5.5f} au = {:5.5f} eV'.format(
        EX_max_au, sciconv.hartree_to_ev(EX_max_au)))
    outfile.write('sigma [s] = ' + str(sciconv.atu_to_second(sigma)) + '\n')
    outfile.write('FWHM [s] = ' + str(sciconv.atu_to_second(FWHM)) + '\n')
    outfile.write('sigma_E [eV] = ' + str(sciconv.hartree_to_ev(sigma_E)) + '\n')
    outfile.write('XUV reaches up to {:5.5f} au = {:5.5f} eV\n'.format(
        EX_max_au, sciconv.hartree_to_ev(EX_max_au)))
print('end of the first pulse [s] = ', sciconv.atu_to_second(TX_au/2))
outfile.write('end of the first pulse [s] = ' + str(sciconv.atu_to_second(TX_au/2)) + '\n')
I_X_au        = sciconv.Wcm2_to_aiu(I_X)
print('I_X [W/cm^2] = ', I_X)
print('I_X_au = ', I_X_au)
E0X           = np.sqrt(I_X_au)
A0X           = E0X / Omega_au
print('A0X [au] = ', A0X)

omega_au      = sciconv.ev_to_hartree(omega_eV)
#FWHM_L_au     = sciconv.second_to_atu(FWHM_L)
#sigma_L_au    = FWHM_L_au / np.sqrt(8 * np.log(2))      # assume Gaussian envelope for second pulse
#a             = 5./2 * sigma_L_au       # half duration of IR pulse (delta_t - a, delta_t + a); in PRA 2020: small-delta t
#print("FWHM_L [s] = ", FWHM_L)
#print("sigma_L [s] = ", sciconv.atu_to_second(sigma_L_au))
TL_au         = n_L * 2 * np.pi / omega_au
#print('start of IR pulse [s] = ', delta_t_s - sciconv.atu_to_second(TL_au/2))
#print('end of IR pulse [s] = ', delta_t_s + sciconv.atu_to_second(TL_au/2))
I_L_au        = sciconv.Wcm2_to_aiu(I_L)
#print('I_L [W/cm^2] = ', I_L)
#print('I_L_au = ', I_L_au)
E0L           = np.sqrt(I_L_au)
##print('E0L [au] = ', E0L)
A0L           = E0L / omega_au
#print('A0L [au] = ', A0L)
delta_t_au    = sciconv.second_to_atu(delta_t_s)        # t diff between the maxima of the two pulses

# parameters of the simulation
tmax_au       = sciconv.second_to_atu(tmax_s)
timestep_au   = sciconv.second_to_atu(timestep_s)
E_step_au = sciconv.ev_to_hartree(E_step_eV)

E_min_au = sciconv.ev_to_hartree(E_min_eV)
E_max_au = sciconv.ev_to_hartree(E_max_eV)

VEr_au        = np.sqrt(Gamma_au/ (2*np.pi))
#VEr_au_1      = VEr_au      # (same as for Er)

cdg_au_V = rdg_au / ( q * np.pi * VEr_au)

if Gamma_type == 'const':
    print('VEr_au = ', VEr_au)
    outfile.write('VEr_au = ' + str(VEr_au) + '\n')
elif Gamma_type == 'R6':
    if partial_GamR:
        VEr_au_woVR = VEr_au
        print('VEr_au = ', VEr_au)
        outfile.write('VEr_au = ' + str(VEr_au) + '\n')
    VEr_au = VEr_au*res_Req**3                            # adjusts VEr_au by the R dependent factor
    print('VEr_au_adjusted = ', VEr_au)
    outfile.write('VEr_au_adjusted = ' + str(VEr_au) + '\n')
elif Gamma_type == 'external':
    if partial_GamR:
        VEr_au_woVR = VEr_au
        print('VEr_au = ', VEr_au)
        outfile.write('VEr_au = ' + str(VEr_au) + '\n')
    VEr_au = 1          # all info about V carried in the 'Franck-Condon' integrals, so ignore VEr_au


#-------------------------------------------------------------------------
# Potential details
# vibrational energies of Morse potentials
print()
print('-----------------------------------------------------------------')
outfile.write('\n' + '-----------------------------------------------------------------' + '\n')
red_mass = wf.red_mass_au(mass1,mass2)
print("red_mass [au] = ", red_mass)

#ground state
print()
print("Ground state")
print('-----------------------------------------------------------------')
print("Energies of vibrational states of the ground state")
outfile.write('\n' + '-----------------------------------------------------------------' + '\n')
outfile.write("Energies of vibrational states of the ground state" + '\n')
lambda_param_gs = np.sqrt(2*red_mass*gs_de) / gs_a
n_gs_max = int(lambda_param_gs - 0.5)   # ? ONLY ONE GS MAY BE POPULATED OR ELSE EQS ARE WRONG
print("n_gs_max = ", n_gs_max)
print('n_gs  ' + 'E [au]            ' + 'E [eV]')
outfile.write('n_gs  ' + 'E [au]            ' + 'E [eV]' + '\n')
E_kappas = wf.eigenvalue(np.arange(0, n_gs_max + 1), gs_de, gs_a, red_mass)  # collects vibr energies of GS
for n, (eigv_au, eigv_eV) in enumerate(np.column_stack(
        (E_kappas,sciconv.hartree_to_ev(E_kappas)))):
    outfile.write('{:4d}  {:14.10E}  {:14.10E}\n'.format(n, eigv_au, eigv_eV))
    print('{:4d}  {:14.10E}  {:14.10E}'.format(n, eigv_au, eigv_eV))

#resonance state
print()
print("Resonance state")
print('-----------------------------------------------------------------')
print("Energies of vibrational states of the resonance state")
outfile.write('\n' + '-----------------------------------------------------------------' + '\n')
outfile.write("Energies of vibrational states of the resonance state" + '\n')
lambda_param_res = np.sqrt(2*red_mass*res_de) / res_a
n_res_max = int(lambda_param_res - 0.5)
print("n_res_max = ", n_res_max)
print('n_res  ' + 'E [au]            ' + 'E [eV]')
outfile.write('n_res  ' + 'E [au]            ' + 'E [eV]' + '\n')
E_lambdas = wf.eigenvalue(np.arange(0, n_res_max + 1), res_de, res_a, red_mass)
for n, (eigv_au, eigv_eV) in enumerate(np.column_stack(
        (E_lambdas,sciconv.hartree_to_ev(E_lambdas)))):
    outfile.write('{:5d}  {:14.10E}  {:14.10E}\n'.format(n, eigv_au, eigv_eV))
    print('{:5d}  {:14.10E}  {:14.10E}'.format(n, eigv_au, eigv_eV))

#final state
print()
print("Final state")
print('-----------------------------------------------------------------')
outfile.write('\n' + '-----------------------------------------------------------------' + '\n')
if (fin_pot_type == 'morse'):
    print("Energies of vibrational states of the final state")
    outfile.write("Energies of vibrational states of the final state" + '\n')
    fin_de    = fin_a
    fin_a     = fin_b
    fin_Req   = fin_c
    fin_const = fin_d
    lambda_param_fin = np.sqrt(2*red_mass*fin_de) / fin_a
    n_fin_max = int(lambda_param_fin - 0.5)     # Maximum quantum number = n_fin_max -> number of states = n_fin_max + 1
    print("n_fin_max = ", n_fin_max)
    print('n_fin  ' + 'E [au]            ' + 'E [eV]')
    outfile.write('n_fin  ' + 'E [au]            ' + 'E [eV]' + '\n')
    E_mus = wf.eigenvalue(np.arange(0, n_fin_max + 1), fin_de, fin_a, red_mass)
    for n, (eigv_au, eigv_eV) in enumerate(np.column_stack(
            (E_mus, sciconv.hartree_to_ev(E_mus)))):
        outfile.write('{:5d}  {:14.10E}  {:14.10E}\n'.format(n, eigv_au, eigv_eV))
        print('{:5d}  {:14.10E}  {:14.10E}'.format(n, eigv_au, eigv_eV))
elif (fin_pot_type in ('hyperbel','hypfree')):
    print('Final state is repulsive')
    outfile.write('Final state is repulsive' + '\n')
    fin_hyp_a = fin_a
    fin_hyp_b = fin_b
    E_fin_au = fin_hyp_b        # Since for an all-repulsive state there is no minimum (E_fin), E_fin is set to the final potential at infinite distance, i.e. fin_hyp_b
    E_fin_au_1 = fin_hyp_b
    R_hyp_step = fin_c
    threshold = fin_d   # If, coming from high mu, for a certain mu all |<mu|kappa>| and |<mu|lambda>| are < threshold, don't calc FCF and integrals for all mu < that mu
    R_start_EX_max = fin_hyp_a / (EX_max_au - fin_hyp_b)        # R_start of hyperbola corresponding to EX_max_au, used as minimum starting point for discretizing final vibr states
    outfile.write('Continuous vibrational states of the final state are discretized:\n')
    outfile.write('Energy of highest possibly considered vibrational state\n of the final state is {0:.5f} eV\nStep widths down from there decrease as (eV) {1:.5f}, {2:.5f} ...\n'.format(
        sciconv.hartree_to_ev(EX_max_au - fin_hyp_b),
        sciconv.hartree_to_ev(fin_hyp_a / R_start_EX_max  -  fin_hyp_a / (R_start_EX_max + R_hyp_step)),
        sciconv.hartree_to_ev(fin_hyp_a / (R_start_EX_max + R_hyp_step)  - fin_hyp_a / (R_start_EX_max + 2 * R_hyp_step)) ))
    outfile.write('Each E_mu is calculated as {0} au / R_start,\n where R_start begins at {1:.5f} au = {2:.5f} A\n and increases in constant steps of width {3:.5f} au = {4:.5f} A\n'.format(
        fin_hyp_a, R_start_EX_max, sciconv.bohr_to_angstrom(R_start_EX_max), R_hyp_step, sciconv.bohr_to_angstrom(R_hyp_step) ))
    print('Continuous vibrational states of the final state are discretized:')
    print('Energy of highest possibly considered vibrational state\n of the final state is {0:.5f} eV\nStep widths down from there decrease as (eV) {1:.5f}, {2:.5f} ...'.format(
        sciconv.hartree_to_ev(EX_max_au - fin_hyp_b),
        sciconv.hartree_to_ev(fin_hyp_a / R_start_EX_max  -  fin_hyp_a / (R_start_EX_max + R_hyp_step)),
        sciconv.hartree_to_ev(fin_hyp_a / (R_start_EX_max + R_hyp_step)  - fin_hyp_a / (R_start_EX_max + 2 * R_hyp_step)) ))
    print('Each E_mu is calculated as {0} au / R_start,\n where R_start begins at {1:.5f} au = {2:.5f} A\n and increases in constant steps of width {3:.5f} au = {4:.5f} A'.format(
        fin_hyp_a, R_start_EX_max, sciconv.bohr_to_angstrom(R_start_EX_max), R_hyp_step, sciconv.bohr_to_angstrom(R_hyp_step) ))

#-------------------------------------------------------------------------
# Franck-Condon factors
#-------------------------------------------------------------------------
gs_res =  []    # collects sub-lists of FC overlaps: [<l0|k0>, <l1|k0>, ...], [<l0|k1, <l1|k1>, ...], ...
gs_fin =  []
res_fin = []
R_min = sciconv.angstrom_to_bohr(1.5)+0.01
R_max = sciconv.angstrom_to_bohr(30.0)

for k in range(0,n_gs_max+1):   # prepare the above (empty) sub-lists
    gs_fin.append(list())
for l in range(0,n_res_max+1):
    res_fin.append(list())

if not fc_precalc and args.fc:
    outfile.close
    pure_out.close
    movie_out.close
    wp_res_out.close
    sys.exit('!!! FC input file was provided without being requested. Programme terminated.')
elif fc_precalc and not args.fc:
    outfile.close
    pure_out.close
    movie_out.close
    wp_res_out.close
    sys.exit('!!! FC input file was requested but not provided. Programme terminated.')
elif not (Gamma_type == 'external') and args.gamma:
    outfile.close
    pure_out.close
    movie_out.close
    wp_res_out.close
    sys.exit('!!! Gamma-R-dependence file was provided although Gamma_type is not "external". Programme terminated.')
elif (Gamma_type == 'external') and not args.fc and not args.gamma:
    outfile.close
    pure_out.close
    movie_out.close
    wp_res_out.close
    sys.exit('!!! Gamma_type is "external" but no additional input file was provided. Programme terminated.')
elif args.fc and args.gamma:
    outfile.close
    pure_out.close
    movie_out.close
    wp_res_out.close
    sys.exit('!!! FC input file and Gamma-R-dependence file were provided at the same time. Programme terminated.')
elif not part_fc_pre and args.FC:
    outfile.close
    pure_out.close
    movie_out.close
    wp_res_out.close
    sys.exit('!!! FC input file for partial Gamma-R dependence was provided without being requested. Programme terminated.')
elif part_fc_pre and not args.FC:
    outfile.close
    pure_out.close
    movie_out.close
    wp_res_out.close
    sys.exit('!!! FC input file for partial Gamma-R dependence was requested but not provided. Programme terminated.')
elif part_fc_pre and not partial_GamR:
    outfile.close
    pure_out.close
    movie_out.close
    wp_res_out.close
    sys.exit('!!! FC input file for partial Gamma-R dependence was requested although no such treatment was requested. Programme terminated.')
elif partial_GamR and ((args.fc and not args.FC) or (args.FC and not args.fc)):   # If partial_GamR but -f and -F not either both present or both absent, throw error (FC calc code would have to be changed)  
    outfile.close
    pure_out.close
    movie_out.close
    wp_res_out.close
    sys.exit('!!! If partial_GamR is requested, then either both or none of the additional FC input files with (-f) and without (-F) Gamma-R dependence must be provided at the moment. Programme terminated.')


if Gamma_type == 'const':
    V_of_R = lambda R: 1
    partial_GamR = None     # If Gamma(R)=const., then FC integrals with and without Gamma(R) are identical
elif Gamma_type == 'R6':
    V_of_R = lambda R: R**(-3)
elif args.gamma:
    with open(args.gamma, 'rb') as gammafile:
        Gamma_of_R = dill.load(gammafile)
    V_of_R = lambda R: np.sqrt(Gamma_of_R(R) / (2*np.pi))
else:                           # For 'external' but from FC file
    V_of_R = lambda R: 1

if partial_GamR:
    res_fin_woVR = []
    for l in range(0,n_res_max+1):
        res_fin_woVR.append(list())



# Integration bounds;       and calc ground state - resonance state <lambda|kappa>
if not args.fc:                 # If, however, an FC input file is provided, FC integrals will be read from it in the next step and their calculation skipped
    # Numerical integration failsafe check: calculate test FC overlap integral
    print()
    print('-----------------------------------------------------------------')
    outfile.write('\n' + '-----------------------------------------------------------------' + '\n')
    # print('Numerical integration test')
    #
    # if fin_pot_type == 'morse':
    #    func = lambda R: (np.conj(wf.psi_n(R,0,res_a,res_Req,red_mass,res_de))
    #                      * wf.psi_n(R,0,fin_a,fin_Req,red_mass,fin_de)
    #                      * V_of_R(R))
    # elif fin_pot_type == 'hypfree':
    #    func = lambda R: (np.conj(wf.psi_n(R,0,res_a,res_Req,red_mass,res_de))
    #                      * wf.psi_freehyp(R,fin_hyp_a,fin_hyp_b,red_mass,R_start_EX_max)
    #                      * V_of_R(R))
    # elif fin_pot_type == 'hyperbel':
    #    func = lambda R: (np.conj(wf.psi_n(R,0,res_a,res_Req,red_mass,res_de))
    #                      * wf.psi_hyp(R,fin_hyp_a,fin_hyp_b,red_mass,R_start_EX_max)
    #                      * V_of_R(R))
    # tmp = np.zeros(2)
    # while abs(tmp[0]) <= (1000*tmp[1]):                 # checks if the test integral is at least three orders of magnitude larger than the estimated error
    #    R_min -= 0.01                                   # if so: lower the lower integration bound by 0.01 bohr
    #    tmp = integrate.quad(func, R_min, R_max,epsabs=1e-20,limit=500)
    #    #print(R_min, tmp)  #?
    R_min -= 0.01  # Counteract the +0.01 bohr when R_min was defined

    print('Bounds of integration over R for the Franck-Condon factors')
    print('R_min = {:14.10E} au = {:5.5f} A'.format(R_min, sciconv.bohr_to_angstrom(R_min)))
    print('R_max = {:14.10E} au = {:5.5f} A'.format(R_max, sciconv.bohr_to_angstrom(R_max)))
    print('Hope that is in order.')
    outfile.write('Bounds of integration over R for the Franck-Condon factors' + '\n')
    outfile.write('R_min = {:14.10E} au = {:5.5f} A\n'.format(R_min, sciconv.bohr_to_angstrom(R_min)))
    outfile.write('R_max = {:14.10E} au = {:5.5f} A\n'.format(R_max, sciconv.bohr_to_angstrom(R_max)))
    outfile.write('Hope that is in order.' + '\n')

    # calc ground state - resonance state <lambda|kappa>
    for k in range (0,n_gs_max+1):
        tmp = []
        for l in range (0,n_res_max+1):
            FC = wf.mp_FCmor_mor(l,res_a,res_Req,res_de,red_mass,
                                 k,gs_a,gs_Req,gs_de,R_min,R_max)
            tmp.append(FC)
        gs_res.append(tmp)
    
# read in FCs;      or calc ground state - final state <mu|kappa>   and   resonance state - final state <mu|lambda>
if (fin_pot_type == 'morse'):
    if args.fc:            # If an FC input file is provided, read in the FC integrals from it and skip their calculation
        gs_res, gs_fin, res_fin, _, _ = in_out.read_fc_input(args.fc)
        if partial_GamR:
            gs_res_woVR, gs_fin_woVR, res_fin_woVR, _, _ = in_out.read_fc_input(args.FC)
            if not (gs_res_woVR == gs_res and gs_fin_woVR == gs_fin and len(res_fin) == len(res_fin_woVR)):
                outfile.write("gs_res: " + str(gs_res_woVR == gs_res) + ", gs_fin: " + str(gs_fin_woVR == gs_fin) + ", len(res_fin): " + str(len(res_fin) == len(res_fin_woVR)) + "\n")
                outfile.close
                pure_out.close
                movie_out.close
                wp_res_out.close
                sys.exit('!!! Files of FC integrals with and without Gamma(R) dependence are incompatible. Programme terminated.')
    else:
        for m in range(0,n_fin_max+1):
            for k in range(0,n_gs_max+1):
                FC = wf.mp_FCmor_mor(m,fin_a,fin_Req,fin_de,red_mass,
                                     k,gs_a,gs_Req,gs_de,R_min,R_max)
                gs_fin[k].append(FC)
            for l in range(0,n_res_max+1):
                FC = wf.mp_FCmor_mor(m,fin_a,fin_Req,fin_de,red_mass,
                                     l,res_a,res_Req,res_de,R_min,R_max,
                                     V_of_R=V_of_R)      # Gamma(R) dependence only influences res-fin FC integrals (interaction mediated by V)
                res_fin[l].append(FC)
                if partial_GamR:
                    FC = wf.mp_FCmor_mor(m,fin_a,fin_Req,fin_de,red_mass,
                                         l,res_a,res_Req,res_de,R_min,R_max,
                                         V_of_R=lambda R: 1)
                    res_fin_woVR[l].append(FC)


elif (fin_pot_type in ('hyperbel','hypfree')):
    if args.fc:            # If an FC input file is provided, read in the FC integrals from it and skip their calculation
        gs_res, gs_fin, res_fin, n_fin_max_list, n_fin_max_X = in_out.read_fc_input(args.fc)
        R_starts = R_start_EX_max + R_hyp_step * np.arange(n_fin_max_X + 1) # Initialize R_start at the lowest considered value, then increase by a constant R_hyp_step
        E_mus = fin_hyp_a / R_starts[::-1] # Start with high R_start do that low energies get low mu numbers
        norm_factor = 1.
        if partial_GamR:
            gs_res_woVR, gs_fin_woVR, res_fin_woVR, n_fin_max_list_woVR, n_fin_max_X_woVR = in_out.read_fc_input(args.FC)
            if not (gs_res_woVR == gs_res and gs_fin_woVR == gs_fin and n_fin_max_list_woVR == n_fin_max_list
                    and n_fin_max_X_woVR == n_fin_max_X and len(res_fin) == len(res_fin_woVR)):
                outfile.write("gs_res: " + str(gs_res_woVR == gs_res) + ", gs_fin: " + str(gs_fin_woVR == gs_fin) + ", max_list: " + str(n_fin_max_list_woVR == n_fin_max_list)
                              + ", max_X: " + str(n_fin_max_X_woVR == n_fin_max_X) + ", len(res_fin): " + str(len(res_fin) == len(res_fin_woVR)) + "\n")
                outfile.close
                pure_out.close
                movie_out.close
                wp_res_out.close
                sys.exit('!!! Files of FC integrals with and without Gamma(R) dependence are incompatible. Programme terminated.')

    else:
        E_mus = []
        FCfunc = wf.mp_FCmor_hyp if (fin_pot_type == 'hyperbel') else wf.mp_FCmor_freehyp
        Req_max = max(gs_Req, res_Req)
        R_start = R_start_EX_max        # Initialize R_start at the lowest considered value (then increase R_start by a constant R_hyp_step)
        thresh_flag = -1                # Initialize flag for FC-calc stop. Counts how often in a (mu) row all FC fall below threshold
        while (thresh_flag < 3):        # Stop FC calc if all |FC| < threshold for 3 consecutive mu
            E_mu = fin_hyp_a / R_start
            E_mus.insert(0,E_mu)        # Present loop starts at high energies, but these shall get high mu numbers = stand at the end of the lists -> fill lists from right to left
            print(f'--- R_start = {R_start:7.4f} au = {sciconv.bohr_to_angstrom(R_start):7.4f} A   ###   E_mu = {E_mu:7.5f} au = {sciconv.hartree_to_ev(E_mu):7.4f} eV   ###   steps: {int((R_start - R_start_EX_max) / R_hyp_step  + 0.1)}')    #?
    #        outfile.write(f'R_start = {R_start:5.5f} au = {sciconv.bohr_to_angstrom(R_start):5.5f} A, E_mu = {E_mu:5.5f} au = {sciconv.hartree_to_ev(E_mu):5.5f} eV, steps: {int((R_start - R_start_EX_max) / R_hyp_step  + 0.1)}\n')  #?
            for k in range(0,n_gs_max+1):
                FC = FCfunc(k,gs_a,gs_Req,gs_de,red_mass,
                            fin_hyp_a,fin_hyp_b,R_start,R_min,R_max)
                gs_fin[k].insert(0,FC)
                print(f'k = {k}, gs_fin  = {FC: 10.10E}, |gs_fin|  = {np.abs(FC):10.10E}')   #?
    #            outfile.write(f'k = {k}, gs_fin  = {FC: 10.10E}, |gs_fin|  = {np.abs(FC):10.10E}\n')   #?
            for l in range(0,n_res_max+1):
                FC = FCfunc(l,res_a,res_Req,res_de,red_mass,
                            fin_hyp_a,fin_hyp_b,R_start,R_min,R_max,
                            V_of_R=V_of_R)
                res_fin[l].insert(0,FC)
                print(f'l = {l}, res_fin = {FC: 10.10E}, |res_fin| = {np.abs(FC):10.10E}')   #?
    #            outfile.write(f'l = {l}, res_fin = {FC: 10.10E}, |res_fin| = {np.abs(FC):10.10E}\n')   #?
                if partial_GamR:
                    FC = FCfunc(l,res_a,res_Req,res_de,red_mass,
                                fin_hyp_a,fin_hyp_b,R_start,R_min,R_max,
                                V_of_R=lambda R: 1)
                    res_fin_woVR[l].insert(0,FC)
                    print(f'l = {l}, res_fin_woVR = {FC: 10.10E}, |res_fin_woVR| = {np.abs(FC):10.10E}')   #?
    #               outfile.write(f'l = {l}, res_fin_woVR = {FC: 10.10E}, |res_fin_woVR| = {np.abs(FC):10.10E}\n')   #?
            if (R_start > Req_max):         # Do not stop FC calc as long as R_start has not surpassed all Req
                if (all(np.abs( gs_fin[k][0]) < threshold for k in range(0, n_gs_max+1)) and
                    all(np.abs(res_fin[l][0]) < threshold for l in range(0,n_res_max+1)) ): # To keep consistency, the res_fin_woVR are not included in this check
                    if (thresh_flag != -1):     # -1 can only occur at lowest R_start values (once any FC > threshold: flag is set to 0, then stays >= 0) -> dont stop calc right at start just bc FC are small there
                        thresh_flag = thresh_flag + 1
                else:
                    thresh_flag = 0         # If any FC overlap > threshold, reset flag -> only (mu-)consecutive threshold check passes shall stop calc
            print(f'thresh_flag = {thresh_flag}')                                                                               #?
    #        outfile.write(f'thresh_flag = {thresh_flag}\n')                                                                               #?
            R_start = R_start + R_hyp_step

        # Enforce FC sum rule: for a bound vibr state |b> (b=kappa,lambda), int_0^inf dEmu <b|mu><mu|b> = 1, or discretized, sum_Emu DeltaE <b|mu><mu|b> = 1, i. e. sum_Rmu = DeltaR Va/Rmu^2 <b|mu><mu|b> = 1
    #    norm_fin_gs = []        # Current values of the sum_Rmu with |b> = |kappa>
    #    norm_fin_res = []       # Current values of the sum_Rmu with |b> = |lambda>
    #    for k in range(0,n_gs_max+1):
    #        norm_fin_gs.append(R_hyp_step / fin_hyp_a * np.sum(np.abs(gs_fin[k])**2 * np.array(E_mus)**2))
    #        gs_fin[k] = gs_fin[k] / np.sqrt(norm_fin_gs[k])     # Rescale FC overlaps <k|m> so that sum_Rmu = 1
    #    for l in range(0,n_res_max+1):
    #        norm_fin_res.append(R_hyp_step / fin_hyp_a * np.sum(np.abs(res_fin[l])**2 * np.array(E_mus)**2))
    #        res_fin[l] = res_fin[l] / np.sqrt(norm_fin_res[l])  # Rescale FC overlaps <l|m> so that sum_Rmu = 1
    #    print('norm_fin_gs =', norm_fin_gs)
    #    print('norm_fin_res =', norm_fin_res)
    #    outfile.write('norm_fin_gs = ' + str(norm_fin_gs) + '\n')       #?
    #    outfile.write('norm_fin_res = ' + str(norm_fin_res) + '\n')     #?
#        norm_factor = 1.
#       norm_factor = R_hyp_step / fin_hyp_a * np.sum(np.abs(gs_fin[0])**2 * np.array(E_mus)**2)   # All FC overlaps will be rescaled using the sum_Rmu with |b> = |k=0>
#        for k in range(0,n_gs_max+1):
#            gs_fin[k] = gs_fin[k] / np.sqrt(norm_factor)        # Rescale FC overlaps <k|m>
#        for l in range(0,n_res_max+1):
#            res_fin[l] = res_fin[l] / np.sqrt(norm_factor)      # Rescale FC overlaps <l|m>

        n_fin_max_list = []             # Max quantum number considered in non-direct ionization for each lambda (all vibr fin states above the resp res state are discarded)
        for E_l in E_lambdas:
            for n_fin in range(len(E_mus)-1, -1, -1):           # Loop over E_mus from back to start
                if (E_fin_au + E_mus[n_fin] <= Er_au + E_l):    # The highest (i.e. first, since loop starts at high n_fin) n_fin for which (E_fin + E_mu <= E_res + E_l) is n_fin_max for this l
                    n_fin_max_list.append(n_fin)
                    break
        n_fin_max_X = len(E_mus) - 1                            # Will be used in hyperbel/hypfree case as the very highest nmu

# print FC integrals
#   gs-res
print()
print('-----------------------------------------------------------------')
print("Franck-Condon overlaps between ground and resonance state")
print('n_gs  ' + 'n_res  ' + '<res|gs>')
outfile.write('\n' + '-----------------------------------------------------------------' + '\n')
outfile.write("Franck-Condon overlaps between ground and resonance state" + '\n')
outfile.write('n_gs  ' + 'n_res  ' + '<res|gs>' + '\n')

for k in range (0,n_gs_max+1):
    for l in range (0,n_res_max+1):
        FC = gs_res[k][l]
        outfile.write('{:4d}  {:5d}  {:14.10E}\n'.format(k,l,FC))
        print(('{:4d}  {:5d}  {:14.10E}'.format(k,l,FC)))

#   gs-fin
print()
print('-----------------------------------------------------------------')
print("Franck-Condon overlaps between ground and final state")
outfile.write('\n' + '-----------------------------------------------------------------' + '\n')
outfile.write("Franck-Condon overlaps between ground and final state" + '\n')
#if (fin_pot_type in ('hyperbel','hypfree')):
#    print('norm_factor =', norm_factor)
#    outfile.write('norm_factor = ' + str(norm_factor) + '\n')

print('n_gs  ' +'n_fin  ' + '<fin|gs>')
outfile.write('n_gs  ' +'n_fin  ' + '<fin|gs>' + '\n')

if (fin_pot_type in ('hyperbel','hypfree')):
    n_fin_max = n_fin_max_X
for k in range(0,n_gs_max+1):
    for m in range(0,n_fin_max+1):
        FC = gs_fin[k][m]
        outfile.write('{:4d}  {:5d}  {: 14.10E}\n'.format(k,m,FC))
        if (fin_pot_type == 'morse'):
            print(('{:4d}  {:5d}  {: 14.10E}'.format(k,m,FC)))
        elif (fin_pot_type in ('hyperbel','hypfree')):
            if (m == 0 or m == n_fin_max-1 or m == n_fin_max):      # Don't print all the FC, just the first two and last two (per GS vibr state)
                print(('{:4d}  {:5d}  {: 14.10E}'.format(k,m,FC)))
            elif (m == 1):
                print(('{:4d}  {:5d}  {: 14.10E}'.format(k,m,FC)))
                print('  ...')

#   res-fin
print()
print('-----------------------------------------------------------------')
print("Franck-Condon overlaps between final and resonance state")
print('n_res  ' +'n_fin  ' + '<fin|res>')
outfile.write('\n' + '-----------------------------------------------------------------' + '\n')
outfile.write("Franck-Condon overlaps between final and resonance state" + '\n')
outfile.write('n_res  ' +'n_fin  ' + '<fin|res>' + '\n')

for l in range(0,n_res_max+1):
    if (fin_pot_type in ('hyperbel','hypfree')):
        n_fin_max = n_fin_max_list[l]
    for m in range(0,n_fin_max+1):
        FC = res_fin[l][m]
        outfile.write('{:5d}  {:5d}  {: 14.10E}\n'.format(l,m,FC))
        if (fin_pot_type == 'morse'):
            print(('{:5d}  {:5d}  {: 14.10E}'.format(l,m,FC)))
        elif (fin_pot_type in ('hyperbel','hypfree')):
            if (m == 0 or m == n_fin_max-1 or m == n_fin_max):
                print(('{:5d}  {:5d}  {: 14.10E}'.format(l,m,FC)))
            elif (m == 1):
                print(('{:5d}  {:5d}  {: 14.10E}'.format(l,m,FC)))
                print('   ...')
if (fin_pot_type in ('hyperbel','hypfree')):
    print("All overlaps between ground or resonance state and final state\n outside the indicated quantum numbers are considered zero")
    outfile.write("All overlaps between ground or resonance state and final state\n outside the indicated quantum numbers are considered zero\n")

if partial_GamR:
    print()
    print('-----------------------------------------------------------------')
    print("Franck-Condon overlaps between final & resonance state - no V(R)")
    outfile.write('\n' + '-----------------------------------------------------------------' + '\n')
    outfile.write("Franck-Condon overlaps between final & resonance state - no V(R)" + '\n')
    print('n_res  ' +'n_fin  ' + '<fin|res>')
    outfile.write('n_res  ' +'n_fin  ' + '<fin|res>' + '\n')
    
    for l in range(0,n_res_max+1):
        if (fin_pot_type in ('hyperbel','hypfree')):
            n_fin_max = n_fin_max_list[l]
        for m in range(0,n_fin_max+1):
            FC = res_fin_woVR[l][m]
            outfile.write('{:5d}  {:5d}  {: 14.10E}\n'.format(l,m,FC))
            if (fin_pot_type == 'morse'):
                print(('{:5d}  {:5d}  {: 14.10E}'.format(l,m,FC)))
            elif (fin_pot_type in ('hyperbel','hypfree')):
                if (m == 0 or m == n_fin_max-1 or m == n_fin_max):
                    print(('{:5d}  {:5d}  {: 14.10E}'.format(l,m,FC)))
                elif (m == 1):
                    print(('{:5d}  {:5d}  {: 14.10E}'.format(l,m,FC)))
                    print('   ...')
    print('These additional overlaps without the V(R) dependence are used\n only in',
            'the prefactors to the time integrals' if (partial_GamR == 'exp') else 'the calculation of the W_lambda values')
    outfile.write('These additional overlaps without the V(R) dependence are used\n only in '
            + ('the prefactors to the time integrals' if (partial_GamR == 'exp') else 'the calculation of the W_lambda values')
            + '\n')     # If 'exp', then W_lambda is calced with Gamma(R) but they prefactors are not

# sum over mup of product <lambda|mup><mup|kappa>       where mup means mu prime
indir_FCsums = []
for l in range (0,n_res_max+1):
    indir_FCsum = 0
    factor = 1
    if (fin_pot_type in ('hyperbel','hypfree')):
        n_fin_max = n_fin_max_list[l]
    for m in range (0, n_fin_max + 1):
        if (fin_pot_type in ('hyperbel','hypfree')):            # R-DOS for 'integration' over R_mu instead of [E_]mu
            factor = R_hyp_step * E_mus[m]**2 / fin_hyp_a
        if not partial_GamR == 'exp':
            tmp = np.conj(res_fin[l][m]) * gs_fin[0][m] * factor    # <mu|lambda>* <mu|kappa=0> = <lambda|mu><mu|kappa=0> = <l|m><m|k=0>
        else:   # If Gamma(R) only in exponent (i.e. Wl), then indir_FCsums is woVR since it is part of prefactor
            tmp = np.conj(res_fin_woVR[l][m]) * gs_fin[0][m] * factor
        indir_FCsum = indir_FCsum + tmp                         # sum_m <l|m><m|k=0>
    indir_FCsums.append(indir_FCsum)                            # [sum_m <l=0|m><m|k=0>, sum_m <l=1|m><m|k=0>, ...]
print()
print('-----------------------------------------------------------------')
outfile.write('\n' + '-----------------------------------------------------------------' + '\n')

#-------------------------------------------------------------------------
# determine total decay width matrix element
print('Effective decay widths in eV and lifetimes in s:')
print('n_res  W_l [eV]          tau_l [s]')
outfile.write('Effective decay widths in eV and lifetimes in s:' + '\n')
outfile.write('n_res  W_l [eV]          tau_l [s]' + '\n')
W_lambda = []   # [W_(l=0), W_(l=1), ...]
for l in range (0,n_res_max+1):
    tmp = 0
    factor = 1
    if (fin_pot_type in ('hyperbel','hypfree')):
        n_fin_max = n_fin_max_list[l]       # To each lambda their own n_fin_max (v.s.)
    for m in range (0, n_fin_max + 1):
        if (fin_pot_type in ('hyperbel','hypfree')):
            factor = R_hyp_step * np.array(E_mus[m])**2 / fin_hyp_a
        if not partial_GamR == 'pre':
            tmp = tmp + VEr_au**2 * np.abs(res_fin[l][m])**2 * factor      # W_l = sum_m ( VEr**2 |<m|l>|**2 ) for Morse or W_l = sum_m ( DeltaR R-DOS(m) VEr**2 |<m|l>|**2 ) for cont vibr fin states
        else:
            tmp = tmp + VEr_au_woVR**2 * np.abs(res_fin_woVR[l][m])**2 * factor
    W_lambda.append(tmp)
    ttmp = 1./ (2 * np.pi * tmp)        # lifetime tau_l = 1 / (2 pi W_l)
    print(f'{l:5d}  {sciconv.hartree_to_ev(tmp):14.10E}  {sciconv.atu_to_second(ttmp):14.10E}')
    outfile.write(f'{l:5d}  {sciconv.hartree_to_ev(tmp):14.10E}  {sciconv.atu_to_second(ttmp):14.10E}\n')
print()
outfile.write('\n')


#-------------------------------------------------------------------------
in_out.check_input(Er_au, E_fin_au, Gamma_au,
                   Omega_au, TX_au, n_X, A0X,
                   omega_au, TL_au, A0L, delta_t_au,
                   tmax_au, timestep_au, E_step_au)
#-------------------------------------------------------------------------
# physical definitions of functions
# functions for the shape of the XUV pulse
if (X_sinsq):
    print('use sinsq function')
    f_t1  = lambda t1: 1./4 * ( np.exp(2j * np.pi * (t1 + TX_au/2) / TX_au) # There should be a minus sign in front [from (1/2i)**2]
                          + 2                                               # & this 2 be negative [sin**2 = (exp - exp*)**2 = exp**2 - 2 exp exp* + (exp*)**2]
                          + np.exp(-2j * np.pi * (t1 + TX_au/2) /TX_au) )
    # fp_t1 = f'(t1)
    fp_t1 = lambda t1: np.pi/(2j*TX_au) * ( - np.exp(2j*np.pi* (t1 + TX_au/2) / TX_au)  # Accordingly, these signs must be flipped
                                         + np.exp(-2j*np.pi* (t1 + TX_au/2) / TX_au) )
elif (X_gauss):
    print('use gauss function')
    f_t1  = lambda t1: ( 1./ np.sqrt(2*np.pi * sigma**2)
                       * np.exp(-t1**2 / (2*sigma**2)))
    # fp_t1 = f'(t1)
    fp_t1 = lambda t1: ( -t1 / np.sqrt(2*np.pi) / sigma**3
                       * np.exp(-t1**2 / (2*sigma**2)))
else:
    print('no pulse shape selected')

print()

if (Xshape == 'convoluted'):    # Calculate field strength EX = -(AX fX)'
    FX_t1 = lambda t1: (
                        0
                        - (A0X
                           * np.cos(Omega_au * t1)
                           * fp_t1(t1)
                          )
                        + (A0X
                           * Omega_au
                           * np.sin(Omega_au * t1)
                           * f_t1(t1)
                          )
                       )
elif (Xshape == 'infinite'):
    FX_t1 = lambda t1: + A0X * Omega_au * np.cos(Omega_au * t1)
    #FX_t1 = lambda t1: - A0X * np.sin(Omega_au * t1)
                       

#-------------------------------------------------------------------------
# technical definitions of functions (remember: FX is the field strength EX)
#direct ionization
fun_t_dir_1 = lambda t1: FX_t1(t1)   * np.exp(1j * E_fin_au * (t1-t_au)) \
                                     * np.exp(1j * E_kin_au * (t1-t_au))        # Note: before any of these fncts are called, E_fin is redefined to also include E_mu
fun_TX2_dir_1 = lambda t1: FX_t1(t1) * np.exp(1j * E_fin_au * (t1-t_au)) \
                                     * np.exp(1j * E_kin_au * (t1-t_au))        # Same as fun_t_dir_1 - why keep ?

#res_inner_fun = lambda t2: np.exp(-t2 * (np.pi * W_au + 1j*(Er_au))) \
#                           * IR_during(t2)

if (integ == 'romberg'):                                                        # numerical inner int not possible (res_inner_fun deactivated) ?
    res_inner = lambda t1: integrate.romberg(res_inner_fun, t1, t_au)
elif (integ == 'quadrature'):
    res_inner = lambda t1: integrate.quad(res_inner_fun, t1, t_au)[0]
elif (integ == 'analytic'):
    # analytic inner integral
    res_inner = lambda t1: (1./(1j*(E_kin_au + E_fin_au - Er_au - E_lambda)     # See the above note on E_fin also including E_mu
                                    - np.pi * W_au)
                            * (np.exp(t_au * (1j*(E_kin_au + E_fin_au
                                                  - Er_au - E_lambda)
                                                  - np.pi * W_au))
                              - np.exp(t1 * (1j*(E_kin_au + E_fin_au
                                                 - Er_au - E_lambda)
                                                  - np.pi * W_au)))
                            * np.exp(-1j*t_au * (E_kin_au + E_fin_au))
                           )

res_outer_fun = lambda t1: FX_t1(t1) \
                           * np.exp(t1 * (np.pi* W_au + 1j*(Er_au + E_lambda))) \
                           * res_inner(t1)


# for wavepacket in resonance state
def t_plus(t):
    return 1/(sigma*mp.sqrt(2)) * (t - 1.j*sigma**2*(Er_au+E_lambda-1.j*mp.pi*W_au+Omega_au))
def t_minus(t):
    return 1/(sigma*mp.sqrt(2)) * (t - 1.j*sigma**2*(Er_au+E_lambda-1.j*mp.pi*W_au-Omega_au))

def gamma_plus(T_up):
    return ((Er_au+E_lambda-1.j*mp.pi*W_au) * (mp.erf(t_plus(T_up)) \
                                               - mp.erf(t_plus(-TX_au/2))) \
            + 1.j/sigma * mp.sqrt(2/mp.pi) * (mp.exp(-t_plus(T_up)**2) \
                                              - mp.exp(-t_plus(-TX_au/2)**2)))
def gamma_minus(T_up):
    return ((Er_au+E_lambda-1.j*mp.pi*W_au) * (mp.erf(t_minus(T_up)) \
                                               - mp.erf(t_minus(-TX_au/2))) \
            + 1.j/sigma * mp.sqrt(2/mp.pi) * (mp.exp(-t_minus(T_up)**2) \
                                              - mp.exp(-t_minus(-TX_au/2)**2)))

def wp_res_int(t,T_up):
    return (-A0X*0.25j * mp.exp(-1.j*t*(Er_au+E_lambda-1.j*mp.pi*W_au)) \
            * (mp.exp(-sigma**2/2 * (Er_au+E_lambda-1.j*mp.pi*W_au+Omega_au)**2) \
                * gamma_plus(T_up) \
               + mp.exp(-sigma**2/2 * (Er_au+E_lambda-1.j*mp.pi*W_au-Omega_au)**2) \
                * gamma_minus(T_up)))


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# initialization
t_au = -TX_au/2


# construct list of energy points
Ekins = sciconv.hartree_to_ev(np.array(E_min_au, E_max_au, E_step_au))


#-------------------------------------------------------------------------
# constants / prefactors
prefac_dir1 = 1j * cdg_au_V
if not partial_GamR == 'exp':
    prefac_res1 = VEr_au * rdg_au / (n_res_max + 1)
    prefac_indir1 = -1j * np.pi * VEr_au**2 * cdg_au_V / (n_res_max + 1)
else:
    prefac_res1 = VEr_au_woVR * rdg_au / (n_res_max + 1)
    prefac_indir1 = -1j * np.pi * VEr_au_woVR**2 * cdg_au_V / (n_res_max + 1)

if (fin_pot_type in ('hyperbel','hypfree')):
    n_fin_max = n_fin_max_X

# for wavepacket in resonance state(s)
wp_prefs = [(1.j/(n_res_max+1) * rdg_au * gs_res[0][nlambda] \
               + mp.pi/(n_res_max+1) * VEr_au * cdg_au_V * indir_FCsums[nlambda])
            for nlambda in range(n_res_max+1)]


########################################
# now follow the integrals themselves, for the temporal phases:
# 'during the first pulse' (-TX/2, TX/2)
# 'between the pulses' (TX/2, tmax)

#-------------------------------------------------------------------------
while ((t_au <= TX_au/2) and (t_au <= tmax_au)):
#-------------------------------------------------------------------------
    outfile.write('during the first pulse \n')
    print('during the first pulse')

    outlines = []       # will contain lines containing triples of E_kin, time and signal intensity
    squares = np.array([])  # signal intensity ( = |amplitude|**2 = |J|**2 )
    E_kin_au = E_min_au
    
    t_s = sciconv.atu_to_second(t_au)
    print('t_s = ', t_s)
    outfile.write('t_s = ' + str(t_s) + '\n')
    movie_out.write('"' + format(t_s*1E15, '.3f') + ' fs' + '"' + '\n')
    cnt = 0     # initialize counter for printing progress
    if not wavepac_only: 
        while (E_kin_au <= E_max_au):
            if (cnt == 4):  # print progress: for each E_kin one '-', but for every fifth one '|' instead
                print('|', end = '', flush = True)
                cnt = 0
            else:
                print('-', end = '', flush = True)
                cnt = cnt + 1
            #print(f'{sciconv.hartree_to_ev(E_kin_au):.2} eV')           #?
            #outfile.write(f'{sciconv.hartree_to_ev(E_kin_au):.2} eV\n') #?
            p_au = np.sqrt(2*E_kin_au)
            sum_square = 0      # Total spectrum |J @ E_kin|**2 = sum_mu |J_mu @ E_kin|**2  (sum of contributions of all final states with E_kin); for continuous mu: int ~ sum
            if t_au==-TX_au/2:
                squares = np.append(squares, 0.)
                string = in_out.prep_output(0., E_kin_au, t_au)
                outlines.append(string)
                E_kin_au = E_kin_au + E_step_au
                continue
    
            for nmu in range (0, n_fin_max + 1):           # loop over all mu, calculate J_mu = J_dir,mu + J_nondir,mu
                E_fin_au = E_fin_au_1 + E_mus[nmu]      # E_fin_au_1: inputted electronic E_fin_au, E_mus: vibrational eigenvalues of fin state
        #            Er_au = Er_a_au
                
                # Direct term
                if (integ_outer == "quadrature"):
                    I1 = ci.complex_quadrature(fun_t_dir_1, (-TX_au/2), t_au)
                    dir_J1 = prefac_dir1 * I1[0] * gs_fin[0][nmu]        # [0] of quad integ result = integral (rest is est error & info); FC = <mu_n|kappa_0>
    
                elif (integ_outer == "romberg"):
                    I1 = ci.complex_romberg(fun_t_dir_1, (-TX_au/2), t_au)
                    dir_J1 = prefac_dir1 * I1 * gs_fin[0][nmu]           # romberg returns only the integral, so no [0] necessary
                 
                # J_nondir,mu = sum_lambda J_nondir,mu,lambda = sum_lambda (J_res,mu,lambda + J_indir,mu,lambda)
                J = 0
                for nlambda in range (0,n_res_max+1):
                    if (fin_pot_type in ('hyperbel','hypfree') and nmu > n_fin_max_list[nlambda]):  # J_nondir,mu,lambda = 0 if repulsive |fin>|mu> lies higher than |res>|lambda>
                        continue
                    E_lambda = E_lambdas[nlambda]
                    W_au = W_lambda[nlambda]
                    if (integ_outer == "quadrature"):
                        res_I = ci.complex_quadrature(res_outer_fun, (-TX_au/2), t_au)
        
                        if not partial_GamR == 'exp':
                            res_J1 = (prefac_res1 * res_I[0]
                                      * gs_res[0][nlambda] * res_fin[nlambda][nmu])
                            indir_J1 = (prefac_indir1 * res_I[0]
                                        * indir_FCsums[nlambda] * res_fin[nlambda][nmu])
                        else:
                            res_J1 = (prefac_res1 * res_I[0]
                                      * gs_res[0][nlambda] * res_fin_woVR[nlambda][nmu])
                            indir_J1 = (prefac_indir1 * res_I[0]
                                        * indir_FCsums[nlambda] * res_fin_woVR[nlambda][nmu])
    
                    elif (integ_outer == "romberg"):
                        res_I = ci.complex_romberg(res_outer_fun, (-TX_au/2), t_au)
                    
                        if not partial_GamR == 'exp':
                            res_J1 = (prefac_res1 * res_I
                                      * gs_res[0][nlambda] * res_fin[nlambda][nmu])
                            indir_J1 = (prefac_indir1 * res_I
                                        * indir_FCsums[nlambda] * res_fin[nlambda][nmu])
                        else:
                            res_J1 = (prefac_res1 * res_I
                                      * gs_res[0][nlambda] * res_fin_woVR[nlambda][nmu])
                            indir_J1 = (prefac_indir1 * res_I
                                        * indir_FCsums[nlambda] * res_fin_woVR[nlambda][nmu])
        
                    J = (J
                         + res_J1
                         + indir_J1
                         )
        
                # Total trs prob (@E_kin, t) = sum_mu |J_mu|**2
                # For cont rep fin: int (dE_mu |J_mu|**2 E-DOS(E_mu)) = int (dR_mu |J_mu|**2 R-DOS(R_mu))
                #   R-DOS = E-DOS * Va / R_mu**2 = E-DOS * E_mu**2 / Va. If E-DOS = 1 & R_hyp_step = const: int (dR_mu |J_mu|**2 R-DOS) ~ sum_mu (R_hyp_step |J_mu|**2 E_mu**2 / Va)
                square = np.absolute(J + dir_J1)**2     # |J_mu|**2
                if (fin_pot_type in ('hyperbel','hypfree')):
                    factor = R_hyp_step * E_mus[nmu]**2 / fin_hyp_a
                    old_square = square
                    square = square * factor
                sum_square = sum_square + square        # |J|**2 = sum_mu |J_mu|**2
                #print(f'nmu = {nmu:>3}  f = {factor:.5f}  osq = {old_square:.5E}  sq = {square:.5E}  sum = {sum_square:.5E}')
                #outfile.write(f'nmu = {nmu:>3}  f = {factor:.5f}  osq = {old_square:.5E}  sq = {square:.5E}  sum = {sum_square:.5E}\n')
    
            squares = np.append(squares, sum_square)
    
            string = in_out.prep_output(sum_square, E_kin_au, t_au)     # returns str: E_kin_eV, t_s, sum_square = intensity
            outlines.append(string)
            
            E_kin_au = E_kin_au + E_step_au     # @ t = const.
        
        
        in_out.doout_1f(pure_out, outlines)     # writes each (E_kin, t = const, |J|**2) triple in a sep line into output file
        in_out.doout_movie(movie_out, outlines)
        print()
        max_pos = argrelextrema(squares, np.greater)[0]      # finds position of relative (i. e. local) maxima of |J|**2 in an array
        if (len(max_pos > 0)):                               # if there are such:
            for i in range (0, len(max_pos)):
                print(Ekins[max_pos[i]], squares[max_pos[i]])      # print all loc max & resp E_kin
                outfile.write(str(Ekins[max_pos[i]]) + '  ' + str(squares[max_pos[i]]) + '\n')
    
    # wavepacket in resonance state(s)
    wp_ampls = []
    for nlambda in range (0,n_res_max+1):
        E_lambda = E_lambdas[nlambda]
        W_au = W_lambda[nlambda]
        wp_I = wp_res_int(t_au,t_au)
        wp_pref = wp_prefs[nlambda] 
        wp_ampl = wp_pref * wp_I
        wp_string = format(nlambda, 'd') + '   ' + format(sciconv.atu_to_second(t_au), ' .18f') \
                + '   ' + format(complex(wp_ampl), ' .15e')
        wp_ampls.append(wp_string)
    in_out.doout_1f(wp_res_out, wp_ampls)


    t_au = t_au + timestep_au



#-------------------------------------------------------------------------
while (t_au >= TX_au/2\
#        and (t_au <= (delta_t_au - a))\
        and (t_au <= tmax_au)):
#-------------------------------------------------------------------------
    outfile.write('between the pulses \n')
    print('between the pulses')

    # all equal to during-1st-pulse section, except for integrating over entire XUV pulse now

    outlines = []       # will contain lines containing triples of E_kin, time and signal intensity
    squares = np.array([])  # signal intensity ( = |amplitude|**2 = |J|**2 )
    E_kin_au = E_min_au
    
    t_s = sciconv.atu_to_second(t_au)
    print('t_s = ', t_s)
    outfile.write('t_s = ' + str(t_s) + '\n')
    movie_out.write('"' + format(t_s*1E15, '.3f') + ' fs' + '"' + '\n')
    cnt = 0     # initialize counter for printing progress
    if not wavepac_only: 
        while (E_kin_au <= E_max_au):
            if (cnt == 4):  # print progress: for each E_kin one '-', but for every fifth one '|' instead
                print('|', end = '', flush = True)
                cnt = 0
            else:
                print('-', end = '', flush = True)
                cnt = cnt + 1
            #print(f'{sciconv.hartree_to_ev(E_kin_au):.2} eV')           #?
            #outfile.write(f'{sciconv.hartree_to_ev(E_kin_au):.2} eV\n') #?
            p_au = np.sqrt(2*E_kin_au)
            sum_square = 0      # Total spectrum |J @ E_kin|**2 = sum_mu |J_mu @ E_kin|**2  (sum of contributions of all final states with E_kin)
    
            for nmu in range (0, n_fin_max + 1):           # loop over all mu, calculate J_mu = J_dir,mu + J_nondir,mu
                E_fin_au = E_fin_au_1 + E_mus[nmu]      # E_fin_au_1: inputted electronic E_fin_au, E_mus: vibrational eigenvalues of fin state
        #            Er_au = Er_a_au
                
                # Direct term
                if (integ_outer == "quadrature"):
                    I1 = ci.complex_quadrature(fun_t_dir_1, (-TX_au/2), TX_au/2)
                    dir_J1 = prefac_dir1 * I1[0] * gs_fin[0][nmu]        # [0] of quad integ result = integral (rest is est error & info); FC = <mu_n|kappa_0>
    #                    print(nmu, gs_fin[0][nmu], dir_J1)   #?
        
                elif (integ_outer == "romberg"):
                    I1 = ci.complex_romberg(fun_t_dir_1, (-TX_au/2), TX_au/2)
                    dir_J1 = prefac_dir1 * I1 * gs_fin[0][nmu]           # romberg returns only the integral, so no [0] necessary
    
                # J_nondir,mu = sum_lambda J_nondir,mu,lambda = sum_lambda (J_res,mu,lambda + J_indir,mu,lambda)
                J = 0
                for nlambda in range (0,n_res_max+1):
                    if (fin_pot_type in ('hyperbel','hypfree') and nmu > n_fin_max_list[nlambda]):  # J_nondir,mu,lambda = 0 if repulsive |fin>|mu> lies higher than |res>|lambda>
    #                    print(nmu, nlambda, 'skipped')  #?
                        continue
                    E_lambda = E_lambdas[nlambda]
                    W_au = W_lambda[nlambda]
                    if (integ_outer == "quadrature"):
                        res_I = ci.complex_quadrature(res_outer_fun, (-TX_au/2), TX_au/2)
                        
                        if not partial_GamR == 'exp':
                            res_J1 = (prefac_res1 * res_I[0]
                                      * gs_res[0][nlambda] * res_fin[nlambda][nmu])
                            indir_J1 = (prefac_indir1 * res_I[0]
                                        * indir_FCsums[nlambda] * res_fin[nlambda][nmu])
    #                        print(nmu, nlambda, 'res_J1 =', res_J1, 'indir_J1 =', indir_J1)   #?
                        else:
                            res_J1 = (prefac_res1 * res_I[0]
                                      * gs_res[0][nlambda] * res_fin_woVR[nlambda][nmu])
                            indir_J1 = (prefac_indir1 * res_I[0]
                                        * indir_FCsums[nlambda] * res_fin_woVR[nlambda][nmu])
    #                        print(nmu, nlambda, 'res_J1 =', res_J1, 'indir_J1 =', indir_J1)   #?
        
                    elif (integ_outer == "romberg"):
                        res_I = ci.complex_romberg(res_outer_fun, (-TX_au/2), TX_au/2)
                        
                        if not partial_GamR == 'exp':
                            res_J1 = (prefac_res1 * res_I
                                      * gs_res[0][nlambda] * res_fin[nlambda][nmu])
                            indir_J1 = (prefac_indir1 * res_I
                                        * indir_FCsums[nlambda] * res_fin[nlambda][nmu])
                        else:
                            res_J1 = (prefac_res1 * res_I
                                      * gs_res[0][nlambda] * res_fin_woVR[nlambda][nmu])
                            indir_J1 = (prefac_indir1 * res_I
                                        * indir_FCsums[nlambda] * res_fin_woVR[nlambda][nmu])
    
    
                    J = (J
                         + res_J1
                         + indir_J1
                         )
        
                # Total trs prob (@E_kin, t) = sum_mu |J_mu|**2
                # For cont rep fin: int (dE_mu |J_mu|**2 E-DOS(E_mu)) = int (dR_mu |J_mu|**2 R-DOS(R_mu))
                #   R-DOS = E-DOS * Va / R_mu**2 = E-DOS * E_mu**2 / Va. If E-DOS = 1 & R_hyp_step = const: int (dR_mu |J_mu|**2 R-DOS) ~ sum_mu (R_hyp_step |J_mu|**2 E_mu**2 / Va)
                square = np.absolute(J + dir_J1)**2     # |J_mu|**2
                if (fin_pot_type in ('hyperbel','hypfree')):
                    factor = R_hyp_step * E_mus[nmu]**2 / fin_hyp_a
                    old_square = square
                    square = square * factor
                sum_square = sum_square + square        # |J|**2 = sum_mu |J_mu|**2
                #print(f'nmu = {nmu:>3}  f = {factor:.5f}  osq = {old_square:.5E}  sq = {square:.5E}  sum = {sum_square:.5E}')
                #outfile.write(f'nmu = {nmu:>3}  f = {factor:.5f}  osq = {old_square:.5E}  sq = {square:.5E}  sum = {sum_square:.5E}\n')
    
            squares = np.append(squares, sum_square)
    
            string = in_out.prep_output(sum_square, E_kin_au, t_au)     # returns str: E_kin_eV, t_s, sum_square = intensity
            outlines.append(string)
            
            E_kin_au = E_kin_au + E_step_au     # @ t = const.
        
        
        in_out.doout_1f(pure_out, outlines)     # writes each (E_kin, t = const, |J|**2) triple in a sep line into output file
        in_out.doout_movie(movie_out, outlines)
        print()
        max_pos = argrelextrema(squares, np.greater)[0]      # finds position of relative (i. e. local) maxima of |J|**2 in an array
        if (len(max_pos > 0)):                               # if there are such:
            for i in range (0, len(max_pos)):
                print(Ekins[max_pos[i]], squares[max_pos[i]])      # print all loc max & resp E_kin
                outfile.write(str(Ekins[max_pos[i]]) + '  ' + str(squares[max_pos[i]]) + '\n')
    
    # wavepacket in resonance state(s)
    wp_ampls = []
    for nlambda in range (0,n_res_max+1):
        E_lambda = E_lambdas[nlambda]
        W_au = W_lambda[nlambda]
        wp_I = wp_res_int(t_au,TX_au/2)
        wp_pref = wp_prefs[nlambda] 
        wp_ampl = wp_pref * wp_I
        wp_string = format(nlambda, 'd') + '   ' + format(sciconv.atu_to_second(t_au), ' .18f') \
                + '   ' + format(complex(wp_ampl), ' .15e')
        wp_ampls.append(wp_string)
    in_out.doout_1f(wp_res_out, wp_ampls)


    t_au = t_au + timestep_au




print('In order to process the wavepacket results, consider running res_wavepacket.py')

dt_end = datetime.now()
print('Total runtime:', str(dt_end - dt_start))
print(str(dt_end))
outfile.write('\n' + str(dt_end) + '\n')
outfile.write('Total runtime:' + ' ' + str(dt_end - dt_start))

outfile.close
pure_out.close
movie_out.close
wp_res_out.close
