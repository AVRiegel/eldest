##########################################################################
#                     INPUT AND OUTPUT ROUTINES                          #
##########################################################################
# Purpose:                                                               #
#          - A python module handling input and output.                  #
#                                                                        #
##########################################################################
# written by: Elke Fasshauer May 2018                                    #
##########################################################################

import sciconv
import numpy as np
import sys


#-------------------------------------------------------------------------
#   input
def read_input(inputfile, outfile):
#-------------------------------------------------------------------------
    # define default parameters
    # transitions dipole moments
    rdg_au       = 0.3            # transition dipole moment into the resonance state
    cdg_au       = 0.9            # transition dipole moment into any continuum state
    q            = 1
    # parameters of the investigated system
    # the ground state (vibrational gs of electronic gs) energy is being defined as EG = 0
    Er_a_eV       =  150.0        # resonance energy (at potential minimum) in eV 
    Er_b_eV       =    0.0
    tau_a_s       =  0.0
    tau_b_s       =  0.0
    E_fin_eV      =  70.0         # final state energy (at potential minimum) in eV
    tau_s         =  4.0E-16      # lifetime in s
    E_fin_eV_2    =  0.0
    tau_s_2       =  4.0E-16
    interact_eV   =  0.0
    #
    # laser parameters
    Omega_eV      = 150.0         # mean photon energy of the XUV pulse in eV
    n_X           = 5             # number of cycles within the XUV pulse
    I_X           = 1.0E12        # intensity of the XUV pulse in W/cm^2
    X_shape       = "gauss"       # options: gauss, sinsq
    Xshape        = "convoluted"  # options: convoluted, infinite
    #
    # dressing laser parameters
    omega_eV      = 1.6           # IR pulse
    n_L           = 10
    I_L           = 1.0E12
    Lshape        = "sinsq"
    delta_t_s     = 0.0E-18
    shift_step_s  = 500.0E-18
    phi           = 0
    FWHM_L       = 500E-18
    #
    # parameters of the simulation
    tmax_s        = 2.5E-15          # simulate until time tmax in s
    timestep_s    = 0.5E-16          # evaluate expression every timestep_s seconds
    #
    E_min_eV      =  30.0
    E_max_eV      =  50.0
    E_step_eV     =  1.00            # energy difference between different evaluated electron kinetic energies
    #
    integ         = "analytic"       # options: analytic, (quadrature, romberg - both currently unavailable)
    integ_outer   = "quadrature"     # options: quadrature, romberg
    Gamma_type    = "const"          # options: const, R6, external
    #
    fc_precalc    = False            # use file with pre-calculated "Franck-Condon overlap integrals" for gs-fin and res-fin, flag -f
    partial_GamR  = None             # options: None, pre, exp (i. e. use Gamma-of-R dependence everywhere / only in prefactors / only in Wl)
    part_fc_pre   = False            # use file with pre-calculated FC overlap integrals without Gamma-of-R dependence, flag -F
    wavepac_only  = False            # calculate only the resonance-state projections, skip final-state projections'
    #
    # parameters for the nuclear dynamics
    mass1         = 20.1797       # in g/mol
    mass2         = 20.1797       # in g/mol
    grad_delta    = 0.001
    R_eq_AA       = 3.08
    # vibrational states parameters  # provide everything in au (de in Hartree, a in inverse Bohr, Req in Bohr)
    # ground-state parameters
    gs_de      = 0
    gs_a       = 0
    gs_Req     = 0
    gs_const   = 47.6930
    # resonance-state parameters
    res_a         = 0.0183747        # for morse: res_de; for hyperbel or hypfree: V_a in au (Hartree * Bohr)                                                    
    res_b         = 15.3994          # for morse: res_a; for hyperbel or hypfree: V_b in au (Hartree)
    res_c         = 6.0              # for morse: res_Req; for hyperbel or hypfree: step width for R_start for vibrational energies (res_a/R_start) in au (Bohr)
    res_d         = 0.0              # for morse: res_const; for hyperbel or hypfree: FC factor threshold for calculating the trs integral
    res_pot_type  = 'morse'          # options: morse, hyperbel, hypfree
    # final-state parameters
    fin_a      = -15.869110       # for morse: fin_de; for hyperbel or hypfree: V_a in au (Hartree * Bohr)
    fin_b      = 1.659155         # for morse: fin_a; for hyperbel or hypfree: V_b in au (Hartree)
    fin_c      = 75.293906        # for morse: fin_Req; for hyperbel or hypfree: step width for R_start for vibrational energies (fin_a/R_start) in au (Bohr)
    fin_d      = 47.6930          # for morse: fin_const; for hyperbel or hypfree: FC factor threshold for calculating the trs integral
    fin_pot_type  = 'morse'       # options: morse, hyperbel, hypfree
#-------------------------------------------------------------------------

    f = open(inputfile, 'r')
    
    lines = f.readlines()
    
    for line in lines:
        words = line.split()
        if (words[0] == 'rdg_au'):
            rdg_au = float(words[2])
            print('rdg_au = ', rdg_au)
            outfile.write('rdg_au = ' + str(rdg_au) + '\n')
        elif (words[0] == 'cdg_au'):
            cdg_au = float(words[2])
            print('cdg_au = ', cdg_au)
            outfile.write('cdg_au = ' + str(cdg_au) + '\n')
    
    # energy parameters of the system
        elif (words[0] == 'Er_a_eV'):
            Er_a_eV = float(words[2])
            print('Er_a_eV = ', Er_a_eV)
            outfile.write('Er_a_eV = ' + str(Er_a_eV) + '\n')
        elif (words[0] == 'Er_b_eV'):
            Er_b_eV = float(words[2])
            print('Er_b_eV = ', Er_b_eV)
            outfile.write('Er_b_eV = ' + str(Er_b_eV) + '\n')
        elif (words[0] == 'E_fin_eV'):
            E_fin_eV = float(words[2])
            print('E_fin_eV = ', E_fin_eV)
            outfile.write('E_fin_eV = ' + str(E_fin_eV) + '\n')
        elif (words[0] == 'tau_s'):
            tau_s = float(words[2])
            print('tau_s = ', tau_s)
            outfile.write('tau_s = ' + str(tau_s) + '\n')
        elif (words[0] == 'E_fin_eV_2'):
            E_fin_eV_2 = float(words[2])
            print('E_fin_eV_2 = ', E_fin_eV_2)
            outfile.write('E_fin_eV_2 = ' + str(E_fin_eV_2) + '\n')
        elif (words[0] == 'tau_a_s'):
            tau_a_s = float(words[2])
            print('tau_a_s = ', tau_a_s)
            outfile.write('tau_a_s = ' + str(tau_a_s) + '\n')
        elif (words[0] == 'tau_b_s'):
            tau_b_s = float(words[2])
            print('tau_b_s = ', tau_b_s)
            outfile.write('tau_b_s = ' + str(tau_b_s) + '\n')
        elif (words[0] == 'tau_s_2'):
            tau_s_2 = float(words[2])
            print('tau_s_2 = ', tau_s_2)
            outfile.write('tau_s_2 = ' + str(tau_s_2) + '\n')
        elif (words[0] == 'interact_eV'):
            interact_eV = float(words[2])
            print('interact_eV = ', interact_eV)
            outfile.write('interact_eV = ' + str(interact_eV) + '\n')
    
    # exciting laser parameters
        elif (words[0] == 'Omega_eV'):
            Omega_eV = float(words[2])
            print('Omega_eV = ', Omega_eV)
            outfile.write('Omega_eV = ' + str(Omega_eV) + '\n')
        elif (words[0] == 'n_X'):
            n_X = float(words[2])
            print('n_X = ', n_X)
            outfile.write('n_X = ' + str(n_X) + '\n')
        elif (words[0] == 'I_X'):
            I_X = float(words[2])
            print('I_X = ', I_X)
            outfile.write('I_X = ' + str(I_X) + '\n')
        elif (words[0] == 'X_shape'):
            if (words[2] == 'sinsq'):
                X_sinsq = True
                X_gauss = False
                print('X_shape = Sin**2')
                outfile.write('X_shape = Sin**2 \n')
            elif (words[2] == 'gauss'):
                X_sinsq = False
                X_gauss = True
                print('X_shape = Gauss')
                outfile.write('X_shape = Gauss \n')
            else:
                print('no XUV pulse shape selected')
                outfile.write('no XUV pulse shape selected \n')
        elif (words[0] == 'Xshape'):
            if (words[2] == 'infinite'):
                Xshape = 'infinite'
                outfile.write('Infinite XUV pulse selected \n')
                print('Infinite XUV pulse selected')
            elif (words[2] == 'convoluted'):
                Xshape = 'convoluted'
                outfile.write('Convoluted XUV pulse selected \n')
                print('Convoluted XUV pulse selected')
    
    # dressing laser parameters
        elif (words[0] == 'omega_eV'):
            omega_eV = float(words[2])
            print('omega_eV = ', omega_eV)
            outfile.write('omega_eV = ' + str(omega_eV) + '\n')
        elif (words[0] == 'n_L'):
            n_L = float(words[2])
            print('n_L = ', n_L)
            outfile.write('n_L = ' + str(n_L) + '\n')
        elif (words[0] == 'I_L'):
            I_L = float(words[2])
            print('I_L = ', I_L)
            outfile.write('I_L = ' + str(I_L) + '\n')
        elif (words[0] == 'Lshape'):
            if (words[2] == 'gauss'):
                Lshape = 'gauss'
                outfile.write('Gaussian shaped IR pulse selected \n')
                print('Gaussian shaped IR pulse selected')
            elif (words[2] == 'sinsq'):
                Lshape = 'sinsq'
                outfile.write('Sinsq shaped IR pulse selected \n')
                print('Sinsq shaped IR pulse selected')
        elif (words[0] == 'delta_t_s'):
            delta_t_s = float(words[2])
            print('delta_t_s = ', delta_t_s)
            outfile.write('delta_t_s = ' + str(delta_t_s) + '\n')
        elif (words[0] == 'shift_step_s'):
            shift_step_s = float(words[2])
            print('shift_step_s = ', shift_step_s)
            outfile.write('shift_step_s = ' + str(shift_step_s) + '\n')
        elif (words[0] == 'FWHM_L'):
            FWHM_L = float(words[2])
            print('FWHM_L_s = ', FWHM_L)
            outfile.write('FWHM_L = ' + str(FWHM_L) + '\n')
        elif (words[0] == 'phi'):
            phi = float(words[2])
            print('phi = ', phi)
            outfile.write('phi = ' + str(phi) + '\n')
        elif (words[0] == 'q'):
            q = float(words[2])
            print('q = ', q)
            outfile.write('q = ' + str(q) + '\n')
    
    # parameters of the simulation
        elif (words[0] == 'tmax_s'):
            tmax_s = float(words[2])
            print('tmax_s = ', tmax_s)
            outfile.write('tmax_s = ' + str(tmax_s) + '\n')
        elif (words[0] == 'timestep_s'):
            timestep_s = float(words[2])
            print('timestep_s = ', timestep_s)
            outfile.write('timestep_s = ' + str(timestep_s) + '\n')
        elif (words[0] == 'E_step_eV'):
            E_step_eV = float(words[2])
            print('E_step_eV = ', E_step_eV)
            outfile.write('E_step_eV = ' + str(E_step_eV) + '\n')
    
        elif (words[0] == 'E_min_eV'):
            E_min_eV = float(words[2])
            print('E_min_eV = ', E_min_eV)
            outfile.write('E_min_eV = ' + str(E_min_eV) + '\n')
        elif (words[0] == 'E_max_eV'):
            E_max_eV = float(words[2])
            print('E_max_eV = ', E_max_eV)
            outfile.write('E_max_eV = ' + str(E_max_eV) + '\n')

        elif (words[0] == 'integ'):
            if (words[2] == 'romberg'):
                integ = 'romberg'
                print('Integration Scheme of the inner integral = Romberg')
                outfile.write('Integration Scheme of the inner integral = Romberg \n')
            elif (words[2] == 'quadrature'):
                integ = 'quadrature'
                print('Integration Scheme of the inner integral = Gaussian Quadrature')
                outfile.write('Integration Scheme of the inner integral = Gaussian Quadrature \n')
            elif (words[2] == 'analytic'):
                integ = 'analytic'
                print('Integration Scheme of the inner integral = analytic')
                outfile.write('Integration Scheme of the inner integral = analytic \n')
            else:
                print('no integration scheme selected')
                outfile.write('no integration scheme selected \n')
            
        elif (words[0] == 'integ_outer'):
            if (words[2] == 'romberg'):
                integ_outer = 'romberg'
                print('Integration Scheme of the outer integral = Romberg')
                outfile.write('Integration Scheme of the outer integral = Romberg \n')
            elif (words[2] == 'quadrature'):
                integ_outer = 'quadrature'
                print('Integration Scheme of the outer integral = Gaussian Quadrature')
                outfile.write('Integration Scheme of the outer integral = Gaussian Quadrature \n')
            else:
                print('no integration scheme selected')
                outfile.write('no integration scheme selected \n')

        elif (words[0] == 'Gamma_type'):
            if (words[2] == 'const'):
                Gamma_type = 'const'
                print('Dependence of Gamma on R: constant')
                outfile.write('Dependence of Gamma on R = constant \n')
            elif (words[2] == 'R6'):
                Gamma_type = 'R6'
                print('Dependence of Gamma on R: R^(-6)')
                outfile.write('Dependence of Gamma on R: R^(-6) \n')
            elif (words[2] == 'exp'):
                Gamma_type = 'exp'
                print('Dependence of Gamma on R: e^(-aR)')
                outfile.write('Dependence of Gamma on R: e^(-aR) \n')
            elif (words[2] == 'external'):
                Gamma_type = 'external'
                print('Dependence of Gamma on R: provided by external file')
                outfile.write('Dependence of Gamma on R: provided by external file \n')
            else:
                print('no Gamma type selected')
                outfile.write('no Gamma type selected \n')

        elif (words[0] == 'fc_precalc'):
            fc_precalc = True if words[2].lower() == 'true' else False
        elif (words[0] == 'partial_GamR'):
            if (words[2] in ('pre', 'prefactor')):
                partial_GamR = 'pre'
                print('Gamma(R) dependence used only in overlap integrals in transition-amplitude prefactors')
                outfile.write('Gamma(R) dependence used only in overlap integrals in transition-amplitude prefactors\n')
            elif (words[2] in ('exp', 'exponent', 'Wl')):
                partial_GamR = 'exp'
                print('Gamma(R) dependence used only in W_lambda in the transition-amplitude exponents')
                outfile.write('Gamma(R) dependence used only in W_lambda in the transition-amplitude exponents\n')
            else:
                partial_GamR = None
        elif (words[0] == 'part_fc_pre'):
            part_fc_pre = True if words[2].lower() == 'true' else False
        elif (words[0] == 'wavepac_only'):
            wavepac_only = True if words[2].lower() == 'true' else False

    # vibrational-states parameters
        elif (words[0] == 'gs_de'):
            outfile.write('Parameters of potential energy curves:' + '\n')
            gs_de = float(words[2])
            outfile.write('gs_de = ' + str(gs_de) + '\n')
        elif (words[0] == 'gs_a'):
            gs_a = float(words[2])
            outfile.write('gs_a = ' + str(gs_a) + '\n')
        elif (words[0] == 'gs_Req'):
            gs_Req = float(words[2])
            outfile.write('gs_Req = ' + str(gs_Req) + '\n')
        elif (words[0] == 'gs_const'):
            gs_const = float(words[2])
            outfile.write('gs_const = ' + str(gs_const) + '\n')
        elif (words[0] == 'res_a'):
            res_a = float(words[2])
            outfile.write('res_a = ' + str(res_a) + '\n')
        elif (words[0] == 'res_b'):
            res_b = float(words[2])
            outfile.write('res_b = ' + str(res_b) + '\n')
        elif (words[0] == 'res_c'):
            res_c = float(words[2])
            outfile.write('res_c = ' + str(res_c) + '\n')
        elif (words[0] == 'res_d'):
            res_d = float(words[2])
            outfile.write('res_d = ' + str(res_d) + '\n')
        elif (words[0] == 'res_pot_type'):
            res_pot_type = str(words[2])
            outfile.write('res_pot_type = ' + str(res_pot_type) + '\n')
            if (res_pot_type not in ['morse','hyperbel', 'hypfree']):
                print('Non-existent resonance-state-potential type chosen, QUIT')
                sys.exit()
        elif (words[0] == 'fin_a'):
            fin_a = float(words[2])
            outfile.write('fin_a = ' + str(fin_a) + '\n')
        elif (words[0] == 'fin_b'):
            fin_b = float(words[2])
            outfile.write('fin_b = ' + str(fin_b) + '\n')
        elif (words[0] == 'fin_c'):
            fin_c = float(words[2])
            outfile.write('fin_c = ' + str(fin_c) + '\n')
        elif (words[0] == 'fin_d'):
            fin_d = float(words[2])
            outfile.write('fin_d = ' + str(fin_d) + '\n')
        elif (words[0] == 'fin_pot_type'):
            fin_pot_type = str(words[2])
            outfile.write('fin_pot_type = ' + str(fin_pot_type) + '\n')
            if (fin_pot_type not in ['morse','hyperbel', 'hypfree']):
                print('Non-existent final-state-potential type chosen, QUIT')
                sys.exit()
    
    f.close()
    return (rdg_au, cdg_au,
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
            res_a, res_b, res_c, res_d, res_pot_type,
            fin_a, fin_b, fin_c, fin_d, fin_pot_type
            )

#-------------------------------------------------------------------------
#   input
def read_input_old(inputfile, outfile):
#-------------------------------------------------------------------------
    # define default parameters
    rdg_au       = 0.3
    cdg_au       = 0.9
    q            = 1
    # parameters of the investigated system
    # the ground state energy is being defined as Eg = 0
    Er_a_eV       =  150.0        # resonance energy in eV
    Er_b_eV       =    0.0
    tau_a_s       =  0.0
    tau_b_s       =  0.0
    E_fin_eV      =  70.0         # final state energy in eV
    tau_s         =  4.0E-16      # lifetime
    E_fin_eV_2    =  0.0
    tau_s_2       =  4.0E-16
    interact_eV   =  0.0
    #
    # laser parameters
    Omega_eV      = 150.0         #
    n_X           = 5
    I_X           = 1.0E12        # intensity of the XUV pulse in W/cm^2
    X_shape       = "gauss"       # options: gauss, sinsq
    Xshape        = "convoluted"
    #
    # dressing laser parameters
    omega_eV      = 1.6           # IR pulse
    n_L           = 10
    I_L           = 1.0E12        # intensity of the IR pulse in W/cm^2
    Lshape        = "sinsq"
    delta_t_s     = 0.0E-18       # time difference between the maxima of the two pulses
    shift_step_s  = 500.0E-18
    phi           = 0
    FWHM_L       = 500E-18
    #
    # parameters of the simulation
    tmax_s        = 2.5E-15       # simulate until time tmax in seconds
    timestep_s    = 0.5E-16     # evaluate expression every timestep_s seconds 
    E_step_eV     = 1.00          # energy difference between different evaluated Omegas
    #
    E_min_eV      =  30.0
    E_max_eV      =  50.0
    #
    integ         = "analytic"
    integ_outer   = "romberg"
    # parameters for the nuclear dynamics
    mass1         = 20.1797 #in g/mol
    mass2         = 20.1797 # in g/mol
    grad_delta    = 0.001
    R_eq_AA       = 3.08
#flat test
    #V_RICD_in_a   = 0
    #V_RICD_in_b   = 0
    #V_RICD_in_c   = 0
    #V_RICD_in_d   = 47.6930
#ungerade initial state
    V_RICD_in_a   = -33.179112
    V_RICD_in_b   = 1.930064
    V_RICD_in_c   = 37.757254
    V_RICD_in_d   = 47.6930
#gerade initial state
    #V_RICD_in_a   = -15.869110
    #V_RICD_in_b   = 1.659155
    #V_RICD_in_c   = 75.293906
    #V_RICD_in_d   = 47.6930
    V_fin_RICD_a  = 13.915571
    V_fin_RICD_b  = 42.516162
    V_ICD_in_a    = -33.179112
    V_ICD_in_b    = 1.930064
    V_ICD_in_c    = 37.757254
    V_ICD_in_d    = 48.4750
    V_fin_ICD_a   = 13.915571
    V_fin_ICD_b   = 42.516162
    V_fin_ICD_b   = 43.298162
#-------------------------------------------------------------------------

    f = open(inputfile, 'r')
    
    lines = f.readlines()
    
    for line in lines:
        words = line.split()
        if (words[0] == 'rdg_au'):
            rdg_au = float(words[2])
            print('rdg_au = ', rdg_au)
            outfile.write('rdg_au = ' + str(rdg_au) + '\n')
        elif (words[0] == 'cdg_au'):
            cdg_au = float(words[2])
            print('cdg_au = ', cdg_au)
            outfile.write('cdg_au = ' + str(cdg_au) + '\n')
    
    # energy parameters of the system
        elif (words[0] == 'Er_a_eV'):
            Er_a_eV = float(words[2])
            print('Er_a_eV = ', Er_a_eV)
            outfile.write('Er_a_eV = ' + str(Er_a_eV) + '\n')
        elif (words[0] == 'Er_b_eV'):
            Er_b_eV = float(words[2])
            print('Er_b_eV = ', Er_b_eV)
            outfile.write('Er_b_eV = ' + str(Er_b_eV) + '\n')
        elif (words[0] == 'E_fin_eV'):
            E_fin_eV = float(words[2])
            print('E_fin_eV = ', E_fin_eV)
            outfile.write('E_fin_eV = ' + str(E_fin_eV) + '\n')
        elif (words[0] == 'tau_s'):
            tau_s = float(words[2])
            print('tau_s = ', tau_s)
            outfile.write('tau_s = ' + str(tau_s) + '\n')
        elif (words[0] == 'E_fin_eV_2'):
            E_fin_eV_2 = float(words[2])
            print('E_fin_eV_2 = ', E_fin_eV_2)
            outfile.write('E_fin_eV_2 = ' + str(E_fin_eV_2) + '\n')
        elif (words[0] == 'tau_a_s'):
            tau_a_s = float(words[2])
            print('tau_a_s = ', tau_a_s)
            outfile.write('tau_a_s = ' + str(tau_a_s) + '\n')
        elif (words[0] == 'tau_b_s'):
            tau_b_s = float(words[2])
            print('tau_b_s = ', tau_b_s)
            outfile.write('tau_b_s = ' + str(tau_b_s) + '\n')
        elif (words[0] == 'tau_s_2'):
            tau_s_2 = float(words[2])
            print('tau_s_2 = ', tau_s_2)
            outfile.write('tau_s_2 = ' + str(tau_s_2) + '\n')
        elif (words[0] == 'interact_eV'):
            interact_eV = float(words[2])
            print('interact_eV = ', interact_eV)
            outfile.write('interact_eV = ' + str(interact_eV) + '\n')
    
    # exciting laser parameters
        elif (words[0] == 'Omega_eV'):
            Omega_eV = float(words[2])
            print('Omega_eV = ', Omega_eV)
            outfile.write('Omega_eV = ' + str(Omega_eV) + '\n')
        elif (words[0] == 'n_X'):
            n_X = float(words[2])
            print('n_X = ', n_X)
            outfile.write('n_X = ' + str(n_X) + '\n')
        elif (words[0] == 'I_X'):
            I_X = float(words[2])
            print('I_X = ', I_X)
            outfile.write('I_X = ' + str(I_X) + '\n')
        elif (words[0] == 'X_shape'):
            if (words[2] == 'sinsq'):
                X_sinsq = True
                X_gauss = False
                print('X_shape = Sin**2')
                outfile.write('X_shape = Sin**2 \n')
            elif (words[2] == 'gauss'):
                X_sinsq = False
                X_gauss = True
                print('X_shape = Gauss')
                outfile.write('X_shape = Gauss \n')
            else:
                print('no XUV pulse shape selected')
                outfile.write('no XUV pulse shape selected \n')
        elif (words[0] == 'Xshape'):
            if (words[2] == 'infinite'):
                Xshape = 'infinite'
                outfile.write('Infinite XUV pulse selected \n')
                print('Infinite XUV pulse selected')
            elif (words[2] == 'convoluted'):
                Xshape = 'convoluted'
                outfile.write('Convoluted XUV pulse selected \n')
                print('Convoluted XUV pulse selected')
    
    # dressing laser parameters
        elif (words[0] == 'omega_eV'):
            omega_eV = float(words[2])
            print('omega_eV = ', omega_eV)
            outfile.write('omega_eV = ' + str(omega_eV) + '\n')
        elif (words[0] == 'n_L'):
            n_L = float(words[2])
            print('n_L = ', n_L)
            outfile.write('n_L = ' + str(n_L) + '\n')
        elif (words[0] == 'I_L'):
            I_L = float(words[2])
            print('I_L = ', I_L)
            outfile.write('I_L = ' + str(I_L) + '\n')
        elif (words[0] == 'Lshape'):
            if (words[2] == 'gauss'):
                Lshape = 'gauss'
                outfile.write('Gaussian shaped IR pulse selected \n')
                print('Gaussian shaped IR pulse selected')
            elif (words[2] == 'sinsq'):
                Lshape = 'sinsq'
                outfile.write('Sinsq shaped IR pulse selected \n')
                print('Sinsq shaped IR pulse selected')
        elif (words[0] == 'delta_t_s'):
            delta_t_s = float(words[2])
            print('delta_t_s = ', delta_t_s)
            outfile.write('delta_t_s = ' + str(delta_t_s) + '\n')
        elif (words[0] == 'shift_step_s'):
            shift_step_s = float(words[2])
            print('shift_step_s = ', shift_step_s)
            outfile.write('shift_step_s = ' + str(shift_step_s) + '\n')
        elif (words[0] == 'FWHM_L'):
            FWHM_L = float(words[2])
            print('FWHM_L_s = ', FWHM_L)
            outfile.write('FWHM_L = ' + str(FWHM_L) + '\n')
        elif (words[0] == 'phi'):
            phi = float(words[2])
            print('phi = ', phi)
            outfile.write('phi = ' + str(phi) + '\n')
        elif (words[0] == 'q'):
            q = float(words[2])
            print('q = ', q)
            outfile.write('q = ' + str(q) + '\n')
    
    # parameters of the simulation
        elif (words[0] == 'tmax_s'):
            tmax_s = float(words[2])
            print('tmax_s = ', tmax_s)
            outfile.write('tmax_s = ' + str(tmax_s) + '\n')
        elif (words[0] == 'timestep_s'):
            timestep_s = float(words[2])
            print('timestep_s = ', timestep_s)
            outfile.write('timestep_s = ' + str(timestep_s) + '\n')
        elif (words[0] == 'E_step_eV'):
            E_step_eV = float(words[2])
            print('E_step_eV = ', E_step_eV)
            outfile.write('E_step_eV = ' + str(E_step_eV) + '\n')
    
        elif (words[0] == 'E_min_eV'):
            E_min_eV = float(words[2])
            print('E_min_eV = ', E_min_eV)
            outfile.write('E_min_eV = ' + str(E_min_eV) + '\n')
        elif (words[0] == 'E_max_eV'):
            E_max_eV = float(words[2])
            print('E_max_eV = ', E_max_eV)
            outfile.write('E_max_eV = ' + str(E_max_eV) + '\n')

        elif (words[0] == 'integ'):
            if (words[2] == 'romberg'):
                integ = 'romberg'
                print('Integration Scheme of the inner integral = Romberg')
                outfile.write('Integration Scheme of the inner integral = Romberg \n')
            elif (words[2] == 'quadrature'):
                integ = 'quadrature'
                print('Integration Scheme of the inner integral = Gaussian Quadrature')
                outfile.write('Integration Scheme of the inner integral = Gaussian Quadrature \n')
            elif (words[2] == 'analytic'):
                integ = 'analytic'
                print('Integration Scheme of the inner integral = analytic')
                outfile.write('Integration Scheme of the inner integral = analytic \n')
            else:
                print('no integration scheme selected')
                outfile.write('no integration scheme selected \n')
            
        elif (words[0] == 'integ_outer'):
            if (words[2] == 'romberg'):
                integ_outer = 'romberg'
                print('Integration Scheme of the outer integral = Romberg')
                outfile.write('Integration Scheme of the outer integral = Romberg \n')
            elif (words[2] == 'quadrature'):
                integ_outer = 'quadrature'
                print('Integration Scheme of the outer integral = Gaussian Quadrature')
                outfile.write('Integration Scheme of the outer integral = Gaussian Quadrature \n')
            else:
                print('no integration scheme selected')
                outfile.write('no integration scheme selected \n')
    
    f.close()
    return (rdg_au, cdg_au,
            Er_a_eV, Er_b_eV, tau_a_s, tau_b_s, E_fin_eV, tau_s, E_fin_eV_2, tau_s_2,
            interact_eV,
            Omega_eV, n_X, I_X, X_sinsq, X_gauss, Xshape,
            omega_eV, n_L, I_L, Lshape, delta_t_s, shift_step_s, phi, q, FWHM_L,
            tmax_s, timestep_s, E_step_eV,
            E_min_eV, E_max_eV,
            integ, integ_outer,
            mass1, mass2, grad_delta, R_eq_AA,
            V_RICD_in_a, V_RICD_in_b, V_RICD_in_c, V_RICD_in_d,
            V_fin_RICD_a, V_fin_RICD_b,
            V_ICD_in_a, V_ICD_in_b, V_ICD_in_c, V_ICD_in_d,
            V_fin_ICD_a, V_fin_ICD_b
            )

#-------------------------------------------------------------------------
def check_input(Er, E_fin, Gamma,
                Omega, TX, n_X, A0X,
                omega, TL, A0L, delta_t,
                tmax, timestep, E_step):
    print('Input Check')

    if (E_fin > Omega):
        exit('Warning: E_fin > Omega' + '\n'
             + 'Stopping Script')

    print('Input fulfills requirements')

    return 0
    

#-------------------------------------------------------------------------
# input of Franck-Condon overlap integrals including final states.

def read_fc_input(inputfile):
    # Python 3 compatibility hack
    try:
        unicode('')
    except NameError:
        unicode = str


    state = 'pre_gs-res'       # Encodes where we are in the input file
    prev_n = 0      # The quantum number of gs in gs-res / gs-fin or of res in res-fin in the previous line
    gs_res = [[]]
    gs_fin = [[]]
    res_fin = [[]]

    with open(inputfile, 'r') as f:
        lines = f.readlines()

    lines_iter = iter(lines)
    for line in lines_iter:
        if state == 'pre_gs-res':
            if line.startswith("Franck-Condon overlaps between ground and resonance state"):
                state = 'gs-res'
                next(lines_iter)
                continue
            else:
                continue

        words = line.split()
        if len(words) == 0:
            continue

        if unicode(words[0]).isnumeric():
            if state == 'pre_gs-fin':
                state = 'gs-fin'
            elif state == 'pre_res-fin':
                state = 'res-fin'
        else:
            if state == 'gs-res':
                state = 'pre_gs-fin'
                prev_n = 0
            elif state == 'gs-fin':
                state = 'pre_res-fin'
                prev_n = 0
            elif state == 'res-fin':
                break
            continue

        if state == 'gs-res':
            fcs = gs_res
        elif state == 'gs-fin':
            fcs = gs_fin
        elif state == 'res-fin':
            fcs = res_fin
        else:
            sys.exit('The FC integral read-in situation has developed not necessarily to our advantage.')

        if prev_n != int(words[0]):
            prev_n = int(words[0])
            fcs.append(list())
        fcs[prev_n].append(complex(words[-1]))

    n_fin_max_list = []             # Max quantum number considered in non-direct ionization for each lambda (all vibr fin states above the resp res state are discarded)
    for l in res_fin:
        n_fin_max_list.append(len(l) - 1)
    n_fin_max_X = len(gs_fin[0]) - 1                            # Will be used in hyperbel/hypfree case as the very highest nmu

    return (gs_res, gs_fin, res_fin, n_fin_max_list, n_fin_max_X)





#-------------------------------------------------------------------------
#   output
#   output
def prep_output(I, Omega_au, t_au):
#    square = np.absolute(I)**2
    #print I
    Omega_eV = sciconv.hartree_to_ev(Omega_au)
    t_s = sciconv.atu_to_second(t_au)
    string = format(Omega_eV, '>8.5f') + '   ' + format(t_s, ' .18f') + '   ' + format(I, '.15e')
    #string = str(Omega_eV) + '   ' + format(t_s, '.18f') + '   ' + format(I, '.15e')
    return string

def prep_output_comp(I, I2, Omega_au, t_au):
#    square = np.absolute(I)**2
    #print I
    Omega_eV = sciconv.hartree_to_ev(Omega_au)
    t_s = sciconv.atu_to_second(t_au)
    string = str(Omega_eV) + '   ' + format(t_s, '.18f') + '   ' + format(I, '.15e') + '   ' + format(I2, '.15e')
    return string

def prep_output_t(point, time_au):
    time_s = sciconv.atu_to_second(time_au)
    point_R = np.real(point)
    point_I = np.imag(point)
    string = format(time_s, '.18f') + '   ' + str(point_I)
    return string

def prep_output_int(I, Omega_au, integer):
#    square = np.absolute(I)**2
    #print I
    Omega_eV = sciconv.hartree_to_ev(Omega_au)
    string = str(Omega_eV) + '   ' + str(integer) + '   ' + format(I, '.15e')
    return string

def doout(t_au, outlines):
    # output filename will give the time in ps
    t_s = sciconv.atu_to_second(t_au)
    t_ps = t_s * 1E12
    if t_ps < 0.0:
        t_ps = np.absolute(t_ps)
        filename = 'm' + format(t_ps, '.8f') + '.dat'
    else:
        filename = format(t_ps, '.8f') + '.dat'
    outfile = open(filename, mode='w')
    res_lines = '\n'.join(outlines)
    outfile.write(res_lines)
    outfile.close

def doout_1f(filename, outlines):
    res_lines = '\n'.join(outlines)
    res_lines = res_lines + '\n' + '' + '\n'
    filename.write(res_lines)
    #outfile.close

def doout_movie(filename, outlines):
    res_lines = '\n'.join(outlines)
    res_lines = res_lines + '\n' + '' + '\n' + '' + '\n'
    filename.write(res_lines)
