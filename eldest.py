#!/usr/bin/python

##########################################################################
#                                    ELDEST                              #
#        Investigating Electronic Decay Processes with Streaking         #
##########################################################################
# Purpose:                                                               #
#          - A program to simulate the streaking process of electronic   #
#            decay processes.                                            #
#                                                                        #
##########################################################################
# written by: Elke Fasshauer May 2018                                    #
##########################################################################

import scipy
import scipy.integrate as integrate
#from numpy import sqrt, sin, cos, pi, absolute, exp
import numpy as np
#from sympy import *
#from mpmath import *


#-------------------------------------------------------------------------
# Input parameters

rdg           = 0.5           # transition dipole moment into the resonant state
cdg           = 0.5           # transition dipole moment into any continuum state

# parameters of the investigated system
# the ground state energy is being defined as Eg = 0
Er_eV         = 14.0          # resonance energy in eV
E_kin_eV      = 2.0           # kinetic energy of secondary electron
E_fin_eV      = 12.0          # final state energy in eV

Gamma_eV      = 0.5           # electronic decay width of the resonant state

# laser parameters
Omega_min_eV  = 10.0          # scanning XUV pulse from Omega_min-eV to
Omega_max_eV  = 18.0          #
TX_s          = 100E-18       # duration of the XUV pulse in seconds
A0X           = 1.0           # amplitude of the XUV pulse

omega_eV      = 1.0           # IR pulse
TL_s          = 1.0E-12       # duration of the IR streaking pulse
A0L           = 1.0           # amplitude of the IR pulse
delta_t_s     = 1.0E-15       # time difference between the maxima of the two pulses

# parameters of the simulation
tmax_s        = 1.0E-10       # simulate until time tmax in seconds
timestep_s    = 10E-18        # evaluate expression every timestep_s seconds 
Omega_step_eV = 0.1           # energy difference between different evaluated Omegas


print 'Hello World'
Omega= 13.5

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = integrate.quad(real_func, a, b, **kwargs)
    imag_integral = integrate.quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:],
            imag_integral[1:])

def complex_double_quadrature(func1, func2, a, b, c, d, **kwargs):
    def real_f1(x):
        return scipy.real(func1(x))
    def real_f2(y):
        return scipy.real(func2(y))
    def imag_f1(x):
        return scipy.imag(func1(x))
    def imag_f2(y):
        return scipy.imag(func2(y))

    first_real = lambda y,x: scipy.real(func1(x)) * scipy.real(func2(y))
    sec_real   = lambda y,x: - scipy.imag(func1(x)) * scipy.imag(func2(y))
    first_imag = lambda y,x: scipy.imag(func1(x)) * scipy.real(func2(y))
    sec_imag   = lambda y,x: scipy.real(func1(x)) * scipy.imag(func2(y))
    
    first_real_integral = integrate.dblquad(first_real, a, b, c, d, **kwargs)
    sec_real_integral   = integrate.dblquad(sec_real, a, b, c, d, **kwargs)
    first_imag_integral = integrate.dblquad(first_imag, a, b, c, d, **kwargs)
    sec_imag_integral   = integrate.dblquad(sec_imag, a, b, c, d, **kwargs)
    return (first_real_integral[0] + sec_real_integral[0]
            + 1j*first_imag_integral[0] + 1j*sec_imag_integral[0],
            first_real_integral[1:])

f  = lambda t1: 1./4 (np.exp(2j*np.pi*t1/TX) + 2 + np.exp(-2j*np.pi*t1/TX) )
fp = lambda t1: np.pi/(2j*TX) * (-np.exp(2j*np.pi*t1/TX) + np.exp(-2j*np.pi*t1/TX) )
FX = lambda t1: - A0X * np.cos(Omega * t1) * fp + A0X * Omega * np.sin(Omega * t1) * f

## very important: The first Variable in the definition of the function marks the inner
## integral, while the second marks the outer integral.
## If any limit is replaced by the integration variable of the outer integral,
## this is always specified as x, never as the actual name of the variable.
f = lambda t2, t1: np.exp(t1 * complex(Gamma_eV/2,Er_eV)) * np.exp(t2 * complex(Gamma_eV/2, Er_eV + E_kin_eV))
##f = lambda t2, t1: np.exp(t1 * Gamma_eV/2) * np.exp(t2 * Gamma_eV/2)
#


f = lambda x: x
g = lambda y: np.exp(y)

h = lambda y,x: f(x) * g(y)

#I = integrate.dblquad(h, -TX_s/2, TX_s/2, lambda x: x, lambda x: TX_s/2)
I = integrate.dblquad(h, 0, 1, lambda x: x, lambda x: 1)
#
#x,y = symbols("x y")
#f = x**2 * y
# integration using sympy instead of scipy
#I = integrate(f, (x, 0, 2), (y, 0, x)) 
#

#I = complex_quadrature(f,0,1)
#J = complex_quadrature(g,0,1)

print "I"
print I

#print J

fun1 = lambda t1: np.exp(t1 * complex(Gamma_eV/2,Er_eV))
fun2 = lambda t2: np.exp(t2 * complex(Gamma_eV/2, Er_eV + E_kin_eV))

K = complex_double_quadrature(fun1,fun2, -TX_s/2, TX_s/2, lambda x: x, lambda x: TX_s/2)
#K = complex_double_quadrature(f,g, 0, 1, lambda x: x, lambda x: 1)

print "K"
print K
