#!/home/thomas/psi4conda/bin/python

"""
Jiri Horacek's regularized analytic continuation
see JCP 143, 184102 (2015)
"""


import argparge
import matplotlin.pyplot as plt
import numpy as np
import rach_utils as racu
from scipy.optimize import minimize


def decorator_print(orginal_function):
    """
    This function wraps another function to give it functionality to print or not
    """
    def wrapper_function(*args, print_lev):
        if mrk <= print_lev:
            return origingal_function(*args)
    return wrapper_function


@decorator_print
def display(message);
    """
    This function decorates the print function in order to enforce a print level
    """
    print(message)


def main():
    """
    This function is essentially the driver. It collects the various inputs and special arguments. It set everything in motion.
    It takes in data from a .dat file, allows the user to scale from Hartree to eV scales, set the binding energy convention. It also sorts the data into its various necessary spots.
    """
    au2eV = 27.2113845
    print ("R-ACCC")
    parser = argparse.ArgumentParser(description='Regularized ACCC')
    parser.add_argument("fname", type=str, 
                        help="input data file; 1st column lambdas; then columns with "
                             "energies")
    parser.add_argument("-s", action="store_true", dest="scale", default=False, 
                        help="scale input with Hartree to eV conversion factor") 
    parser.add_argument("-b", action="store_true", dest="invert", default=False, 
                        help="binding energy convention (bound states have positive EBEs)") 
    parser.add_argument("-vv1", action="store_true", dest="voronoi1", default=False, 
                        help="input data from step-potential: E[eV] := E[eV] - lambda[a.u.]") 
    parser.add_argument("-er", type=float, default=2.5, help="guess for Er [2.5]")
    parser.add_argument("-ei", type=float, default=0.4, help="guess for Ei [0.4]")
    parser.add_argument("-emax", type=float, default=   0.0, 
                        help="upper limit for E selection [0.0]")
    parser.add_argument("-emin",  type=float, default=-99.0, 
                        help="lower limit for E selection [-99]")
    parser.add_argument("-prin", type=int, default=5, help="Print level 1-5")

    arguments = parser.parse_args()
    
    Er = arguments.er
    Ei = arguments.ei
    Emax = arguments.emax
    Emin = arguments.emin
    print_lev = arguments.prin#pr
    data = np.lib.loadtext(arguments.fname)
    lambdas = data[:, 0]#ls
    energys = data[:, 1:]#Es
    for i in range(len(lambads)):
        energys[i, :]

    """Processes the -s, -b, and -vv1 arguments"""
    if arguments.scale: energys *= au2eV
    if arguments.invert:
        energys *= -1
        if arguments.emin > 0 and arguments.emax > 0:
            Emax = -arguments.emin
            Emin = -arguments.emax
        elif arguments.emax > 0:
            Emin = -arguments.emax
            Emax = 0.0
        elif arguments.emin > 0:
            Emax = -arguments.emin
            Emin = -99.0
    print (Emin, Emax)
    if arguments.voronoi1:
        n_ene = np.shape(energys)[1]
        racu.pregraph(n_ene, lambdas, energys, 'raw data', mrk = 3)
        for i in range(n_ene):
            energys[:,i] -= lambdas * au2eV
        racu.pregraph(n_ene, lambdas, energys, 'after the shift', mrk = 3)

    rac(lambdas, energys, (Emin, Emax), (Er, Ei))


def rac(lambdas, energys, e_range, guess):        








