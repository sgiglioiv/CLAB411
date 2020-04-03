#!/home/thomas/psi4conda/bin/python
"""
This is the code I'm trying rewrite completely fully sourced. And working at least as well as the old code.
"""
"""
Jiri Horacek's regularized analytic continuation
see JCP 143, 184102 (2015)
"""


import argparse
import matplotlib.pyplot as plt
import numpy as np
import rach_utils as racu
from scipy.optimize import minimize


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
    mrk = arguments.prin  #pr
    data = np.lib.loadtxt(arguments.fname)
    lambdas = data[:, 0]  #ls
    energys = data[:, 1:]  #Es
    for i in range(len(lambdas)):
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
        racu.pregraph(n_ene, lambdas, energys, 'raw data', print_lev=3, mrk=mrk)
        for i in range(n_ene):
            energys[:,i] -= lambdas * au2eV
        racu.pregraph(n_ene, lambdas, energys, 'after the shift', print_lev=3, mrk=mrk)

    rac(lambdas, energys, (Emin, Emax), (Er, Ei), mrk)


def rac(lambdas, energys, e_range, guess, mrk):
    """
    Main driver of the R-ACCC procedure

    PARAMETERS
    −−−−−−−−−−
    1. lambdas : object
                List of lambda terms
    2. energys : object
                List of parameters
    3. e_range : list
                List from Emin to Emax (Emin, Emax)
    4. guess : list
                List of guesses for resonance energy and width (Er, Ei)
    5. print_lev : int
                An integer value that determines the level of printing

    RETURNS
    -------
    6. None
        Returns the calculations as print outputs
    """
    global kappas, g_lambdas, sigmas
    g_lambdas, temp_energys = racu.select_energies(lambdas, energys, e_range, mrk)
    if len(g_lambdas) == 0:
        print("No input selected")
        return

    sigmas = np.ones(len(energys))
    """  
    Jiri's 2015 notation: E = -kappa^2     
    """
    kappas = np.sqrt(-temp_energys)

    """
    make a guess for the parameters in eq. (2)
    to do so eq.(5) must be inverted
    """

    res_en = guess[0]
    res_wid = 2 * guess[1]
    alpha_param = 0.5 * np.sqrt(2.0 ) * (-2 * res_en + np.sqrt(4 * res_en ** 2 +
                                                               res_wid ** 2)) ** 0.25
    beta_param = 0.5 * res_wid / np.sqrt(-2 * res_en + np.sqrt(4 * res_en ** 2 +
                                                               res_wid ** 2))
    lambda_param = g_lambdas[0] / racu.pade_21(kappas[0], [alpha_param, beta_param, 1],
                                             [])
    point_plot = 200
    kap_points = np.linspace(0, kappas[-1], point_plot)
    lam_points = np.zeros(point_plot)
    params = [alpha_param, beta_param, lambda_param]
    con_params = ["pade_21"]
    racu.graph(kappas, kap_points, g_lambdas, lam_points, params, con_params,
               point_plot, print_lev=4, mrk=mrk)

    """
    fit a [2,1] Pade approximant
    """
    racu.display("------------------------------------------------", print_lev=2,
                 mrk=mrk)
    params = [alpha_param, beta_param, lambda_param]
    con_params = ["pade_21"]
    racu.display("[2,1] Guess a, b, l =  " + str(params), print_lev=2, mrk=mrk)
    res = minimize(racu.chi, params, args=(kappas, g_lambdas, sigmas, con_params),
                   method='BFGS', options={'gtol': 1e-6})
    params = res.x
    racu.display("[2,1] Opt   a, b, l =  " + str(params), print_lev=2, mrk=mrk)
    chi2 = racu.chi2s(params, kappas, g_lambdas, sigmas, con_params)
    racu.display("[2,1] chi^2 " + str(chi2), print_lev=2, mrk=mrk)
    chi2 = racu.get_chis()
    print(chi2)

if __name__ == "__main__":
    main()