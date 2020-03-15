#!/home/thomas/psi4conda/bin/python

"""
Jiri Horacek's Regularized Analytic Continuation
See JCP 143, 184102 (2015)
"""
from typing import Any, Union
import matplotlib.pyplot as plt
import numpy as np


Ers = []
Gs = []
chis = []


def decorator_print(orginal_function):
    """
    This function wraps another function to give it functionality to print or not
    """
    def wrapper_function(*args, print_lev):
        if mrk <= print_lev:
            return origingal_function(*args)
    return wrapper_function

##############################3
def select_energies(ls, Es, E_range, pr):
    """ 
    select all energies in the first column that lie within E_range
    """
    n_ene = numpy.shape(Es)[1]
    n_ls = numpy.shape(ls)[0]
    print ("No. of lambda-points: ", n_ls)
    E_min = E_range[0]
    E_max = E_range[1]
    print ("Input energy range: {0:.4f} to {1:.4f}".format(E_min, E_max)) 
    if E_max > 0 : 
        print ("Upper energy range is", E_max)
        print ("Only negative energies may be used.")
        print ("Upper range is reset to 0")
        E_max = 0
    """ find upper limit """
    for i in range(n_ls):
        i_max = i
        if Es[i,0] < E_max:
            break
    """ find lower limit """
    for i in range(n_ls):
        i_min = n_ls - 1 - i
        if Es[i_min,0] > E_min:
            break

    print ("Largest  energy used is {0:.4f} (index {1:d})".format(Es[i_max,0],i_max))
    print ("Smallest energy used is {0:.4f} (index {1:d})".format(Es[i_min,0],i_min))
    Ebs = Es[i_max:i_min+1,0] 
    lbs = ls[i_max:i_min+1]
    print ("No. of points used:", len(lbs))

    if pr > 3:
        for i in range(n_ene):
            plt.plot(ls, Es[:,i], marker='d', color='orange')
        plt.plot(lbs, Ebs, marker='o', color='blue')
        plt.title('raw data')
        plt.show()

    return lbs, Ebs
##############################################33


def pade_21(kappas, params, con_params):
    """
    This function calculates the [2/1] Pade approximant
    Pade [2/1] eq.(2)

    PARAMETERS
    −−−−−−−−−−
    1. kappas : object
                List of kappa terms
    2. params : object
                List of parameters
    3. con_params : list
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [2/1] approximant at the corresponding
               kappa terms and parameters
    """
    alpha_param = params[]
    beta_param = params[]
    lambda_param = params[]
    a4b2 = alpha_param * alpha_param * alpha_param * alpha_param + beta_param * beta_param
    aak2 = alpha_param * alpha_param * kappas * 2
    return lambda_param * (kappas * kappas + aak2 + a4b2) / (a4b2 + aak2)


def pade_31(kappas, params, con_params):
    """
    This function calculates the [3/1] Pade approximant
    Pade [3/1] eq.(9) corrected by missing factor 2 in a^2k terms

    PARAMETERS
    −−−−−−−−−−
    1. kappas : object
                    List of kappa terms
    2. params : object
                    List of parameters
    3. con_params : list
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [3/1] approximant at the corresponding
               kappa terms and parameters
    """
    alpha_param = params[]
    beta_param = params[]
    delta_param = params[]
    lambda_param = params[]
    a4b2 = alpha_param * alpha_param * alpha_param * alpha_param + beta_param * beta_param
    aak2 = alpha_param * alpha_param * kappas * 2
    ddk = delta_param * delta_param * kappas
    num = (kappas * kappas + aak2 + a4b2) * (1 + ddk)
    den = a4b2 + aak2 + ddk * a4b2
    return lambda_param * num / den


def pade_32(kappas, params, con_params):
    """
    This function calculates the [3/2] Pade approximant
    Pade [3/2] as given in 2016 correction paper

    PARAMETERS
    −−−−−−−−−−
    1. kappas : object
                    List of kappa terms
    2. params : object
                    List of parameters
    3. con_params : list
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [3/2] approximant at the corresponding
               kappa terms and parameters
    """
    alpha_param = params[]
    beta_param = params[]
    delta_param = params[]
    epsilon_param = params[]
    lambda_param = params[]
    a4b2 = alpha_param * alpha_param * alpha_param * alpha_param + beta_param * beta_param
    aak2 = alpha_param * alpha_param * kappas * 2
    ddk = delta_param * delta_param * kappas
    num = (kappas * kappas + aak2 + a4b2) * (1 + ddk)
    den = a4b2 + aak2 + ddk * a4b2 + epsilon_param * epsilon_param * kappas * kappas
    return lambda_param * num / den


def pade_42(kappas, params, con_params):
    """
    This function calculates the [4/2] Pade approximant
    Pade [4/2] in eqs. (10) and (11) had several typos

    PARAMETERS
    −−−−−−−−−−
    1. kappas : object
                    List of kappa terms
    2. params : object
                    List of parameters
    3. con_params : list
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [4/2] approximant at the corresponding
               kappa terms and parameters
    """
    alpha_param = params[]
    beta_param = params[]
    delta_param = params[]
    gamma_param = params[]
    lambda_param = params[]
    omega_param = params[]
    a4b2 = alpha_param * alpha_param * alpha_param * alpha_param + beta_param * beta_param
    g4d2 = gamma_param * gamma_param * gamma_param * gamma_param + delta_param * delta_param
    ta2 = 2 * alpha_param * alpha_param
    tg2 = 2 * gamma_param * gamma_param
    k2 = kappas * kappas
    mu2 = ta2 * g4d2 + tg2 * a4b2
    num = (k2 + ta2 * kappas + a4b2) * (k2 + tg2 * kappas + g4d2)
    den = a4b2 * g4d2 + mu2 * kappas + omega_param * omega_param * k2
    return lambda_param * num / den


def pade_52(kappas, params, con_params):
    """
    This function calculates the [5/2] Pade approximant
    Pade [5/2] based on notes from Roman Curik 1/24/2020 & 1/30/2020

    PARAMETERS
    −−−−−−−−−−
    1. kappas : object
                    List of kappa terms
    2. params : object
                    List of parameters
    3. con_params : list
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [5/2] approximant at the corresponding
               kappa terms and parameters
    """
    alpha_param = params[]
    beta_param = params[]
    delta_param = params[]
    epsilon_param = params[]
    gamma_param = params[]
    lambda_param = params[]
    omega_param = params[]
    a4b2 = alpha_param * alpha_param * alpha_param * alpha_param + beta_param * beta_param
    g4d2 = gamma_param * gamma_param * gamma_param * gamma_param + delta_param * delta_param
    ta2 = 2 * alpha_param * alpha_param
    tg2 = 2 * gamma_param * gamma_param
    k2 = kappas * kappas
    mu2 = ta2 * g4d2 + tg2 * a4b2 + epsilon_param * epsilon_param * a4b2 * g4d2
    num = (k2 + ta2 * kappas + a4b2) * (k2 + tg2 * kappas + g4d2) * (1 +
                                                                     epsilon_param *
                                                                     epsilon_param *
                                                                     kappas)
    den = a4b2 * g4d2 + mu2 * kappas + omega_param * omega_param * k2
    return lambda_param * num / den


def pade_53(kappas, params, con_params):
    """
    This function calculates the [5/3] Pade approximant
    Pade [5/3] based on notes from Roman Curik 1/24/2020 & 1/30/2020

    PARAMETERS
    −−−−−−−−−−
    1. kappas : object
                    List of kappa terms
    2. params : object
                    List of parameters
    3. con_params : list
                    List of paramters to be constant with the first term being the Pade approximant used

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [5/3] approximant at the corresponding
               kappa terms and parameters
    """
    alpha_param = params[]
    beta_param = params[]
    delta_param = params[]
    epsilon_param = params[]
    zeta_param = params[]
    gamma_param = params[]
    lambda_param = params[]
    omega_param = params[]
    a4b2 = alpha_param * alpha_param * alpha_param * alpha_param + beta_param * beta_param
    g4d2 = gamma_param * gamma_param * gamma_param * gamma_param + delta_param * delta_param
    ta2 = 2 * alpha_param * alpha_param
    tg2 = 2 * gamma_param * gamma_param
    k2 = kappas * kappas
    mu2 = ta2 * g4d2 + tg2 * a4b2 + epsilon_param * epsilon_param * a4b2 * g4d2
    num = (k2 + ta2 * kappas + a4b2) * (k2 + tg2 * kappas + g4d2) * (1 +
                                                                     epsilon_param *
                                                                     epsilon_param *
                                                                     kappas)
    den = a4b2 * g4d2 + mu2 * kappas + zeta_param * zeta_param * k2 + omega_param * \
          omega_param * kappas * k2
    return lambda_param * num / den


funcs={
    """
    Dictionary of the functions to be used
    """
       "pade_21": pade_21,
       "pade_31": pade_31,
       "pade_32": pade_32,
       "pade_42": pade_42,
       "pade_52": pade_52,
       "pade_53": pade_53
       }


def chi(params, kappas, lbs, sigmas, con_params):
    """
         This function calculates the chi^2 value based on eq. (6)
         See JCP 143, 184102 (2015)

    PARAMETERS
    −−−−−−−−−−
    1. params : object
                List of parameters
    2. kappas : object
                List of kappa values
    3. lambas : object
                List of lambda values
    4. sigmas : object
                Used to calculate the weights
    5. con_params : list
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used

    RETURNS
    -------
    6. chi2/len(kappas) : float
                          This corresponds to the chi^2 value from eq. (6)
    """
    fn = funcs[con_params[0]]
    chi2 = 0
    weights = 1 / sigmas**2
    for i in range(len(kappas)):
        chi2 += ((lambdas[i] - fn(kappas[i], params, con_params[1:])) * weights[i])**2
    return (chi2/len(kappas))


@decorator_print
def display(orignial_function):
    def wrapper_function(*args, pr):
        if mrk <= pr:
            return original_function(*args)
    return wrapper_function

@decorater_print
def pregraph(n_ene, lambdas, energys, msg):
    for i in range(n_ene):
        plt.plot(lambdas, energys[:,i], marker= 'd', color= 'orange')
    plt.title(msg)
    plt.show()

@decorator_print
def graph(kappas, kap_points, lambdas, lam_points, params, con_params, point_plot, pr):
    """
        This function creates a graph of kappa values versus the lambda values for
        use in determining if the program is function correctly

    PARAMETERS
    −−−−−−−−−−
    1. kappas : object
                List of kappas
    2. kap_points : object
                    Number of points used to make the curve seem smooth
    3. lambdas : object
                 List of lambdas
    4. lam_points : object
                    Number of points used to make the curve seem smotth
    5. params : object
                List of parameters
    6. con_params : list
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used
    7. point_plot : object
                    The number of points used to plot the respective graphs
    8. pr : print level
            probably take this one out

    RETURNS
    -------
    9. None, but does produce graphs
    """
    fn = funcs[con_params[0]]
    plt.plot(kappas, lambdas, marker='o', color='blue')
        for i in range(point_plot):
            lam_points[i]=fn(kap_points[i], params, con_params[1:])
        plt.plot(kappas, lambdas, marker='', color="orange")
        plt.xlabel('kappa')
        plt.ylabel('lambda')
        plt.title(cs[0])
        plt.show()


def res_eng(alpha_param, beta_param):
    """
    This function calculates the resonance energies and associated gamma values based
    on eq. (5)
    See JCP 143, 184102 (2015)

    PARAMETERS
    −−−−−−−−−−
    1. alpha_param : object
                     The optimized alpha parameter
    2. beta_param : object
                    The optimized beta parameter

    RETURNS
    -------
    3. None, but appends the list of calculated resonance energies (Ers) and gammas (Gs)
    """
    Er = beta_param ** 2 - alpha_param ** 4
    G = 4 * alpha_param ** 2 * beta_param
    Ers.append(Er)
    Gs.append(G)


def getErs():
    """
    This function only returns the list of resonance energies

    PARAMETERS
    −−−−−−−−−−
    1. None

    RETURNS
    -------
    4. The list of resonance energies
    """
    return Ers


def getGs():
    """
    This function only returns the list of gammas

    PARAMETERS
    −−−−−−−−−−
    1. None

    RETURNS
    -------
    4. The list of gammas
    """
    return Gs

"""
def getchis():
    ""
    This function only returns the list of gammas

    PARAMETERS
    −−−−−−−−−−
    1. None

    RETURNS
    -------
    4. The list of gammas
    ""
    return chis
"""
