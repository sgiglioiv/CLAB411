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
chi2 = []


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
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used

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
    3. lbs : object
             not sure what these are
    4. sigmas : object
                again not sure
    5. con_params : list
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used

    RETURNS
    -------
    6. lamda : float
               The evaluation of the Pade [5/3] approximant at the corresponding
               kappa terms and parameters
    """
    fn = funcs[con_params[0]]
    chi2 = 0
    weights = 1 / sigmas**2
    for i in range(len(kappas)):
        chi2 += ((lbs[i] - fn(kappas[i], params, con_params[1:])) * weights[i])**2
    return (chi2/len(kappas))


def graph(kappas, kps, lbs, lps, params, con_params, nplt, pr):
    """
        This function creates a graph of kappa values versus the lambda values for
        use in determining if the program is function correctly

    PARAMETERS
    −−−−−−−−−−
    1. kappas : object
                List of kappas
    2. kps : object
                not sure
    3. lbs : object
             not sure what these are
    4. lps : object
                again not sure
    5. params : object
                List of parameters
    6. con_params : list
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used
    7. nplt : object
              no idea
    8. pr : print level
            probably take this one out

    RETURNS
    -------
    9. None, but does produce graphs
    """
    fn = funcs[con_params[0]]
    if pr > 4:
        plt.plot(kappas, lbs, marker='o', color='blue')
        for i in range(nplt):
            lps[i]=fn(kps[i], params, con_params[1:])
        plt.plot(kps, lps, marker='', color="orange")
        plt.xlabel('kappa')
        plt.ylabel('lambda')
        #plt.title(cs[0])
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