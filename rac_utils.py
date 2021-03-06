#!/home/thomas/psi4conda/bin/python
"""
Current running version of the code
"""
"""
Jiri Horacek's regularized analytic continuation
see JCP 143, 184102 (2015)
"""

import numpy as np
import matplotlib.pyplot as plt

Ers = []
Gs = []
chis = []

def pade_21(k, ps, cs):
    """
    This function calculates the [2/1] Pade approximant
    Pade [2/1] eq.(2)

    PARAMETERS
    −−−−−−−−−−
    1. k : object
                List of kappa terms
    2. ps : object
                List of parameters
                *This function optimizes the following parameters*
                                                                alpha(a)
                                                                beta(b)
                                                                lambda(l)
    3. cs : list
                List of parameters to be kept constant with the first term being
                the Pade approximant used
                *This function holds no parameters constant*

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [2/1] approximant at the corresponding
               kappa terms and parameters
    """
    l = ps[0]
    a = ps[1]
    b = ps[2]
    a4b2=a*a*a*a + b*b
    aak2=a*a*k*2
    return l*(k*k + aak2 + a4b2) / (a4b2 + aak2)
    
def pade_31(k, ps, cs):
    """
    This function calculates the [3/1] Pade approximant
    Pade [3/1] eq.(9) corrected by missing factor 2 in a^2k terms

    PARAMETERS
    −−−−−−−−−−
    1. k : object
                    List of kappa terms
    2. ps : object
                    List of parameters
                    *This function optimizes the following parameters*
                                                                alpha(a)
                                                                beta(b)
                                                                delta(d)
                                                                lambda(l)
    3. cs : list
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used
                    *This function holds no parameters constant*

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [3/1] approximant at the corresponding
               kappa terms and parameters
    """
    d = ps[0]
    l = ps[1]
    a = ps[2]
    b = ps[3]
    a4b2=a*a*a*a + b*b
    aak2=a*a*k*2
    ddk=d*d*k
    num = (k*k + aak2 + a4b2) * (1 + ddk)
    den = a4b2 + aak2 + ddk*a4b2
    return l * num / den

def ppade_32(k, ps, cs):
    """
    This function calculates the pre-fit for the [3/2] Pade approximant
    Pade [3/2] as given in 2016 correction paper

    PARAMETERS
    −−−−−−−−−−
    1. k : object
                List of kappa terms
    2. ps : object
                List of parameters
                *This function optimizes the following parameters*
                                                                delta(d)
                                                                epsilon(e)
                                                                lambda(l)
    3. cs : list
                List of parameters to be kept constant with the first term being
                the Pade approximant used
                *This function holds the following parameters constant*
                                                                    alpha(a)
                                                                    beta(b)

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [3/2] approximant at the corresponding
               kappa terms and parameters
    """
    l = ps[0]
    d = ps[1]
    e = ps[2]
    a = cs[0]
    b = cs[1]
    a4b2 = a * a * a * a + b * b
    aak2 = a * a * k * 2
    ddk = d * d * k
    num = (k * k + aak2 + a4b2) * (1 + ddk)
    den = a4b2 + aak2 + ddk * a4b2 + e * e * k * k
    return l * num / den

def pade_32(k, ps, cs):
    """
    This function calculates the [3/2] Pade approximant
    Pade [3/2] as given in 2016 correction paper

    PARAMETERS
    −−−−−−−−−−
    1. k : object
                List of kappa terms
    2. ps : object
                List of parameters
                *This function optimizes the following parameters*
                                                                alpha(a)
                                                                beta(b)
                                                                delta(d)
                                                                epsilon(e)
                                                                lambda(l)
    3. cs : list
                List of parameters to be kept constant with the first term being
                the Pade approximant used
                *This function holds no parameters constant*

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [3/2] approximant at the corresponding
               kappa terms and parameters
    """
    l = ps[0]
    d = ps[1]
    e = ps[2]
    a = ps[3]
    b = ps[4]
    a4b2=a*a*a*a + b*b
    aak2=a*a*k*2
    ddk=d*d*k
    num = (k*k + aak2 + a4b2) * (1 + ddk)
    den = a4b2 + aak2 + ddk*a4b2 + e*e*k*k
    return l * num / den

def ppade_42(k, ps, cs):
    """
    This function calculates the pre-fit for the [4/2] Pade approximant
    Pade [4/2] in eqs. (10) and (11) had several typos

    PARAMETERS
    −−−−−−−−−−
    1. k : object
                List of kappa terms
    2. ps : object
                List of parameters
                *This function optimizes the following parameters*
                                                                delta(d)
                                                                gamma(g)
                                                                omega(w)
                                                                lambda(l)
    3. cs : list
                List of parameters to be kept constant with the first term being
                the Pade approximant used
                *This function holds the following parameters constant*
                                                                    alpha(a)
                                                                    beta(b)
    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [4/2] approximant at the corresponding
               kappa terms and parameters
    """

    l = ps[3]
    g = ps[1]
    d = ps[2]
    w = ps[0]
    a = cs[0]
    b = cs[1]
    a4b2 = a * a * a * a + b * b
    g4d2 = g * g * g * g + d * d
    ta2 = 2 * a * a
    tg2 = 2 * g * g
    k2 = k * k
    mu2 = ta2 * g4d2 + tg2 * a4b2
    num = (k2 + ta2 * k + a4b2) * (k2 + tg2 * k + g4d2)
    den = a4b2 * g4d2 + mu2 * k + w * w * k2
    return l * num / den

def pade_42(k, ps, cs):
    """
    This function calculates the [4/2] Pade approximant
    Pade [4/2] in eqs. (10) and (11) had several typos

    PARAMETERS
    −−−−−−−−−−
    1. k : object
                List of kappa terms
    2. ps : object
                List of parameters
                *This function optimizes the following parameters*
                                                                alpha(a)
                                                                beta(b)
                                                                delta(d)
                                                                gamma(g)
                                                                omega(w)
                                                                lambda(l)
    3. cs : list
                List of parameters to be kept constant with the first term being
                the Pade approximant used
                *This function holds no parameters constant*

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [4/2] approximant at the corresponding
               kappa terms and parameters
    """
    
    l = ps[3]
    g = ps[1]
    d = ps[2]
    w = ps[0]
    a = ps[4]
    b = ps[5]
    a4b2=a*a*a*a + b*b
    g4d2=g*g*g*g + d*d
    ta2=2*a*a
    tg2=2*g*g
    k2=k*k
    mu2 = ta2*g4d2  + tg2*a4b2
    num = (k2 + ta2*k + a4b2) * (k2 + tg2*k + g4d2)
    den = a4b2 * g4d2 + mu2*k + w*w*k2
    return l * num / den

def ppade_52(k, ps, cs):
    """
    This function calculates the pre-fit for the [5/2] Pade approximant
    Pade [5/2] based on notes from Roman Curik 1/24/2020 & 1/30/2020

    PARAMETERS
    −−−−−−−−−−
    1. k : object
                List of kappa terms
    2. ps : object
                List of parameters
                *This function optimizes the following parameters*
                                                                delta(d)
                                                                gamma(g)
                                                                epsilon(e)
                                                                omega(w)
                                                                lambda(l)
    3. cs : list
                List of parameters to be kept constant with the first term being
                the Pade approximant used
                *This function holds the following parameters constant*
                                                                    alpha(a)
                                                                    beta(b)
    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [5/2] approximant at the corresponding
               kappa terms and parameters
    """
    l = ps[4]
    e = ps[0]
    w = ps[1]
    d = ps[2]
    g = ps[3]
    a = cs[0]
    b = cs[1]
    a4b2=a*a*a*a + b*b
    g4d2=g*g*g*g + d*d
    ta2=2*a*a
    tg2=2*g*g
    k2=k*k
    mu2 = ta2*g4d2  + tg2*a4b2 + e*e*a4b2*g4d2
    num = (k2 + ta2*k + a4b2) * (k2 + tg2*k + g4d2) * (1 + e*e*k)
    den = a4b2 * g4d2 + mu2*k + w*w*k2
    return l * num / den

def pade_52(k, ps, cs):
    """
    This function calculates the [5/2] Pade approximant
    Pade [5/2] based on notes from Roman Curik 1/24/2020 & 1/30/2020

    PARAMETERS
    −−−−−−−−−−
    1. k : object
                List of kappa terms
    2. ps : object
                List of parameters
                *This function optimizes the following parameters*
                                                                alpha(a)
                                                                beta(b)
                                                                delta(d)
                                                                gamma(g)
                                                                epsilon(e)
                                                                omega(w)
                                                                lambda(l)
    3. cs : list
                List of parameters to be kept constant with the first term being
                the Pade approximant used
                *This function holds no parameters constant*

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [5/2] approximant at the corresponding
               kappa terms and parameters
    """

    l = ps[4]
    e = ps[0]
    w = ps[1]
    d = ps[2]
    g = ps[3]
    a = ps[4]
    b = ps[5]
    a4b2=a*a*a*a + b*b
    g4d2=g*g*g*g + d*d
    ta2=2*a*a
    tg2=2*g*g
    k2=k*k
    mu2 = ta2*g4d2  + tg2*a4b2 + e*e*a4b2*g4d2
    num = (k2 + ta2*k + a4b2) * (k2 + tg2*k + g4d2) * (1 + e*e*k)
    den = a4b2 * g4d2 + mu2*k + w*w*k2
    return l * num / den

def pppade_53(k, ps, cs):
    """
    This function calculates the pre-pre-fit for the [5/3] Pade approximant
    Pade [5/3] based on notes from Roman Curik 1/24/2020 & 1/30/2020

    PARAMETERS
    −−−−−−−−−−
    1. k : object
                List of kappa terms
    2. ps : object
                List of parameters
                *This function optimizes the following parameters*
                                                                    epsilon(e)
                                                                    zeta(z)
                                                                    omega(w)
                                                                    lambda(l)
    3. cs : list
                List of parameters to be kept constant with the first term being
                the Pade approximant used
                *This function holds the following parameters constant*
                                                                    alpha(a)
                                                                    beta(b)
                                                                    delta(d)
                                                                    gamma(g)
    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [5/3] approximant at the corresponding
               kappa terms and parameters
    """
    l = ps[3]
    z = ps[0]
    e = ps[1]
    w = ps[2]
    a = cs[2]
    b = cs[3]
    d = cs[0]
    g = cs[1]
    a4b2=a*a*a*a + b*b
    g4d2=g*g*g*g + d*d
    ta2=2*a*a
    tg2=2*g*g
    k2=k*k
    mu2 = ta2*g4d2  + tg2*a4b2 + e*e*a4b2*g4d2
    num = (k2 + ta2*k + a4b2) * (k2 + tg2*k + g4d2) * (1 + e*e*k)
    den = a4b2 * g4d2 + mu2*k + z*z*k2 + w*w*k*k2
    return l * num / den

def ppade_53(k, ps, cs):
    """
    This function calculates the pre-fit for the [5/3] Pade approximant
    Pade [5/3] based on notes from Roman Curik 1/24/2020 & 1/30/2020

    PARAMETERS
    −−−−−−−−−−
    1. k : object
                List of kappa terms
    2. ps : object
                List of parameters
                *This function optimizes the following parameters*
                                                                    delta(d)
                                                                    gamma(g)
                                                                    epsilon(e)
                                                                    zeta(z)
                                                                    omega(w)
                                                                    lambda(l)
    3. cs : list
                List of parameters to be kept constant with the first term being
                the Pade approximant used
                *This function holds the following parameters constant*
                                                                    alpha(a)
                                                                    beta(b)

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [5/3] approximant at the corresponding
               kappa terms and parameters
    """

    l = ps[5]
    z = ps[0]
    e = ps[1]
    w = ps[2]
    d = ps[3]
    g = ps[4]
    a = cs[0]
    b = cs[1]
    a4b2=a*a*a*a + b*b
    g4d2=g*g*g*g + d*d
    ta2=2*a*a
    tg2=2*g*g
    k2=k*k
    mu2 = ta2*g4d2  + tg2*a4b2 + e*e*a4b2*g4d2
    num = (k2 + ta2*k + a4b2) * (k2 + tg2*k + g4d2) * (1 + e*e*k)
    den = a4b2 * g4d2 + mu2*k + z*z*k2 + w*w*k*k2
    return l * num / den

def pade_53(k, ps, cs):
    """
    This function calculates the [5/3] Pade approximant
    Pade [5/3] based on notes from Roman Curik 1/24/2020 & 1/30/2020

    PARAMETERS
    −−−−−−−−−−
    1. k : object
                List of kappa terms
    2. ps : object
                List of parameters
                *This function optimizes the following parameters*
                                                                    alpha(a)
                                                                    beta(b)
                                                                    delta(d)
                                                                    gamma(g)
                                                                    epsilon(e)
                                                                    zeta(z)
                                                                    omega(w)
                                                                    lambda(l)
    3. cs : list
                List of parameters to be kept constant with the first term being
                the Pade approximant used
                *This function holds no parameters constant*

    RETURNS
    -------
    4. lamda : float
               The evaluation of the Pade [5/3] approximant at the corresponding
               kappa terms and parameters
    """
    l = ps[5]
    z = ps[0]
    e = ps[1]
    w = ps[2]
    d = ps[3]
    g = ps[4]
    a = ps[6]
    b = ps[7]
    a4b2=a*a*a*a + b*b
    g4d2=g*g*g*g + d*d
    ta2=2*a*a
    tg2=2*g*g
    k2=k*k
    mu2 = ta2*g4d2  + tg2*a4b2 + e*e*a4b2*g4d2
    num = (k2 + ta2*k + a4b2) * (k2 + tg2*k + g4d2) * (1 + e*e*k)
    den = a4b2 * g4d2 + mu2*k + z*z*k2 + w*w*k*k2
    return l * num / den

"""
I couldn't figure out how to make it decipher when it should use the constants as 
part of the optimization, so I just went back to pre-optimizing.
For the [5/3] I pre-optimized twice. Once holding alpha, beta, delta, and gamma 
constant, and another time just holding alpha and beta constant. Not sure if that's 
that best way to go.
"""


fs={"pade_21": pade_21,
    "pade_31": pade_31,
    "ppade_32": ppade_32,
    "pade_32": pade_32,
    "ppade_42": ppade_42,
    "pade_42": pade_42,
    "ppade_52": ppade_52,
    "pade_52": pade_52,
    "pppade_53": pppade_53,
    "ppade_53": ppade_53,
    "pade_53": pade_53
    }


def chi(ps, xs, ys, sigmas, cs):
    """
    This function calculates the chi^2 value based on eq. (6)
    See JCP 143, 184102 (2015)

    PARAMETERS
    −−−−−−−−−−
    1. ps : object
                List of parameters
    2. xs : object
                List of kappa values
    3. ys : object
                List of lambda values
    4. sigmas : object
                Used to calculate the weights
    5. cs : list
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used

    RETURNS
    -------
    6. chi2/len(xs) : float
                          This corresponds to the chi^2 value from eq. (6)
    """
    fn = fs[cs[0]]
    chi2 = 0
    ws = 1/sigmas**2
    for i in range(len(xs)):
       chi2 += ((ys[i] - fn(xs[i], ps, cs[1:]))*ws[i])**2
    #chi2 = np.sum(np.square(ys - fn(xs, ps, cs[1:])))   
    return (chi2/len(xs))
                #took out np.sqrt to give closer numbers
    			#changed sum(ws) to len(xs)

def graph(ks, kps, lbs, lps, ps, cs, nplt, pr):
    """
    This function creates a graph of kappa values versus the lambda values for
    use in determining if the program is function correctly

    PARAMETERS
    −−−−−−−−−−
    1. ks : object
                List of kappas
    2. kps : object
                    Number of points used to make the curve seem smooth
    3. lbs : object
                 List of lambdas
    4. lps : object
                    Number of points used to make the curve seem smotth
    5. ps : object
                List of parameters
    6. cs : list
                    List of parameters to be kept constant with the first term being
                    the Pade approximant used
    7. nplt : object
                    The number of points used to plot the respective graphs
    8. pr : print level
            probably take this one out

    RETURNS
    -------
    9. None, but does produce graphs
    """
    fn = fs[cs[0]]
    if pr > 4:        
        plt.plot(ks, lbs, marker='o', color='blue')
        for i in range(nplt):
            lps[i]=fn(kps[i], ps, cs[1:])
        plt.plot(kps, lps, marker='', color="orange")
        plt.xlabel('kappa')
        plt.ylabel('lambda')
        #plt.title(cs[0])
        plt.show()

def erG(aopt, bopt):
    """
    This function calculates the resonance energies and associated gamma values based
    on eq. (5)
    See JCP 143, 184102 (2015)

    PARAMETERS
    −−−−−−−−−−
    1. aopt : object
                     The optimized alpha parameter
    2. bopt : object
                    The optimized beta parameter

    RETURNS
    -------
    3. None, but appends the list of calculated resonance energies (Ers) and gammas (Gs)
    """
    Er = bopt**2 - aopt**4
    G = 4*aopt**2*bopt
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
    This function only returns the list of gammas (resonance widths)

    PARAMETERS
    −−−−−−−−−−
    1. None

    RETURNS
    -------
    4. The list of gammas
    """
    return Gs

def select_energies(ls, Es, E_range, pr):
    """
    select all energies in the first column that lie within E_range
    """
    n_ene = np.shape(Es)[1]
    n_ls = np.shape(ls)[0]
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