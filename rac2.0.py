#!/home/thomas/psi4conda/bin/python

"""
Jiri Horacek's regularized analytic continuation
see JCP 143, 184102 (2015)
"""


import argparse

import matplotlib.pyplot as plt
import numpy
from scipy.optimize import minimize


def main():

    au2eV = 27.2113845
    print ("R-ACCC")
    parser = argparse.ArgumentParser(description='Regularized ACCC')
    parser.add_argument("fname", type=str, 
                        help="input data file; 1st column lambdas; then columns with energies")
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
    

    arguments = parser.parse_args()

    Er = arguments.er
    Ei = arguments.ei
    Emax = arguments.emax
    Emin = arguments.emin
    data = numpy.lib.loadtxt(arguments.fname)
    ls = data[:,0]    
    Es = data[:,1:]
    for i in range(len(ls)):
        Es[i,:]=sorted(Es[i,:])

    """ process -s, -b, and -vv1 """
    if arguments.scale: Es *= au2eV
    if arguments.invert:
        Es *= -1
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
        n_ene = numpy.shape(Es)[1]
        for i in range(n_ene):
            plt.plot(ls, Es[:,i], marker='d', color='orange')
        plt.title('raw data')
        plt.show()
        for i in range(n_ene):
            Es[:,i] -= ls * au2eV
        for i in range(n_ene):
            plt.plot(ls, Es[:,i], marker='d', color='orange')
        plt.title('after the shift')
        plt.show()

    rac(ls, Es, (Emin,Emax), (Er,Ei))


def rac(ls, Es, E_range, guess):
    """ 
    main driver of R-ACCC procedure 
    """
    global ks, lbs, sigmas 
    lbs, Ebs = select_energies(ls, Es, E_range)
    if len(lbs) == 0:
        print ("no input data selected")
        return
    sigmas=numpy.ones(len(Es))
    """  
    Jiri's 2015 notation: E = -kappa^2     
    """
    ks = numpy.sqrt(-Ebs)
    """
    make a guess for the parameters in eq. (2)
    to do so eq.(5) must be inverted
    """
    Er=guess[0]
    G=2*guess[1]
    ag=0.5*numpy.sqrt(2.0)*(-2*Er + numpy.sqrt(4*Er**2 + G**2))**0.25
    bg=0.5*G/numpy.sqrt(-2*Er + numpy.sqrt(4*Er**2 + G**2))
    lg=lbs[0]/pade_21(ks[0],1,ag,bg)
    plt.plot(ks, lbs, marker='o', color='blue')
    nplt=200
    kps=numpy.linspace(0,ks[-1],nplt)
    lps=numpy.zeros(nplt)
    for i in range(nplt):
        lps[i]=pade_21(kps[i], lg, ag, bg)
    plt.plot(kps, lps, marker='', color="orange")
    plt.xlabel('kappa')
    plt.ylabel('lambda')
    plt.title('Guess for [2,1] approximant')
    plt.show()
    """
    fit a [2,1] Pade approximant
    """ 
    print ("------------------------------------------------")
    popts = [lg, ag, bg]
    print ("[2,1] Guess l, a, b = ", popts) 
    res = minimize(chi2_21, popts, method='BFGS', options={'gtol': 1e-6})
    popts = res.x
    print ("[2,1] Opt   l, a, b = ", popts)
    chi2 = chi2_21(popts)
    print ("[2,1] chi^2", chi2)
    lopt=popts[0]
    aopt=popts[1]
    bopt=popts[2]
    plt.plot(ks, lbs, marker='o', color='blue')
    for i in range(nplt):
        lps[i]=pade_21(kps[i], lopt, aopt, bopt)
    plt.plot(kps, lps, marker='', color="orange")
    plt.xlabel('kappa')
    plt.ylabel('lambda')
    plt.title('[2,1] approximant')
    plt.show()
    Er=bopt**2 - aopt**4
    G=4*aopt**2*bopt
    Ers = [Er]
    Gs = [G]
    chis = [chi2]
    """
    fit a [3,1] Pade approximant
    """
    print ("------------------------------------------------")
    dg=0.0
    lg=lbs[0]/pade_31(ks[0],1,aopt,bopt,dg)
    popts=[dg, lg, aopt, bopt]
    print ("[3,1] Guess d, l, a, b = ", popts)
    res = minimize(chi2_31, popts, method='BFGS', options={'gtol': 1e-6})
    popts = res.x
    print ("[3,1] Opt   d, l, a, b = ", popts)
    chi2 = chi2_31(popts)
    print ("[3,1] chi^2", chi2)
    dopt=popts[0]
    lopt=popts[1]
    aopt=popts[2]
    bopt=popts[3]
    plt.plot(ks, lbs, marker='o', color='blue')
    for i in range(nplt):
        lps[i]=pade_31(kps[i], lopt, aopt, bopt, dopt)
    plt.plot(kps, lps, marker='', color="orange")
    plt.xlabel('kappa')
    plt.ylabel('lambda')
    plt.title('[3,1] approximant')
    plt.show()
    Er=bopt**2 - aopt**4
    G=4*aopt**2*bopt
    Ers.append(Er)
    Gs.append(G)
    chis.append(chi2)
    """
    fit a [3,2] Pade approximant
    """
    print ("------------------------------------------------")
    eg=0.001
    # preopt delta, epsilson, and l; keep alpha and beta fixed
    popts=[lopt, dopt, eg]
    print ("[3,2] pre-guess l, d, e = "); print ("    ", popts)
    res = minimize(chi2_32_pre, popts, args=(aopt,bopt), method='BFGS')
    popts = res.x
    lg=popts[0]
    dg=popts[1]
    eg=popts[2]
    popts = [lg, dg, eg, aopt, bopt] 
    print ("[3,2] pre-opt = full guess l, d, e, a, b = "); print ("    ", popts)
    res = minimize(chi2_32, popts, method='BFGS', options={'gtol': 1e-6})
    popts = res.x
    print ("[3,2] full optimization    l, d, e, a, b = "); print ("    ", popts)
    chi2 = chi2_32(popts)
    print ("[3,2] chi^2", chi2)
    lopt=popts[0]
    dopt=popts[1]
    eopt=popts[2]
    aopt=popts[3]
    bopt=popts[4]
    plt.plot(ks, lbs, marker='o', color='blue')
    for i in range(nplt):
        lps[i]=pade_32(kps[i], lopt, aopt, bopt, dopt, eopt)
    plt.plot(kps, lps, marker='', color="orange")
    plt.xlabel('kappa')
    plt.ylabel('lambda')
    plt.title('[3,2] approximant')
    plt.show()
    Er=bopt**2 - aopt**4
    G=4*aopt**2*abs(bopt)
    Ers.append(Er)
    Gs.append(G)
    chis.append(chi2)
    """
    fit a [4,2] Pade approximant
    the second resonance guess is 4*Er, 4*Gamma, 
    and the mixing factor epsilon is 1.0
    """
    print ("------------------------------------------------")
    Er*=20
    G*=4
    wg=0.1
    gg=0.5*numpy.sqrt(2.0)*(-2*Er + numpy.sqrt(4*Er**2 + G**2))**0.25
    dg=0.5*G/numpy.sqrt(-2*Er + numpy.sqrt(4*Er**2 + G**2))
    lg=lbs[0]/pade_42(ks[0],1,aopt,bopt,gg,dg,wg)
    # preopt gamma, delta, omega, and l; keep alpha and beta fixed
    popts=[wg, gg, dg, lg]
    print ("[4,2] pre-guess w, g, d, l = "); print ("    ", popts)
    res = minimize(chi2_42_pre, popts, args=(aopt,bopt), method='BFGS')
    popts = res.x
    wg=popts[0]
    gg=popts[1]
    dg=popts[2]
    lg=popts[3]
    popts = [wg, gg, dg, lg, aopt, bopt] 
    print ("[4,2] pre-opt = full guess w, g, d, l, a, b = "); print ("    ", popts)   
    res = minimize(chi2_42, popts, method='BFGS', options={'gtol': 1e-6})
    popts = res.x
    print ("[4,2] Full optimization    w, g, d, l, a, b = "); print ("    ", popts)
    chi2 = chi2_42(popts)
    print ("[4,2] chi^2", chi2)
    eopt=popts[0]
    gopt=popts[1]
    dopt=popts[2]
    lopt=popts[3]
    aopt=popts[4]
    bopt=popts[5]
    plt.plot(ks, lbs, marker='o', color='blue')
    for i in range(nplt):
        lps[i]=pade_42(kps[i], lopt, aopt, bopt, gopt, dopt, eopt)
    plt.plot(kps, lps, marker='', color="orange")
    plt.xlabel('kappa')
    plt.ylabel('lambda')
    plt.title('[4,2] approximant')
    plt.show()
    Er=bopt**2 - aopt**4
    G=4*aopt**2*abs(bopt)
    Ers.append(Er)
    Gs.append(G)
    chis.append(chi2)

    pade = ["Pade [2,1]", "Pade [3,1]", "Pade [3,2]", "Pade [4,2]"]
    print ()
    print ("----------------------------------------------------")
    print ("                Er       Gamma           chi^2      ")
    print ("----------------------------------------------------")
    for i in range(len(Ers)):
        print ("{0:s}  {1:8.4f}   {2:11.2e}     {3:11.2e}"\
            .format(pade[i], Ers[i], Gs[i], chis[i]))






def pade_21(k, l, a, b):
    """ Pade [2.1] eq.(2) """
    a4b2=a*a*a*a + b*b
    aak2=a*a*k*2
    return l*(k*k + aak2 + a4b2) / (a4b2 + aak2)
    
def pade_31(k, l, a, b, d):
    """ 
    Pade [3,1] eq.(9) corrected by missing factor 2 in a^2k terms 
    """
    a4b2=a*a*a*a + b*b
    aak2=a*a*k*2
    ddk=d*d*k
    num = (k*k + aak2 + a4b2) * (1 + ddk)
    den = a4b2 + aak2 + ddk*a4b2
    return l * num / den 

def pade_32(k, l, a, b, d, e):
    """ 
    Pade [3,2] as given in 2016 correction paper
    """
    a4b2=a*a*a*a + b*b
    aak2=a*a*k*2
    ddk=d*d*k
    num = (k*k + aak2 + a4b2) * (1 + ddk)
    den = a4b2 + aak2 + ddk*a4b2 + e*e*k*k
    return l * num / den 

def pade_42(k, l, a, b, g, d, w):
    """ 
    Pade [4,2] in eqs.(10) and (11) has several typos 
    this is now the correct formula from the 2016 paper
    """
    a4b2=a*a*a*a + b*b
    g4d2=g*g*g*g + d*d
    ta2=2*a*a
    tg2=2*g*g
    k2=k*k
    mu2 = ta2*g4d2  + tg2*a4b2
    num = (k2 + ta2*k + a4b2) * (k2 + tg2*k + g4d2)
    den = a4b2 * g4d2 + mu2*k + w*w*k2
    return l * num / den 


def chi2_21(params):
    """ 
    Pade-III fit:
    put the most trusted parameters last, 
    because the 1st parameters may be optimized 1st
    """
    l = params[0]
    a = params[1]
    b = params[2]
    chi2 = 0
    for i in range(len(ks)):
        chi2 += ((pade_21(ks[i],l,a,b) - lbs[i])/sigmas[i])**2
    return chi2 / len(ks)


def chi2_31(params):
    """ 
    Pade-III fit: 
    put the most trusted parameters last, 
    because the 1st parameters may be optimized 1st
    """
    d = params[0]
    l = params[1]
    a = params[2]
    b = params[3]
    chi2 = 0
    for i in range(len(ks)):
        chi2 += ((pade_31(ks[i],l,a,b,d) - lbs[i])/sigmas[i])**2
    return chi2 / len(ks)

def chi2_32(params):
    """ 
    Pade-III fit: 
    put the most trusted parameters last, 
    because the 1st parameters may be optimized 1st
    """
    l = params[0]
    d = params[1]
    e = params[2]
    a = params[3]
    b = params[4]
    chi2 = 0
    for i in range(len(ks)):
        chi2 += ((pade_32(ks[i],l,a,b,d,e) - lbs[i])/sigmas[i])**2
    return chi2 / len(ks)

def chi2_32_pre(params, a, b):
    """ 
    Pade-III fit: 
    put the most trusted parameters last, 
    because the 1st parameters may be optimized 1st
    """
    l = params[0]
    d = params[1]
    e = params[2]
    chi2 = 0
    for i in range(len(ks)):
        chi2 += ((pade_32(ks[i],l,a,b,d,e) - lbs[i])/sigmas[i])**2
    return chi2 / len(ks)


def chi2_42(params):
    """ 
    Pade-III fit: 
    put the most trusted parameters last, 
    because the 1st parameters may be optimized 1st
    """
    e = params[0]
    g = params[1]
    d = params[2]
    l = params[3]
    a = params[4]
    b = params[5]
    chi2 = 0
    for i in range(len(ks)):
        chi2 += ((pade_42(ks[i],l,a,b,g,d,e) - lbs[i])/sigmas[i])**2
    return chi2 / len(ks)

def chi2_42_pre(params, a, b):
    """ 
    Pade-III fit: 
    put the most trusted parameters last, 
    because the 1st parameters may be optimized 1st
    """
    e = params[0]
    g = params[1]
    d = params[2]
    l = params[3]
    chi2 = 0
    for i in range(len(ks)):
        chi2 += ((pade_42(ks[i],l,a,b,g,d,e) - lbs[i])/sigmas[i])**2
    return chi2 / len(ks)






def select_energies(ls, Es, E_range):
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

    for i in range(n_ene):
        plt.plot(ls, Es[:,i], marker='d', color='orange')
    plt.plot(lbs, Ebs, marker='o', color='blue')
    plt.title('raw data')
    plt.show()

    return lbs, Ebs


if __name__ == "__main__":
    main()


