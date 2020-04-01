#!/home/thomas/psi4conda/bin/python
"""
Current running version of the code
"""
"""
Jiri Horacek's regularized analytic continuation
see JCP 143, 184102 (2015)
"""


import argparse
import numpy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import rac_utils as racu
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
    parser.add_argument("-prin", type=int, default=5, help="Print level 1-5")
    

    arguments = parser.parse_args()

    Er = arguments.er
    Ei = arguments.ei
    Emax = arguments.emax
    Emin = arguments.emin
    pr = arguments.prin
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
        if pr > 3:
            for i in range(n_ene):
                plt.plot(ls, Es[:,i], marker='d', color='orange')
            plt.title('raw data')
            plt.show()
        for i in range(n_ene):
            Es[:,i] -= ls * au2eV
        if pr > 3:
            for i in range(n_ene):
                plt.plot(ls, Es[:,i], marker='d', color='orange')
            plt.title('after the shift')
            plt.show()

    rac(ls, Es, (Emin,Emax), (Er,Ei), pr)


def rac(ls, Es, E_range, guess, pr):
    """ 
    main driver of R-ACCC procedure 
    """
    global ks, lbs, sigmas 
    lbs, Ebs = racu.select_energies(ls, Es, E_range, pr)
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
    lg=lbs[0]/racu.pade_21(ks[0], [1, ag, bg], [])
    nplt=200
    kps=numpy.linspace(0,ks[-1],nplt)
    lps=numpy.zeros(nplt)
    ps = [lg, ag, bg]
    racu.graph(ks, kps, lbs, lps, ps, ["pade_21"], nplt, pr)


    """
    fit a [2,1] Pade approximant
    """ 
    if pr > 2:
        print ("------------------------------------------------")
    ps= [lg, ag, bg]
    cs= ["pade_21"]
    if pr > 2:
        print ("[2,1] Guess l, a, b = ", ps) 
    res = minimize(racu.chi, ps, args=(ks, lbs, sigmas, cs), method='BFGS', options={'gtol': 1e-6})
    ps = res.x
    if pr > 2:
        print ("[2,1] Opt   l, a, b = ", ps)
    chi2 = racu.chi(ps, ks, lbs, sigmas, cs)
    if pr > 2:
        print ("[2,1] chi^2", chi2)
    lopt=ps[0]
    aopt=ps[1]
    bopt=ps[2]
    chis = [chi2]
    racu.graph(ks, kps, lbs, lps, ps, cs, nplt, pr)
    racu.erG(aopt, bopt)

    """
    fit a [3,1] Pade approximant
    """
    if pr > 2:
        print ("------------------------------------------------")
    dg=0.0
    ps=[dg, 1, aopt, bopt]
    cs=["pade_31"]
    lg=lbs[0]/racu.pade_31(ks[0], ps, cs)
    ps=[dg, lg, aopt, bopt]
    if pr > 2:
        print ("[3,1] Guess d, l, a, b = ", ps)
    res = minimize(racu.chi, ps, args=(ks, lbs, sigmas, cs), method='BFGS', options={'gtol': 1e-6})
    ps = res.x
    if pr > 2:
        print ("[3,1] Opt   d, l, a, b = ", ps)
    chi2 = racu.chi(ps, ks, lbs, sigmas, cs)
    if pr > 2:
        print ("[3,1] chi^2", chi2)
    dopt=ps[0]
    lopt=ps[1]
    aopt=ps[2]
    bopt=ps[3]
    chis.append(chi2)
    racu.graph(ks, kps, lbs, lps, ps, cs, nplt, pr)
    racu.erG(aopt, bopt)

    """
    fit a [3,2] Pade approximant
    """
    if pr > 2:
        print ("------------------------------------------------")
    eg=0.001
    # preopt delta, epsilson, and l; keep alpha and beta fixed
    ps=[lopt, dopt, eg]
    cs=["ppade_32", aopt, bopt]
    if pr > 2:
        print ("[3,2] pre-guess l, d, e = "); print ("    ", ps)
    res = minimize(racu.chi, ps, args=(ks, lbs, sigmas, cs), method='BFGS')
    ps = res.x
    lg=ps[0]
    dg=ps[1]
    eg=ps[2]
    ps = [lg, dg, eg, aopt, bopt]
    cs = ["pade_32"]
    if pr > 2:
        print ("[3,2] pre-opt = full guess l, d, e, a, b = "); print ("    ", ps)
    res = minimize(racu.chi, ps, args=(ks, lbs, sigmas, cs), method='BFGS', options={'gtol': 1e-6})
    ps = res.x
    if pr > 2:
        print ("[3,2] full optimization    l, d, e, a, b = "); print ("    ", ps)
    chi2 = racu.chi(ps, ks, lbs, sigmas, cs)
    if pr > 2:
        print ("[3,2] chi^2", chi2)
    lopt=ps[0]
    dopt=ps[1]
    eopt=ps[2]
    aopt=ps[3]
    bopt=ps[4]
    chis.append(chi2)
    racu.graph(ks, kps, lbs, lps, ps, cs, nplt, pr)
    racu.erG(aopt, bopt)

    """
    fit a [4,2] Pade approximant
    the second resonance guess is 4*Er, 4*Gamma, 
    and the mixing factor epsilon is 1.0
    """
    if pr > 2:
        print ("------------------------------------------------")
    Ers = racu.getErs()
    Gs = racu.getGs()
    Er=Ers[-1]*20
    G=Gs[-1]*4
    wg=0.1
    gg=0.5*numpy.sqrt(2.0)*(-2*Er + numpy.sqrt(4*Er**2 + G**2))**0.25
    dg=0.5*G/numpy.sqrt(-2*Er + numpy.sqrt(4*Er**2 + G**2))
    ps = [wg, gg, dg, 1]
    cs = ["ppade_42", aopt, bopt]
    lg=lbs[0]/racu.ppade_42(ks[0], ps, cs[1:])
    # preopt gamma, delta, omega, and l; keep alpha and beta fixed
    ps=[wg, gg, dg, lg]
    if pr > 2:
        print ("[4,2] pre-guess w, g, d, l = "); print ("    ", ps[:4])
    res = minimize(racu.chi, ps, args=(ks, lbs, sigmas, cs), method='BFGS')
    ps = res.x
    wg=ps[0]
    gg=ps[1]
    dg=ps[2]
    lg=ps[3]
    ps = [wg, gg, dg, lg, aopt, bopt]
    if pr > 2:
        print ("[4,2] pre-opt = full guess w, g, d, l, a, b = "); print ("    ", ps)
    cs = ["pade_42"]
    res = minimize(racu.chi, ps, args=(ks, lbs, sigmas, cs), method='BFGS', options={'gtol': 1e-6})
    ps = res.x
    if pr > 2:
        print ("[4,2] Full optimization    w, g, d, l, a, b = "); print ("    ", ps)
    chi2 = racu.chi(ps, ks, lbs, sigmas, cs)
    if pr > 2:
        print ("[4,2] chi^2", chi2)
    wopt=ps[0]
    gopt=ps[1]
    dopt=ps[2]
    lopt=ps[3]
    aopt=ps[4]
    bopt=ps[5]
    chis.append(chi2)
    racu.graph(ks, kps, lbs, lps, ps, cs, nplt, pr)
    racu.erG(aopt, bopt)


    """
    fit a [5,2] Pade approximant
    the second resonance guess is 4*Er, 4*Gamma, 
    and the mixing factor epsilon is 1.0
    """

    if pr > 2:
        print ("------------------------------------------------")
    Ers = racu.getErs()
    Gs = racu.getGs()
    Er = Ers[-1] * 20
    G = Gs[-1] * 4
    eg = 0.1
    wg= wopt * 0.05
    gg=0.5*numpy.sqrt(2.0)*(-2*Er + numpy.sqrt(4*Er**2 + G**2))**0.25
    dg=0.5*G/numpy.sqrt(-2*Er + numpy.sqrt(4*Er**2 + G**2))
    ps = [eg, wg, gg, dg, 1]
    cs = ["ppade_52", aopt, bopt]
    lg = lbs[0]/racu.ppade_52(ks[0], ps, cs[1:])
    # preopt epsilon, gamma, delta, omega, and l; keep alpha and beta fixed
    ps = [eg, wg, gg, dg, lg]
    if pr > 2:
        print ("[5,2] pre-guess e, w, g, d, l = "); print ("    ", ps[:5])
    res = minimize(racu.chi, ps, args=(ks, lbs, sigmas, cs), method='BFGS')
    ps = res.x
    eg=ps[0]
    wg=ps[1]
    gg=ps[2]
    dg=ps[3]
    lg=ps[4]
    ps = [eg, wg, gg, dg, lg, aopt, bopt]
    if pr > 2:
        print ("[5,2] pre-opt = full guess e, w, g, d, l, a, b = "); print ("    ", ps)
    cs=["pade_52"]
    res = minimize(racu.chi, ps, args=(ks, lbs, sigmas, cs), method='BFGS',
                   options={'gtol': 1e-6})
    ps = res.x
    if pr > 2:
        print ("[5,2] Full optimization    e, w, g, d, l, a, b = "); print ("    ", ps)
    chi2 = racu.chi(ps, ks, lbs, sigmas, cs)
    if pr > 2:
        print("[5,2] chi^2", chi2)
    eopt = ps[0]
    wopt = ps[1]
    gopt = ps[2]
    dopt = ps[3]
    lopt = ps[4]
    aopt = ps[5]
    bopt = ps[6]
    chis.append(chi2)
    racu.graph(ks, kps, lbs, lps, ps, cs, nplt, pr)
    racu.erG(aopt, bopt)


    """
    fit a [5,3] Pade approximant
    the second resonance guess is 4*Er, 4*Gamma, 
    and the mixing factor epsilon is 1.0
    """

    if pr > 2:
        print ("------------------------------------------------")
    Ers = racu.getErs()
    Gs = racu.getGs()
    Er = Ers[-1] * 20
    G = Gs[-1] * 4
    zg = 0.1
    eg = eopt * 0.05
    wg = wopt * 0.05
    gg = 0.5 * numpy.sqrt(2.0) * (-2 * Er + numpy.sqrt(4 * Er ** 2 + G ** 2)) ** 0.25
    dg = 0.5 * G / numpy.sqrt(-2 * Er + numpy.sqrt(4 * Er ** 2 + G ** 2))
    ps = [zg, eg, wg, 1]
    cs = ["pppade_53", gg, dg, aopt, bopt]
    lg = lbs[0] / racu.pppade_53(ks[0], ps, cs[1:])
    # pre-pre-opt zeta, epsilon, omega, and l; keep alpha, beta, gamma, and delta
    # fixed
    ps = [zg, eg, wg, lg]
    if pr > 2:
        print ("[5,3] pre-pre-guess z, e, w, l = "); print ("    ", ps[:4])
    res = minimize(racu.chi, ps, args=(ks, lbs, sigmas, cs), method='BFGS')
    ps = res.x
    zg = ps[0]
    eg = ps[1]
    wg = ps[2]
    lg = ps[3]
    #pre-opt zeta, epsilon, omega, beta, gamma, and l, alpha and beta fixed
    ps = [zg, eg, wg, gg, dg, lg]
    if pr > 2:
        print ("[5,3] pre-guess z, e, w, g, d, l = "); print ("    ", ps[:6])
    cs = ["ppade_53", aopt, bopt]
    res = minimize(racu.chi, ps, args=(ks, lbs, sigmas, cs), method='BFGS')
    ps = res.x
    zg = ps[0]
    eg = ps[1]
    wg = ps[2]
    gg = ps[3]
    dg = ps[4]
    lg = ps[5]
    ps = [zg, eg, wg, gg, dg, lg, aopt, bopt]
    if pr > 2:
        print ("[5,3] pre-opt = full guess z, e, w, g, d, l, a, b = "); print ("    ",
                                                                              ps)
    cs=["pade_53"]
    res = minimize(racu.chi, ps, args=(ks, lbs, sigmas, cs), method='BFGS',
                   options={'gtol': 1e-6})
    ps = res.x
    if pr > 2:
        print("[5,3] Full optimization   z, e, w, g, d, l, a, b = ");
        print("    ", ps)
    chi2 = racu.chi(ps, ks, lbs, sigmas, cs)
    if pr > 2:
        print("[5,3] chi^2", chi2)
    zopt = ps[0]
    eopt = ps[1]
    wopt = ps[2]
    gopt = ps[3]
    dopt = ps[4]
    lopt = ps[5]
    aopt = ps[6]
    bopt = ps[7]
    chis.append(chi2)
    racu.graph(ks, kps, lbs, lps, ps, cs, nplt, pr)
    racu.erG(aopt, bopt)


    pade = ["Pade [2,1]", "Pade [3,1]", "Pade [3,2]", "Pade [4,2]", "Pade [5,2]",
            "Pade [5,3]"]
    Ers = racu.getErs()
    Gs = racu.getGs()
    print ()
    print ("----------------------------------------------------")
    print ("                Er       Gamma           chi^2      ")
    print ("----------------------------------------------------")
    for i in range(len(Ers)):
        print ("{0:s}  {1:8.4f}   {2:11.2e}     {3:11.2e}"\
            .format(pade[i], Ers[i], Gs[i], chis[i]))



if __name__ == "__main__":
    main()


