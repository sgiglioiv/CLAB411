#!/home/thomas/psi4conda/bin/python

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
    """ Pade [2.1] eq.(2) """
    l = ps[0]
    a = ps[1]
    b = ps[2]
    a4b2=a*a*a*a + b*b
    aak2=a*a*k*2
    return l*(k*k + aak2 + a4b2) / (a4b2 + aak2)
    
def pade_31(k, ps, cs):
    """ 
    Pade [3,1] eq.(9) corrected by missing factor 2 in a^2k terms 
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
    Pade [3,2] as given in 2016 correction paper
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
    Pade [3,2] as given in 2016 correction paper
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
    Pade [4,2] in eqs.(10) and (11) has several typos
    this is now the correct formula from the 2016 paper
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
    Pade [4,2] in eqs.(10) and (11) has several typos 
    this is now the correct formula from the 2016 paper
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

def pade_52(k, ps, cs):
    """
    Pade [5/2] based on notes from Roman Curik 
    1/24/2020 & 1/30/2020
    """

    l = ps[0]
    e = ps[0]
    w = ps[0]
    d = ps[0]
    g = ps[0]
    a = cs[0]
    b = cs[0]
    a4b2=a*a*a*a + b*b
    g4d2=g*g*g*g + d*d
    ta2=2*a*a
    tg2=2*g*g
    k2=k*k
    mu2 = ta2*g4d2  + tg2*a4b2 + e*e*a4b2*g4d2
    num = (k2 + ta2*k + a4b2) * (k2 + tg2*k + g4d2) * (1 + e*e*k)
    den = a4b2 * g4d2 + mu2*k + w*w*k2
    return l * num / den

def pade_53(k, ps, cs):
    """
    Pade [5/3] based on notes from Roman Curik
    1/24/2020 & 1/30/2020 
    """

    l = ps[0]
    e = ps[0]
    z = ps[0]
    w = ps[0]
    a = cs[0]
    b = cs[0]
    d = cs[0]
    g = cs[0]
    a4b2=a*a*a*a + b*b
    g4d2=g*g*g*g + d*d
    ta2=2*a*a
    tg2=2*g*g
    k2=k*k
    mu2 = ta2*g4d2  + tg2*a4b2 + e*e*a4b2*g4d2
    num = (k2 + ta2*k + a4b2) * (k2 + tg2*k + g4d2) * (1 + e*e*k)
    den = a4b2 * g4d2 + mu2*k + z*z*k2 + w*w*k*k2
    return l * num / den

fs={"pade_21": pade_21,
    "pade_31": pade_31,
    "ppade_32": ppade_32,
    "pade_32": pade_32,
    "ppade_42": ppade_42,
    "pade_42": pade_42,
    "pade_52": pade_52,
    "pade_53": pade_53
    }


def chi(ps, xs, ys, sigmas, cs):
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
    Er = bopt**2 - aopt**4
    G = 4*aopt**2*bopt
    Ers.append(Er)
    Gs.append(G)

def getErs():
    return Ers

def getGs():
    return Gs

"""
def chi2_21(params, ks, lbs, sigmas):
    "" 
    Pade-III fit:
    put the most trusted parameters last, 
    because the 1st parameters may be optimized 1st
    ""
    l = params[0]
    a = params[1]
    b = params[2]
    chi2 = 0
    for i in range(len(ks)):
        chi2 += ((pade_21(ks[i],l,a,b) - lbs[i])/sigmas[i])**2
    return chi2 / len(ks)


def chi2_31(params, ks, lbs, sigmas):
    ""
    Pade-III fit: 
    put the most trusted parameters last, 
    because the 1st parameters may be optimized 1st
    ""
    d = params[0]
    l = params[1]
    a = params[2]
    b = params[3]
    chi2 = 0
    for i in range(len(ks)):
        chi2 += ((pade_31(ks[i],l,a,b,d) - lbs[i])/sigmas[i])**2
    return chi2 / len(ks)

def chi2_32(params, ks, lbs, sigmas):
    ""
    Pade-III fit: 
    put the most trusted parameters last, 
    because the 1st parameters may be optimized 1st
    ""
    l = params[0]
    d = params[1]
    e = params[2]
    a = params[3]
    b = params[4]
    chi2 = 0
    for i in range(len(ks)):
        chi2 += ((pade_32(ks[i],l,a,b,d,e) - lbs[i])/sigmas[i])**2
    return chi2 / len(ks)

def chi2_32_pre(params, ks, lbs, sigmas, a, b):
    "" 
    Pade-III fit: 
    put the most trusted parameters last, 
    because the 1st parameters may be optimized 1st
    ""
    l = params[0]
    d = params[1]
    e = params[2]
    chi2 = 0
    for i in range(len(ks)):
        chi2 += ((pade_32(ks[i],l,a,b,d,e) - lbs[i])/sigmas[i])**2
    return chi2 / len(ks)


def chi2_42(params, ks, lbs, sigmas):
    "" 
    Pade-III fit: 
    put the most trusted parameters last, 
    because the 1st parameters may be optimized 1st
    ""
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

def chi2_42_pre(params, ks, lbs, sigmas, a, b):
    "" 
    Pade-III fit: 
    put the most trusted parameters last, 
    because the 1st parameters may be optimized 1st
    ""
    e = params[0]
    g = params[1]
    d = params[2]
    l = params[3]
    chi2 = 0
    for i in range(len(ks)):
        chi2 += ((pade_42(ks[i],l,a,b,g,d,e) - lbs[i])/sigmas[i])**2
    return chi2 / len(ks)
"""
