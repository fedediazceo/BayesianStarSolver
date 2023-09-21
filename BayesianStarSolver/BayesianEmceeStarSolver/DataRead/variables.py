# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 17:11:02 2016

@author: Antonio

Standard variables
"""

# (*)From http://www.astro.princeton.edu/~gk/A403/constants.pdf. See "consultas/others" directory
#(**) From https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
G = 6.67384e-8 #cm^3/(g*s^2)
Rsun = 6.955e10 #cm
Msun = 1.9891e33 #g
Lsun = 3.839e33 #erg/s
loggsun = 4.44 #cgs(**)
UA = 1.496e14 #cm
#rhosun = 1.4480; #cgs (= g/cm^3)
rho_sun = 1.409 #Mean density in cgs https://solarsystem.nasa.gov/solar-system/sun/by-the-numbers/ (by Haynes et al. 2012)
Dnusun = 134.8 #muHz (Kjeldsen, Bedding & Christensen-Dalsgaard 2008)
Mbsun = 4.74 #(*) Bolometric absolute magnitude
F0sun = 278.0 #muHz (aprox.)
Dnusun0 = 150.0 #muHz, (150-155) from Andy's model
Teffsun = 5778 #K
nu_max_sun = 3050 #muHz (Kjeldsen & Bedding 1995)

sigma =  5.6704e-5 #cgs (=erg·cm^-2·s^-1·K^-4), Stefan–Boltzmann constant

#a,da  = multiplicative factor and uncertainty of the Dnu-rho relation. Guo's values are default
#b,db  = exponent factor and uncertainty of the Dnu-rho relation. Guo's values are default
#García Hernández et al. (2017): rho/rhosun = a*(Dnu/Dnusun)^b
a,da = 1.501,0.096
b,db = 2.0373,0.0405

#These are the definitions of the constants
solarzx=0.0245
ypr=0.235
dydz=2.2

#=========================================
def feh_to_z(feh):

    #First, computation of Z, then Y and X
    z = (1.0 - ypr)/((1.0/(solarzx*10**feh))+(dydz+1.0))
    
    return z
#=========================================
def z_to_feh(z):

    # Y = dydz*Z+ypr
    feh = 1.0 - ypr - z*(1+dydz)
    
    return feh
