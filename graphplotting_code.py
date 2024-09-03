#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 09:25:17 2024

@author: wakami
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy
from astropy.io import fits
import os
from pathlib import Path
from astropy.stats import sigma_clip
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths
import csv
 
line = []
line_cont = []
colourline= []
colourcont= []
z= []
lmass_line= []
lmass_cont= []
lsfr_line= []
lsfr_cont= []
 
with open('output_zfire3.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
     
    for ROWS in plotting:
        median_data = float(ROWS[1])
        if median_data <= 0.0052:
            line.append(float(ROWS[4]))
            colourline.append(float(ROWS[8]))
            lsfr_line.append(float(ROWS[12]))
            lmass_line.append(float(ROWS[11]))
        else:
            line_cont.append(float(ROWS[4]))
            colourcont.append(float(ROWS[8]))
            lsfr_cont.append(float(ROWS[12]))
            lmass_cont.append(float(ROWS[11]))
        
            
line_no= len(line)
linecont_no= len(line_cont)
smass_cont = [i * 10 for i in lmass_cont]
smass_line = [i * 10 for i in lmass_line]


fits1 = fits.open('/Users/wakami/Downloads/G1.0_tau3000.0_Z0.05 (1).fits')
fits135 = fits.open('/Users/wakami/Downloads/G1.35_tau3000.0_Z0.05 (1).fits')
fits2= fits.open('/Users/wakami/Downloads/G2.0_tau3000.0_Z0.05 (1).fits')

data1= fits1[1].data
data135= fits135[1].data
data2= fits2[1].data

ha1= data1['EW(H_alpha)']
ha1= np.log10(ha1)
colour1= data1['g-r']
ha135= data135['EW(H_alpha)']
ha135= np.log10(ha135)
colour135= data135['g-r']
ha2= data2['EW(H_alpha)']
ha2= np.log10(ha2)
colour2= data2['g-r']

#'#fdb22f': colour used for 'line only' when plotting without sfr
#'#c13b82': colour used for 'line & continuum' when plotting with sfr
plt.figure(figsize=(10,8))
plt.scatter(colourline, line, s=smass_line, c= lsfr_line, cmap= 'plasma', marker="^", label= f'Line only, N= {line_no}')
plt.scatter(colourcont, line_cont, s=smass_cont, c=lsfr_cont, cmap='plasma', marker="*", label= f'Line and Continuum, N= {linecont_no}')
#plt.plot(colour1, ha1, color= 'black')
#plt.plot(colour135, ha135, color='black')
#plt.plot(colour2, ha2, color='black')
plt.title(r'H$\alpha$ EW against colour')
plt.ylabel(r'log$_{10}$[H$\alpha$ EW]')
plt.xlabel(r'(J1-Hl)$_{z=2.1}$')
plt.colorbar(label=r'log$_{10}$[SFR/M$_\odot$ yr$^{-1}]$')
plt.legend(loc='lower left')
plt.xlim(-0.5, 2.0)
plt.ylim(-1.0, 3.0)
plt.show()