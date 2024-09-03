#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:41:36 2024

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
import math

zfire_spec= pd.read_csv('/Users/wakami/Desktop/ucl/2023:2024/msc_thesis/zfire_survey/dr1/only_1D/data_files/master_table/ZFIRE_COSMOS_master_table_DR1.1.csv')

# remove galaxies with AGN, could do in loop for efficiency but will figure that out later
filter_xray= zfire_spec['AGN'].str.contains('x-ray') #x-ray
zfire_spec= zfire_spec[~filter_xray]

filter_ir= zfire_spec['AGN'].str.contains('ir') #ir
zfire_spec= zfire_spec[~filter_ir]

filter_BPT= zfire_spec['AGN'].str.contains('BPT') #BPT
zfire_spec= zfire_spec[~filter_BPT]


# remove rows with NaN for Halpha flux and Ks
zfire_spec= zfire_spec[zfire_spec['fha'].notna()]
zfire_spec= zfire_spec[zfire_spec['Ks'].notna()]


# remove rows with flag value of 1 as SNR<5 (Nanayakkara, 2015)
filter_flag= zfire_spec['conf'] == 1
zfire_spec= zfire_spec[~filter_flag]
filter_flag= zfire_spec['conf'] == 2
zfire_spec= zfire_spec[~filter_flag]

# up to this point, we have acheived the dataset up to the first bullet point in (Nanayakkara, 2017)
# we will proceed to plot the figures in the paper

# set directory
directory= '/Users/wakami/Desktop/ucl/2023:2024/msc_thesis/zfire_survey/dr1/only_1D/spectra_1d/cosmos/all_masks/spectra/'

# function to read file with an inputted ID
def read_file(ID):
    ID = str(ID)
    # find file in directory with ID in file name
    pathlist = list(Path(directory).glob('*' +'K' +'*' + f'{ID}' + '*.fits'))

    if not pathlist:
        raise FileNotFoundError(f"No files found for ID {ID} in directory {directory}")

    path_in_str = str(pathlist[0])  # assume the first matching file

    fits1 = fits.open(path_in_str)
    
    # obtain the fits data from the fits file
    fits_header = fits1[0].header
    flux_1D = fits1[0].data
    error_1D = fits1[1].data
    wave_1D = fits1[2].data

    return wave_1D, flux_1D

# function to apply sigma clipping
def apply_sigma_clipping(flux, sigma, iters):
    clipped_flux = sigma_clip(flux, sigma=sigma, maxiters=iters, cenfunc='median', stdfunc='std')
    return clipped_flux.data, clipped_flux.mask

# function to estimate the continuum using sigma clipping
def estimate_continuum(flux, sigma, iters):
    mask = np.ones_like(flux, dtype=bool)
    for i in range(iters):
        median_flux = np.median(flux[mask])
        std_flux = np.std(flux[mask])
        new_mask = np.abs(flux - median_flux) < sigma * std_flux
        if np.all(new_mask == mask):
            break
        mask = new_mask
    continuum_flux = np.copy(flux)
    continuum_flux[~mask] = median_flux
    
    return continuum_flux, mask

# function to find continuum over 100 iterations
def find_continuum(flux, sigma=3, iters=100):
    continuum_flux, mask = estimate_continuum(flux, sigma=sigma, iters=iters)
    med_continuum= np.median(continuum_flux)
    
    return continuum_flux, med_continuum

# function to obtain the redshift, z of inputted galaxy
def get_redshift(ID):
    # obtain the redshift
    filtered_df = zfire_spec[zfire_spec["# ID"].str.contains(f"{ID}")]
    z= filtered_df.iloc[:, 6]
    z= z.values[0]
    z_err= filtered_df.iloc[:, 6]
    z_err= z_err.values[0]
    
    return z, z_err

# function to obtain the dust extinction, Av of inputted galaxy
def get_Av(ID):
    filtered_df = zfire_spec[zfire_spec["# ID"].str.contains(f"{ID}")]
    Av= filtered_df.iloc[:, 11]
    Av= Av.values[0]
    
    return Av

# function to get the signal-to-noise ratio of inputted galaxy
def check_SNR(ID):
    filtered_df = zfire_spec[zfire_spec["# ID"].str.contains(f"{ID}")]
    S= filtered_df.iloc[:, 13]
    S= S.values[0]
    N= filtered_df.iloc[:, 14]
    N= N.values[0]
    SNR= S/N
    
    return SNR

# function to get the integrated flux from the ZFIRE sample for an inputted galaxy
def get_flux(ID):
    filtered_df = zfire_spec[zfire_spec["# ID"].str.contains(f"{ID}")]
    S= filtered_df.iloc[:, 13]
    S= S.values[0]
    N= filtered_df.iloc[:, 14]
    N= N.values[0]
    
    return S, N

# function to find the boundary limits of the emission line
def find_boundaries(wavelength, intensity, ha_redshifted, threshold_factor=0.3):
    # find the 'peak' of the emission line
    peak_idx = np.argmin(np.abs(wavelength - ha_redshifted))
    
    # set the threshold intensity to cut off
    target_intensity = intensity.iloc[peak_idx]
    threshold = target_intensity * threshold_factor
    
    # left boundary
    left_idx = peak_idx
    while left_idx > 0 and intensity.iloc[left_idx] > threshold:
        left_idx -= 1
        
    # right boundary
    right_idx = peak_idx
    while right_idx < len(intensity) - 1 and intensity.iloc[right_idx] > threshold:
        right_idx += 1
        
    left_boundary= wavelength.iloc[left_idx]
    right_boundary= wavelength.iloc[right_idx]
    
    return left_boundary, right_boundary

# function to calculate the EW of inputted galaxy
def calc_EW(ID):
    # obtain the wavelength and flux of the fits file
    try:
        wave_1D, flux_1D= read_file(ID)
    
    # to skip over files without readable spectra or spectra not in range
    except FileNotFoundError:
        return None, None
    
    if wave_1D is None:
        return None, None
    
    # set range of wavelength boundaries
    z,z_err= get_redshift(ID)
    ha_redshifted= ((1+z) * 6562.81)+5
    low_lim= ha_redshifted-50
    high_lim= ha_redshifted +50
    low= ha_redshifted-100
    high= ha_redshifted+100
    
    
    # create a dataframe for the wavelength and flux
    wave_1D = pd.Series(wave_1D.byteswap().newbyteorder())
    flux_1D = pd.Series(flux_1D.byteswap().newbyteorder())
    
    df= pd.DataFrame({'wavelength': wave_1D, 'flux': flux_1D}, columns= ['wavelength', 'flux'])
    df1 = df[(df['wavelength'] >= low) & (df['wavelength'] <= high)]
    df2 = df.query('@low_lim < wavelength < @high_lim').reset_index(drop=True)
    
    # find the continuum
    continuum_flux, median_data= find_continuum(df1['flux'])
    
    # find the boundarires of emission line
    left_boundary, right_boundary= find_boundaries(df2['wavelength'], df2['flux'], ha_redshifted)
    ha_df= df.query('@left_boundary< wavelength < @right_boundary')
    
    if not left_boundary <= ha_redshifted <= right_boundary:
        return None, None 
    
    # plot the flux data to inspect for peaks
    # uncomment if you unwish to plot the individual graphs
    plt.figure(figsize=(10, 8))
    plt.plot(df2['wavelength'], df2['flux'], color='#cf4c74', label='Flux Data')
    plt.axvline(x=ha_redshifted.item(), color='#9c179e', linestyle='--', label='Ha Redshifted')
    plt.axhline(y=median_data, color='#2a0593', label='Continuum')
    plt.axvspan(left_boundary, right_boundary, color='#fdb22f', alpha=0.3, label= 'Emission Line')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.legend()
    plt.title(f'Spectra of {ID}')
    plt.show()
    
    # calculate the observed EW depending on the continuum value obtained
    if (median_data < 0.0052).item():
        ha_ew= ((ha_df['flux']-0.0052)/0.0052) * (wave_1D[1]-wave_1D[0])
        
    else:
        ha_ew= ((ha_df['flux']-median_data)/median_data) * (wave_1D[1]-wave_1D[0])
    
    ha_ew= ha_ew.sum()
    
    # calculate the real EW from observed EW
    if ha_ew <= 0 or (1+z) <= 0:
        log_ew = float('nan')  # Assign NaN if the value is invalid
        
    else:
        log_ew = np.log10(ha_ew / (1 + z))
    
    return median_data, log_ew

# function to get rid of white spaces in datafile
def clean_data(df):
    # strip whitespace from headers and columns
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

# function to convert series to numeric values for manipulation in subsequent functions
def convert_to_numeric(series):
    # convert the series to numeric values, coercing errors
    return pd.to_numeric(series, errors='coerce')

# function to obtain the flux in J1 and Hl filters to calculate colour later on, from ZFOURGE catalogue
def get_colour(ID):
    ID= str(ID)
    
    # open crossmatch id file and identify line with ID
    crossmatch= pd.read_csv('/Users/wakami/Desktop/ucl/2023:2024/msc_thesis/zfire_survey/dr1/only_1D/data_files/cat_comp/id_crossmatch.dat', sep='\t', engine= 'python', error_bad_lines=False, warn_bad_lines=True)
    crossmatch= clean_data(crossmatch)
    crossmatch['ID'] = convert_to_numeric(crossmatch['ID'])
    zfire_crossmatch = crossmatch[crossmatch['ID'] == int(ID)]
    
    # obtain zfourge id
    zfourge_ID= zfire_crossmatch.iloc[:,2].item()
    zfourge_ID = str(zfourge_ID).strip()
    
    # open zfourge file
    zfourge= pd.read_csv('/Users/wakami/Desktop/ucl/2023:2024/msc_thesis/ZFOURGE_cat.txt', sep='\s+', engine= 'python', error_bad_lines=False, warn_bad_lines=True)
    zfourge = clean_data(zfourge)
    zfourge['#Seq'] = convert_to_numeric(zfourge['#Seq'])
    zfourged = zfourge[zfourge['#Seq'] == float(zfourge_ID)]
    
    # obtained observed colours
    # J1 corresponds to [340]
    # Hl corresponds to [550]
    colour_J1= zfourged.iloc[:,2].item()
    colour_Hl= zfourged.iloc[:,1].item()
    
    
    return colour_J1, colour_Hl

# function to obtain SFR from ZFOURGE catalogue
def get_sfr(ID):
    ID= str(ID)
    
    # open crossmatch id file and identify line with ID
    crossmatch= pd.read_csv('/Users/wakami/Desktop/ucl/2023:2024/msc_thesis/zfire_survey/dr1/only_1D/data_files/cat_comp/id_crossmatch.dat', sep='\t', engine= 'python', error_bad_lines=False, warn_bad_lines=True)
    crossmatch= clean_data(crossmatch)
    crossmatch['ID'] = convert_to_numeric(crossmatch['ID'])
    zfire_crossmatch = crossmatch[crossmatch['ID'] == int(ID)]
    
    # obtain zfourge id
    zfourge_ID= zfire_crossmatch.iloc[:,2].item()
    zfourge_ID = str(zfourge_ID).strip()
    
    # open zfourge file
    cosmos= pd.read_csv('/Users/wakami/Desktop/ucl/2023:2024/msc_thesis/cosmos_sfr.txt', sep='\s+', engine= 'python', error_bad_lines=False, warn_bad_lines=True)
    cosmos = clean_data(cosmos)
    cosmos['#Seq'] = convert_to_numeric(cosmos['#Seq'])
    cosmos_sfr = cosmos[cosmos['#Seq'] == float(zfourge_ID)]
    
    # obtained observed colours
    # J1 corresponds to [340]
    # Hl corresponds to [550]
    lmass= cosmos_sfr.iloc[:,2].item()
    lsfr= cosmos_sfr.iloc[:,3].item()

    return lmass, lsfr        

# function to get restframe fluxes in J1, Hl, g and r filters 
def get_restframe(ID):
    ID= str(ID)
    # open crossmatch id file and identify line with ID
    crossmatch= pd.read_csv('/Users/wakami/Desktop/ucl/2023:2024/msc_thesis/zfire_survey/dr1/only_1D/data_files/cat_comp/id_crossmatch.dat', sep='\t', engine= 'python', error_bad_lines=False, warn_bad_lines=True)
    crossmatch= clean_data(crossmatch)
    crossmatch['ID'] = convert_to_numeric(crossmatch['ID'])
    zfire_crossmatch = crossmatch[crossmatch['ID'] == int(ID)]
    
    # obtain zfourge id
    zfourge_ID= zfire_crossmatch.iloc[:,2].item()
    zfourge_ID = str(zfourge_ID).strip()
    
    # open zfourge file
    cosmos= pd.read_csv('/Users/wakami/Desktop/ucl/2023:2024/msc_thesis/COSMOS_restframe.txt', sep='\s+', engine= 'python', error_bad_lines=False, warn_bad_lines=True)
    cosmos = clean_data(cosmos)
    cosmos['#Seq'] = convert_to_numeric(cosmos['#Seq'])
    cosmos_rest = cosmos[cosmos['#Seq'] == float(zfourge_ID)]
    
    # obtaine rest frame colours
    # J corresponds to [340]
    # H corresponds to [550]
    
    g_rest= cosmos_rest.iloc[:,1].item()
    r_rest= cosmos_rest.iloc[:,2].item()
    J_rest= cosmos_rest.iloc[:,3].item()
    H_rest= cosmos_rest.iloc[:,4].item()
    
    return g_rest, r_rest, J_rest, H_rest

# loop to apply functions to galaxies in ZFIRE sameple
with open('imf_output.txt', 'w') as file:
    for ID in zfire_spec['# ID']:
        median_data, log_ew = calc_EW(ID)
        if median_data is None:
            continue
        SNR = check_SNR(ID)
        
        if SNR<10:
            continue

        # obtain parameters from the ZFIRE and ZFOURGE dataset
        colour_J1, colour_Hl= get_colour(ID)
        z, z_err= get_redshift(ID)
        lmass, lsfr= get_sfr(ID)
        Av= get_Av(ID)
        g, r, J, H= get_restframe(ID)
        
        # calculate colour from flux
        JH_colour= -math.log(colour_J1/colour_Hl, 2.51)
        gr_restcolour= -math.log(float(g)/float(r), 2.51)
        JH_restcolour= -math.log(float(J)/float(H), 2.51)
        # apply dust correction and get dust-corrected colour
        dustcolour_J1= colour_J1 * (10**(0.4*1.56*Av))
        dustcolour_Hl= colour_Hl * (10**(0.4*1.00*Av))
        dustcorr_JHcolour= -math.log(dustcolour_J1/dustcolour_Hl, 2.51)
        f=2
        dustcorr_grcolour= gr_restcolour-(0.319*Av/f)
        dust_corr= log_ew + (0.4*Av*((0.62*f)-0.82))
        S, N= get_flux(ID)
        
        # calculate the Halpha-based SFR and it's error
        c= 3 * 10**5
        dist= (z * c)/70
        Mpc= 3.0856 * 10**22
        lum= (4 * np.pi * (dist * Mpc )**2 * S * 10**(-13)) * 10**(0.4*Av*0.817*f*0.247)
        sfr= 5.5 * (10**(-42)) * lum
        m_sun= 1.989 * (10**(30))
        yr= 60*60*24*365
        #sfr_units= sfr * (yr / m_sun)
        l_sfr= math.log(float(sfr),10)
        delta_sfr= np.sqrt((5.5 * (10**(-42)) * (4 * np.pi * (dist * Mpc )**2 * N * 10**(-13)) * 10**(0.4*Av*0.817*f*0.247))**2 + (z_err * 5.5 * (10**(-42))*(4 * np.pi * (dist * Mpc * c/70 )* 2 * S * 10**(-13)) * 10**(0.4*Av*0.817*f*0.247))**2)
        delta_lsfr= delta_sfr / sfr
        
        # output everything to a file
        file.write(f'{ID},{median_data},{log_ew},{SNR},{dust_corr},{JH_colour},{gr_restcolour},{JH_restcolour},{dustcorr_JHcolour},{dustcorr_grcolour},{z},{lmass},{l_sfr},{delta_lsfr},{lsfr}\n')
    
    