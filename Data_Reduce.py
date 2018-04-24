"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import random

# Load data from files
cat = pd.read_csv('../Data/cdfs.v1.6.9.cat', delimiter=' ')
radio = pd.read_csv('../Data/cdfs.v1.6.9.radio.v0.3.cat', delimiter=' ')
fout = pd.read_csv('../Data/cdfs.v1.6.9.fout', delimiter=' ')
ir = pd.read_csv('../Data/cdfs.v1.6.9.herschel.v0.4.cat', delimiter=' ')
xray = pd.read_csv('../Data/cdfs.v1.6.9.xray.v0.4.cat', delimiter=' ')
agn = pd.read_csv('../Data/cdfs.v1.6.9.agn.v0.5.cat', delimiter=' ')


# Join data together
full_data = pd.concat([cat, radio, fout, ir, xray, agn], axis=1)


# Unifying missing data flags to NA
full_data_uni = full_data.replace(to_replace=-99, value=np.nan)
full_data_uni = full_data_uni.replace(to_replace=-1, value=np.nan)


# Drop unneeded columns
list(full_data_uni)
dropped_data = full_data_uni.drop(['Unnamed: 0', 'Unnamed: 5', 'id', 'x', 'y', 'ra', 'dec', 'SEflags', 'iso_area', 
                              'fap_Ksall', 'eap_Ksall', 'apcorr', 'Ks_ratio', 'fapcirc07_Ksall', 'eapcirc07_Ksall', 
                              'apcorr07', 'fcirc07_Ksall', 'ecirc07_Ksall', 'fauto_Ksall', 'flux50_radius', 'a_vector',
                              'b_vector', 'kron_radius','l_14','LIR','l_xray','chi2','la2t','L2800','nearstar','snr', 
                              'f_F098M','e_F098M','f_xray', 'z_spec', 'HR', 'Av', 'metal'], axis=1)




dropped_data['classes']=0
dropped_data['ir_agn']=dropped_data.ir_agn.fillna(0)
dropped_data['radio_agn']=dropped_data.radio_agn.fillna(0)
dropped_data['xray_agn']=dropped_data.xray_agn.fillna(0)
dropped_data['classes']=[a+2*b+4*c for a,b,c in zip(dropped_data.ir_agn, dropped_data.xray_agn, dropped_data.radio_agn)]
#now remove entries where class==0
#dropped_data_classes = dropped_data[dropped_data.classes != 0]



for column in list(dropped_data):
    if column.startswith('w'):
        dropped_data = dropped_data.drop(column, axis=1)
    
    
list(dropped_data)
    


# Limit to useflag == 1
dropped_data = dropped_data[dropped_data.use == 1]

# For each flux value if its less than 3 times teh rms set it as a uniform random value between 3 * rms and 0.
for column in list(dropped_data):
    
    if column.startswith('f_'):
        print(column)
        if column != 'f_xray':
            flux = dropped_data[column]
            err_col = 'e_'+'_'.join(column.split('_')[1:])
            print(err_col)
            error = dropped_data[err_col]
            
            # Replace non detections with normed flux
            new_flux = [10**random.uniform(-9, e) if f <= 3*e else f for e, f in zip(error, flux)]
            
            
            if column == 'f_14':
                new_flux = [10**random.uniform(-3, 1) if np.isnan(f) else f for f in new_flux]
            
            dropped_data[column] = new_flux
                
        elif column == 'f_xray':
            flux = dropped_data[column]
            new_flux = [10**random.uniform(-18, -17) if np.isnan(f) else f for f in flux]
        
        


for column in list(dropped_data):
    if column.startswith('e'):
        dropped_data = dropped_data.drop(column, axis=1)
    


# Limit to useflag == 1
final_data = dropped_data[dropped_data.use == 1]


final_final_data = final_data.dropna()



