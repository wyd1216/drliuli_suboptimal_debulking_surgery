#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Yudong Wang 
# Created Date: 25/3/2022 Tues.
# version ='1.0'
# ---------------------------------------------------------------------------
""" 
    This project was built for:
    1. Satisfactory reduction model of serous cystadenocarcinoma of ovary based on MR imaging.
    2. Radiomic feature extraction
"""  
# ---------------------------------------------------------------------------
import subprocess
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def wprint(string, style=0):
    if len(string) < 57:
        tot_len = 60
    elif len(string) < 77:
        tot_len = 80
    else:
        tot_len = len(string)+6
    space_len = (tot_len - len(string) - 6) // 2
    print('='*tot_len)
    print('=',' '*space_len, f'\033[1;97;95m{string}\033[0m', ' '*(tot_len-space_len-6-len(string)),'=')
    print('='*tot_len)

if __name__ == '__main__':
    cwd = os.getcwd()
    outdata_dir = os.path.join(cwd,'Data_features')     # Directory in relative path to save output features.
    print(outdata_dir)
    random_state = 2022                                 # Random seed
    sequence_id = [2, 3, 4]                             # The digital id of label T2, DWI, T1CE sequences.
    tag_cols = ['pid', 'label', 'series', 'image', 'mask']   # The columns which are not features while saved in the data to tag data.
    icc_score_threshold = 0.95              # Threshold for the repeatability test by ICC method.
    # Input files of train data
    all_data_info = pd.read_csv('../DataPreprocess/dataset_info/all_data_info.csv')
    icc_data_info = pd.read_csv('../DataPreprocess/dataset_info/icc_data_info.csv')
    external_data_info = pd.read_csv('../DataPreprocess/dataset_info/external_data_info.csv')
    # Split the data into 'dicom' and 'nii' groups.
    data_dicom = all_data_info[all_data_info['img_type']=='dicom']
    data_nii = all_data_info[all_data_info['img_type']=='nii']
    print(len(data_dicom))
    print(len(data_nii))
    icc_dicom = icc_data_info[icc_data_info['img_type']=='dicom']
    icc_nii = icc_data_info[icc_data_info['img_type']=='nii']

    # Name the 4 groups data storage-information files .
    fdata_dicom = os.path.join(cwd, outdata_dir, 'data_info_dicom.csv')
    fdata_nii = os.path.join(cwd, outdata_dir, 'data_info_nii.csv')
    ficc_dicom = os.path.join(cwd, outdata_dir, 'icc_info_dicom.csv')
    ficc_nii = os.path.join(cwd, outdata_dir, 'icc_info_nii.csv')
    fexternal_dicom = os.path.join(cwd, outdata_dir, 'external_data_info.csv')
    data_dicom.to_csv(fdata_dicom, index=0)
    data_nii.to_csv(fdata_nii, index=0)
    icc_dicom.to_csv(ficc_dicom, index=0)
    icc_nii.to_csv(ficc_nii, index=0)

    # Name the 4 group output data storage-features files.
    fdata_feas_dicom = os.path.join(cwd, outdata_dir, 'data_feas_dicom.csv')
    fdata_feas_nii = os.path.join(cwd, outdata_dir, 'data_feas_nii.csv')
    ficc_feas_dicom = os.path.join(cwd, outdata_dir, 'icc_feas_dicom.csv')
    ficc_feas_nii = os.path.join(cwd, outdata_dir, 'icc_feas_nii.csv')
    fexternal_feas = os.path.join(cwd, outdata_dir, 'external_feas.csv')

    print(len(fdata_feas_dicom)+len(fdata_feas_nii))
    # ---------------------------------------Features Extraction----------------------------------------
    # Output files for conserve the extracted features.
    radiomics_script = os.path.join(cwd, 'radiomics/extract.py')
    radiomics_select = os.path.join(cwd, 'radiomics/feature_filter.py')

    print('='*20, 'Start extracting the radiomics features', '='*20)
    subprocess.call('python {} --data_csv {} --output {} --img_reader {}'.format(radiomics_script, fexternal_dicom, fexternal_feas, 'dicom'), shell=True)
    subprocess.call('python {} --data_csv {} --output {} --img_reader {}'.format(radiomics_script, fdata_dicom, fdata_feas_dicom, 'dicom'), shell=True)
    subprocess.call('python {} --data_csv {} --output {} --img_reader {}'.format(radiomics_script, fdata_nii, fdata_feas_nii, 'nii'), shell=True)
    subprocess.call('python {} --data_csv {} --output {} --img_reader {}'.format(radiomics_script, ficc_dicom, ficc_feas_dicom, 'dicom'), shell=True)
    subprocess.call('python {} --data_csv {} --output {} --img_reader {}'.format(radiomics_script, ficc_nii, ficc_feas_nii, 'nii'), shell=True)
    print('='*20, 'Extracting the radiomics features completed', '='*20)
    ''' Comment for timesaving
    '''
    sys.exit(0)
