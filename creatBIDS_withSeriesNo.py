#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Fri Feb 15 15:02:13 2019

@author: Or Duek
A short script that will convert to NIFTI.GZ (from raw DICOM data) and then create a BIDS compatible structure
"""

# convert to NIFTI
import os   
from nipype.interfaces.dcm2nii import Dcm2niix
import shutil

# %% Convert functions Converts DICOM to NIFTI.GZ
def convert (source_dir, output_dir, subName, session): # this is a function that takes input directory, output directory and subject name and then converts everything accordingly
    try:
        os.makedirs(os.path.join(output_dir, subName, session))
    except:
        print ("folder already there")
#    try:
#       os.makedirs(os.path.join(output_dir, subName, ))
#    except:
#       print("Folder Exist")    
    converter = Dcm2niix()
    converter.inputs.source_dir = source_dir
    converter.inputs.compression = 7
    converter.inputs.output_dir = os.path.join(output_dir, subName, session)
    converter.inputs.out_filename = subName + 'seriesNo' '_' + '%2s' + '%p'
    converter.run()

# %% Check functions
def checkGz (extension):
     # check if nifti gz or something else
    if extension[1] =='.gz':
        return '.nii.gz'
    else:
        return extension[1]

def checkTask(filename):
	
    nameTask = filename.split('seriesNo')[1].split('mrrc')[0].replace('-', '')	    
	
    return nameTask


# %%
def organizeFiles(output_dir, subName, session):
    
    fullPath = os.path.join(output_dir, subName, session)
    os.makedirs(fullPath + '/dwi')
    os.makedirs(fullPath + '/anat')    
    os.makedirs(fullPath + '/func')
    os.makedirs(fullPath + '/misc')    
    
    a = next(os.walk(fullPath)) # list the subfolders under subject name

    # run through the possibilities and match directory with scan number (day)
    for n in a[2]:
        print (n)
        b = os.path.splitext(n)
        # add method to find (MB**) in filename and scrape it
        if n.find('diff')!=-1:
            print ('This file is DWI')
            shutil.move((fullPath +'/' + n), fullPath + '/dwi/' + n)
            os.rename((os.path.join(fullPath, 'dwi' ,n)), (fullPath + '/' + 'dwi' +'/' + subName + '_' + session +'_dwi' + checkGz(b)))
   
        elif n.find('MPRAGE')!=-1:
            print (n + ' Is Anat')
            shutil.move((fullPath + '/' + n), (fullPath + '/anat/' + n))
            os.rename(os.path.join(fullPath,'anat' , n), (fullPath + '/anat/' + subName+ '_' + session + '_acq-mprage_T1w' + checkGz(b)))
        elif n.find('t1_flash')!=-1:
            print (n + ' Is Anat')
            shutil.move((fullPath + '/' + n), (fullPath + '/anat/' + n))
            os.rename(os.path.join(fullPath,'anat' , n), (fullPath + '/anat/' + subName+ '_' + session + '_acq-flash_T1w' + checkGz(b)))
        elif n.find('t1_fl2d')!=-1:
            print (n + ' Is Anat')
            shutil.move((fullPath + '/' + n), (fullPath + '/anat/' + n))
            os.rename(os.path.join(fullPath,'anat' , n), (fullPath + '/anat/' + subName+ '_' + session + '_acq-fl2d1_T1w' + checkGz(b))) 
        elif n.find('GRE_3D_Sag_Spoiled')!=-1:
            print (n + ' Is Anat')
            shutil.move((fullPath + '/' + n), (fullPath + '/anat/' + n))
            os.rename(os.path.join(fullPath,'anat' , n), (fullPath + '/anat/' + subName+ '_' + session + '_acq-gre_spoiled_T1w' + checkGz(b)))            
        elif n.find('bold')!=-1:
            print(n  + ' Is functional')
            taskName = checkTask(n)
            shutil.move((fullPath + '/' + n), (fullPath + '/func/' + n))
            os.rename(os.path.join(fullPath, 'func', n), (fullPath  + '/func/' +subName+'_' +session + '_task-' + taskName + '_bold' + checkGz(b)))
        else:
            print (n + 'Is MISC')
            shutil.move((fullPath + '/' + n), (fullPath + '/misc/' + n))
           # os.rename(os.path.join(fullPath, 'misc', n), (fullPath +'/misc/' +'sub-'+subName+'_ses-' +sessionNum + '_MISC' + checkGz(b)))
    
# need to run thorugh misc folder and extract t1's when there is no MPRAGE - Need to solve issue with t1 - as adding the names is not validated with BIDS

# %%
sessionDict = {

      'ses-1': '/media/Data/Lab_Projects/neurofeedback/neuroimaging/raw_scan_data/NF1552/nf1552_scan1_pb10223_localizer',
'ses-2': '/media/Data/Lab_Projects/neurofeedback/neuroimaging/raw_scan_data/NF1552/nf1552_scan2_pb10392_controltask',
#'ses-3': '/media/Data/Lab_Projects/neurofeedback/neuroimaging/raw_scan_data/NF1551/nf1551_scan3_pb9903',
#'ses-4': '/media/Data/Lab_Projects/neurofeedback/neuroimaging/raw_scan_data/NF1551/nf1551_scan4_pb9940',
#'ses-5': '/media/Data/Lab_Projects/neurofeedback/neuroimaging/raw_scan_data/NF1551/nf1551_scan5_pb9978',
'ses-6': '/media/Data/Lab_Projects/neurofeedback/neuroimaging/raw_scan_data/NF1552/nf1552_scan6_pb10570_post_assess_controltask',
#    'ses-7': '/media/Data/Lab_Projects/neurofeedback/neuroimaging/raw_scan_data/NF1555/nf1555_scan7_pb10205',
    #'ses-8': '/media/Data/Lab_Projects/neurofeedback/neuroimaging/raw_scan_data/NF1527/nf1527_scan8_pb9850'
        }
subNumber = '1552'
def fullBids(subNumber, sessionDict):
    output_dir = '/media/Data/Lab_Projects/neurofeedback/neuroimaging/NF_BIDS'
    subName = 'sub-' + subNumber
  #  folder_name = ['anat','func','dwi','other']
    
    for i in sessionDict:
        session = i
        source_dir = sessionDict[i]
        print (session, source_dir)
        fullPath = os.path.join(output_dir, subName, session)
        print(fullPath)
        convert(source_dir,  output_dir, subName, session)
        organizeFiles(output_dir, subName, session)        
        
    
    #print (v)
# %%
fullBids(subNumber, sessionDict)

# %%

