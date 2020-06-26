# -*- coding: utf-8 -*-
# %%
"""
Spyder Editor

This is a temporary script file.
"""
import os

from nipype import Node, Workflow #, MapNode

import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.modelgen as model  # model specification

import nipype.interfaces.utility as util  # utility
import nipype.interfaces.io as nio  # Data i/o
from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces import spm
from nipype.interfaces import fsl

# %%
data_dir = os.path.abspath('/media/Data/Lab_Projects/neurofeedback/neuroimaging/NF_BIDS/derivatives/fmriprep')
output_dir = '/media/Data/work/nf'
fwhm = 6
tr = 2.
removeTR = 0#Number of TR's to remove before initiating the analysis
lastTR = -1 # total number of frames in the scan, after removing removeTR (i.e. if we have a 500 frames scan and we removed 5 frames and the start of scan it should be 495, unless we also want to remove some from end of scan)
thr = 0.5 # scrubbing threshold
# %%
# set spm
spm.SPMCommand.set_mlab_paths(paths='/home/or/Downloads/spm12/',
 matlab_cmd='/home/or/matlab2018b/bin/matlab -nodesktop -nosplash')

# %% Methods
def _bids2nipypeinfo(in_file, events_file, regressors_file, removeTR,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0, thr=0.5 ):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from scrubFunc import scrub
    from nipype.interfaces.base.support import Bunch
    # Process the events file
    events = pd.read_csv(events_file)
    bunch_fields = ['onsets', 'durations', 'amplitudes']
    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]
    out_motion = Path('motion.par').resolve()
    #regress_data = pd.read_csv(regressors_file, sep=r'\s+')
    regress_data = scrub(regressors_file, thr) # grab also per which will be saved as file
    np.savetxt(out_motion, regress_data[motion_columns].values, '%g')
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))
    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']
    runinfo = Bunch(
        scans=in_file,
        conditions=list(set(events.trial_type.values)),
        **{k: [] for k in bunch_fields})
    for condition in runinfo.conditions:
        event = events[events.trial_type.str.match(condition)]
        runinfo.onsets.append(np.round(event.onset.values-removeTR, 3).tolist()) # added -removeTR to align to the onsets after removing X number of TRs from the scan
        runinfo.durations.append(np.round(event.duration.values, 3).tolist())
        if 'amplitudes' in events.columns:
            runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        else:
            runinfo.amplitudes.append([amplitude] * len(event))
    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        runinfo.regressors = regress_data[regressors_names].fillna(0.0).values[removeTR:,].T.tolist() # adding removeTR to cut the first rows
    return runinfo, str(out_motion)

def saveScrub(regressors_file, thr):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from scrubFunc import scrub
    # this function will call scrub and save a file with precentage of scrubbed framewise_displacement
    perFile = Path('percentScrub.txt').resolve()
    regress_data = pd.read_csv(regressors_file, sep=r'\s+')
    regress_data  = scrub(regressors_file, thr) # grab also per which will be saved as file
    x = regress_data.scrub
    per = np.array([sum(x)/len(x)])
    np.savetxt(perFile, per, '%g')
    return str(perFile)

# %%
subject_list = ['1263','1286','1319','1413','1450','1524','1525','1527','1528','1547','1551','1555'] # bad subject '1271', multiple runs - '1423', '030',
# Map field names to individual subject runs.


infosource = pe.Node(util.IdentityInterface(fields=['subject_id'],),
                  name="infosource")

infosource.iterables = [('subject_id', subject_list)]



# SelectFiles - to grab the data (alternativ to DataGrabber)
templates = {'func': os.path.join(data_dir, 'sub-{subject_id}', 'ses-{ses}', 'func', 'sub-{subject_id}_ses-{ses}_task-control{ctrlNum}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),
             'mask': os.path.join(data_dir, 'sub-{subject_id}', 'ses-{ses}', 'func', 'sub-{subject_id}_ses-{ses}_task-control{ctrlNum}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz'),
             'regressors': os.path.join(data_dir, 'sub-{subject_id}', 'ses-{ses}', 'func', 'sub-{subject_id}_ses-{ses}_task-control{ctrlNum}_desc-confounds_regressors.tsv'),
             'events': '/media/Data/Lab_Projects/neurofeedback/neuroimaging/events_file_{ctrlNum}.csv'
             }


selectfiles = pe.Node(nio.SelectFiles(templates,
                               base_directory=data_dir),
                   name="selectfiles")

selectfiles.inputs.ses = ['1','2'] 
selectfiles.inputs.ctrlNum = [1,2]#,3,4]     

# %%
# Extract motion parameters from regressors file
runinfo = pe.MapNode(util.Function(
    input_names=['in_file', 'events_file','regressors_file', 'regressors_names', 'removeTR', 'thr'],
    function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
    name='runinfo', iterfield = ['in_file','events_file', 'regressors_file'])

# Set the column names to be used from the confounds file
runinfo.inputs.regressors_names = ['std_dvars', 'framewise_displacement', 'scrub'] + \
                                   ['a_comp_cor_%02d' % i for i in range(6)]

runinfo.inputs.motion_columns   = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

runinfo.inputs.removeTR = 0
runinfo.inputs.thr = thr # set threshold of scrubbing
#runinfo.inputs.events_file = '/media/Data/Lab_Projects/neurofeedback/neuroimaging/events_file.csv'
# %%
## adding node for the saveScrub functions
svScrub = pe.MapNode(util.Function(
    input_names = ['regressors_file', 'thr'], output_names = ['perFile'],
    function = saveScrub), name = 'svScrub', iterfield = ['regressors_file']
    )

svScrub.inputs.thr = thr
# %%
extract = pe.MapNode(fsl.ExtractROI(), name="extract", iterfield = ['in_file'])
extract.inputs.t_min = removeTR
extract.inputs.t_size = lastTR # set length of sacn
extract.inputs.output_type='NIFTI'

# smoothing
smooth = Node(spm.Smooth(), name="smooth", fwhm = fwhm)


# set contrasts, depend on the condition
cond_names = ['silence','nback','trauma','strategy']


cont1 = ('trauma>silence', 'T', cond_names, [-1, 0, 1, 0])
cont2 = ('trauma>nback', 'T', cond_names, [0,-1, 1, 0])
cont3 = ('trauma>strategy', 'T', cond_names, [0, 0, 1, -1])
cont4 = ('trauma', 'T', cond_names, [0, 0, 1, 0])


contrasts = [cont1, cont2, cont3, cont4]
# %%

modelspec = Node(interface=model.SpecifySPMModel(), name="modelspec")
modelspec.inputs.concatenate_runs = False
modelspec.inputs.input_units = 'secs' # supposedly it means tr
modelspec.inputs.output_units = 'secs'
modelspec.inputs.time_repetition = tr  # make sure its with a dot
modelspec.inputs.high_pass_filter_cutoff = 128.

level1design = pe.Node(interface=spm.Level1Design(), name="level1design") #, base_dir = '/media/Data/work')
level1design.inputs.timing_units = modelspec.inputs.output_units
level1design.inputs.interscan_interval = tr
level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
level1design.inputs.model_serial_correlations = 'AR(1)'

# create workflow
wfSPM = Workflow(name="l1spm_resp", base_dir=output_dir)
wfSPM.connect([
        (infosource, selectfiles, [('subject_id', 'subject_id')]),
        (selectfiles, runinfo, [('events','events_file'),('regressors','regressors_file')]),
        (selectfiles, svScrub, [('regressors', 'regressors_file')]),
        (selectfiles, extract, [('func','in_file')]),
        (extract, smooth, [('roi_file','in_files')]),
        (smooth, runinfo, [('smoothed_files','in_file')]),
        (smooth, modelspec, [('smoothed_files', 'functional_runs')]),
        (runinfo, modelspec, [('info', 'subject_info'), ('realign_file', 'realignment_parameters')]),
        ])
wfSPM.connect([(modelspec, level1design, [("session_info", "session_info")])])

# %%
level1estimate = pe.Node(interface=spm.EstimateModel(), name="level1estimate")
level1estimate.inputs.estimation_method = {'Classical': 1}

contrastestimate = pe.Node(
    interface=spm.EstimateContrast(), name="contrastestimate")
contrastestimate.inputs.contrasts = contrasts


wfSPM.connect([
         (level1design, level1estimate, [('spm_mat_file','spm_mat_file')]),
         (level1estimate, contrastestimate,
            [('spm_mat_file', 'spm_mat_file'), ('beta_images', 'beta_images'),
            ('residual_image', 'residual_image')]),
    ])

# %% Adding data sink
# Datasink
datasink = Node(nio.DataSink(base_directory=os.path.join(output_dir, 'Sink_resp_spm')),
                                         name="datasink")


wfSPM.connect([
        (level1estimate, datasink, [('beta_images',  '1stLevel.@betas.@beta_images'),
                                    ('residual_image', '1stLevel.@betas.@residual_image'),
                                    ('residual_images', '1stLevel.@betas.@residual_images'),
                                    ('SDerror', '1stLevel.@betas.@SDerror'),
                                    ('SDbetas', '1stLevel.@betas.@SDbetas'),
                ])
        ])


wfSPM.connect([
       # here we take only the contrast ad spm.mat files of each subject and put it in different folder. It is more convenient like that.
       (contrastestimate, datasink, [('spm_mat_file', '1stLevel.@spm_mat'),
                                              ('spmT_images', '1stLevel.@T'),
                                              ('con_images', '1stLevel.@con'),
                                              ('spmF_images', '1stLevel.@F'),
                                              ('ess_images', '1stLevel.@ess'),
                                              ])
        ])

# %% run
wfSPM.run('MultiProc', plugin_args={'n_procs': 10})
# wfSPM.run('Linear', plugin_args={'n_procs': 1})
