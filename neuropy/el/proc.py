"""Data processing module for Energy Landscape data."""

# Import modules
import numpy as np
import pandas as pd


def clean_trajectory_data(df):
    """Clean converted TrajectoryData from MATLAB.

    Perform the following cleaning operations:
    spikes -- convert to DataFrame
    stateOffset -- convert to scalar
    stateOnset -- convert to scalar
    successful -- convert to boolean
    tag -- convert to scalar
    targetRadius -- convert to scalar
    trajectoryOnset -- convert to scalar
    trajectoryOffset -- convert to scalar
    trialID -- convert to scalar

    """
    # Convert spikes to dataframe
    df_cols = ['spikes']
    for col in df_cols:
        df[col] = df[col].apply(lambda x: pd.DataFrame(x))

    # Convert columns to scalar values
    scalar_cols = [
        'stateOnset',
        'stateOffset',
        'successful',
        'tag',
        'targSize',
        'trajOnset',
        'trajOffset',
        'trialID',
        'decoderBinWidth',
        'targetAcquired',
        'trialLen',
        'goCue'
    ]
    for col in scalar_cols:
        df[col] = df[col].apply(np.asscalar)
    
    # Convert successful to boolean
    df['successful'] = df['successful'].apply(bool)

    # Remove any unused/unnecessary data fields
    drop_cols = [
        'GPFA', 'averaged', 'binTime', 'binnedChannel', 'binnedSort',
        'binnedSpikes', 'brainControlPos', 'brainControlVel', 'brainKin',
        'controlSource', 'forceKin', 'handKin', 'moveOffset', 'moveOnset',
        'normalized', 'cursorSize', 'datasetName', 'targetCode'
    ]
    df.drop(columns=drop_cols, inplace=True)

    return None


def clean_bci_params(params):
    """Clean BCI parameters converted from MATLAB."""

    # Fields to convert to integers
    int_fields = [
        'Tmax',
        'accelerationScale',
        'binwidth',
        'decoderID',
        'dimensions',
        'exclCh',
        'nBins',
        'nTs',
        'parallelScale',
        'perpendicularScale',
        'segLength',
        'tWin',
        'validSort',
        'xDim'
    ]
    for field in int_fields:
        try:
            params[field] = params[field].astype('int')
        except:
            print('Warning: {} not found.'.format(field))

    # Fields to convert to scalar
    scalar_fields = [
        'Tmax',
        'binwidth',
        'causalGPFA',
        'decoderID',
        'dimensions',
        'includePosition',
        'includeVelocity',
        'includeAcceleration',
        'nBins',
        'nTs',
        'parallelScale',
        'perpendicularScale',
        'removeJaggedDim',
        'rotation',
        'segLength',
        'tWin',
        'velocityScale',
        'accelerationScale',
        'xDim'
    ]
    for field in scalar_fields:
        try:
            params[field] = params[field].item()
        except:
            print('Warning: {} not found.'.format(field))

    # Transpose arrays. For some reason, conversion transposes all of the
    # arrays.
    for k, v in params.items():
        if type(v) == np.ndarray:
            params[k] = v.T

    return None
    

def clean_gpfa_params(params):
    """Clean GPFA parameters converted from MATLAB."""

    # Transpose arrays
    for k, v in params.items():
        if type(v) == np.ndarray:
            params[k] = v.T

    return None
    


