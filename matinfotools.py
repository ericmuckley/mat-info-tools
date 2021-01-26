#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 4 13:22:37 2020

@author: emuckley


This module contains material informatics tools for:
- plotting
- chemical featurization
- file I/O


"""

# standard imports
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import stats

# for compiling plots into video
try:
    import pygifsicle
    import imageio
    import glob
    import json
except ImportError:
    print('Video packages not avilable')
    

try:
    # for featurizing materials data
    from matminer.featurizers.conversions import (
        StrToComposition, CompositionToOxidComposition)
    from matminer.featurizers.structure import DensityFeatures
    from matminer.featurizers.composition import (
        IonProperty, ElementFraction, TMetalFraction,
        Miedema, YangSolidSolution, AtomicPackingEfficiency,
        ElementProperty, Meredig, OxidationStates, AtomicOrbitals,
        BandCenter, ElectronegativityDiff, ElectronAffinity, Stoichiometry,
        CohesiveEnergy, CohesiveEnergyMP,
        # ValenceOrbital,
        )
except ImportError:
    print('Matminer not avilable')


# change matplotlib settings to make plots look nicer
FONTSIZE=16
LINEWIDTH=2
TICKWIDTH=2
plt.rcParams.update({
    'xtick.labelsize': FONTSIZE,
    'ytick.labelsize': FONTSIZE,
    'axes.linewidth': LINEWIDTH,
    'xtick.minor.width': TICKWIDTH,
    'xtick.major.width': TICKWIDTH,
    'ytick.minor.width': TICKWIDTH,
    'ytick.major.width': TICKWIDTH,
    'figure.facecolor': 'w',
    #'figure.dpi': dpi,
})



def create_polynomial_features(df, cols, target, order=3, display=100):
    """
    Generate new features (columns) of a dataframe using polynomial
    combinations of existing column names. The purpose is to try to
    find new features which are highly correlated with a target column
    in the dataframe.
    Inputs:
    df: dataframe
    cols: list of column names to combine
    target: target column for which to correlate with new features
    order: polynomial order for combinations (over three takes hours)
    display: number of new feature correlations to print upon completion
    Returns: Dict with new column names sorted by their r^2 correlation with target
    """
    starttime = time.time()
    # instantiate polynomial feature model
    polyfeats = PolynomialFeatures(order, include_bias=False)
    # inputs for poly feats 
    inputs = df[cols]
    # get names and array of values of new features
    polyfeat_array = polyfeats.fit_transform(inputs.values)
    polyfeat_names = polyfeats.get_feature_names(inputs.columns)
    # loop over new features and get their correlation to target variable
    polyfeat_dict = {}
    for i, n in enumerate(polyfeat_names):
        # if feature looks new, save it to dict with its correlation coefficient
        if not any([np.allclose(polyfeat_array[:, i], df[j]) for j in cols]):
            # find r^2 correlation of new feature with target
            r2 = np.square(stats.pearsonr(polyfeat_array[:, i], df[target])[0])
            polyfeat_dict[n] = r2
    # sort polyfeature dict by correlations
    polyfeat_dict = {k: v for k, v in sorted(
        polyfeat_dict.items(), key=lambda item: item[1], reverse=True)}
    print('Total runtime: {} min'.format(round((time.time() - starttime)/60, 2)))
    print('{} total Pearson r^2 correlation coefficients:'.format(len(polyfeat_dict)))
    for k in list(polyfeat_dict)[:100]:
        print('{:6}   {}'.format(round(polyfeat_dict[k], 5), k))
    return polyfeat_dict


   
def read_json_file(filepath):
    """Import JSON file as a Python dictionary"""
    try:
        with open(filepath, encoding='utf-8') as j:
            d = json.load(j)
    except UnicodeDecodeError:
        with open(filepath, encoding='latin-1') as j:
            d = json.load(j)
    return d
   

def hide_plot_borders():
    """Hide plot matplotlib borders and ticks. Run this
    function just before setting plot axis labels and
    before plt.show()."""
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    
    
    
def make_video(imagelist, fps=8, video_name='vid.mp4', reverse=False):
    """Create a video from a stack of images.
    For video name, use .mp4 or .gif extension."""  
    if not os.path.exists('videos'):
            os.makedirs('videos')
    # sort and reverse if desired
    imagelist = sorted(imagelist)
    imagelist = imagelist[::-1] if reverse else imagelist
    # loop over each frame and add to video
    with imageio.get_writer(
        os.path.join('videos', video_name),
        mode='I',
        fps=fps) as writer:
        for i in range(len(imagelist)):
            img = imageio.imread(imagelist[i])
            try:
                writer.append_data(img)
            except ValueError:
                print('{} failed'.format(imagelist[i]))
    # optimize to decrease gif file size (this is not required)
    if video_name.endswith('gif'):
        pygifsicle.optimize(os.path.join('videos', video_name))   
    
 
    
def plot_setup(
    xlabel=False,
    ylabel=False,
    title=False,         
    fsize=16, 
    legend_fontsize=12,
    limits=False,
    size=False,
    legend=False,
    save=False,
    filename='plot.jpg',):
    """Creates a custom plot configuration to make graphs look nice.
    This can be called with matplotlib for setting axes labels,
    titles, axes ranges, and the font size of plot labels.
    This should be called between plt.plot() and plt.show() commands."""
    fig = plt.gcf()
    if xlabel:
        plt.xlabel(str(xlabel), fontsize=fsize)
    if ylabel:
        plt.ylabel(str(ylabel), fontsize=fsize)
    if title:
        plt.title(str(title), fontsize=fsize)
    if size:
        fig = plt.gcf()
        fig.set_size_inches(size[0], size[1])
    if legend:
        plt.legend(fontsize=legend_fontsize)
    if limits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if save:
        plt.tight_layout()
        fig.savefig(filename,
                    dpi=250,
                    bbox_inches='tight')



def bar_plot(x, y, n=50, width=0.5, rotation=90):
    """
    Create bar plot instance. To change color of bars:
    for b in bars:
        b.set_color('r')
    Use 'n' argument to limit the number of bars shown.
    """
    x, y = x[:n], y[:n]
    ha = 'center' if (rotation == 90) else 'right'
    bars = plt.bar(np.arange(len(x)), y, width=width)
    plt.gca().set_xticks(np.arange(len(x)))
    plt.gca().set_xticklabels(list(x), rotation=rotation, ha=ha)
    plt.gca().set_xlim([-1, len(x)])
    return bars


def bar_plot_h(x, y, n=30, height=0.5):
    """
    Create horizonatal bar plot instance. To change color of bars:
    for b in bars:
        b.set_color('r')
    Use 'n' argument to limit the number of bars shown.
    """
    x, y = x[:n], y[:n]
    bars = plt.barh(np.arange(len(x)), y[::-1], height=0.5)
    plt.gca().set_yticks(np.arange(len(x)))
    plt.gca().set_yticklabels(list(x)[::-1])
    plt.gca().set_ylim([-1, len(x)])
    return bars[::-1]

        
def df_to_heatmap(df, vmin=None, vmax=None, fontsize=14, colorbar=True,
                  title=None, size=None, gridlines=True, show=False,
                  gridcolor='gray', cmap='jet',
                  savefig=False, filename='fig.jpg'):
    '''
    Plot a heatmap from 2D data in a Pandas DataFrame. The y-axis labels 
    should be index names, and x-axis labels should be column names.
    '''
    # create plot
    heatmap = plt.imshow(
        df.values.astype(float),
        cmap=cmap, vmin=vmin, vmax=vmax)
    # plot labels on each axis
    plt.yticks(np.arange(0, len(df.index), 1),
               df.index, fontsize=fontsize)
    plt.xticks(np.arange(0, len(df.columns), 1),
               df.columns, rotation='vertical',
               fontsize=fontsize)
    if gridlines:
        # create minor ticks, hide them, and use them as grid lines
        plt.gca().set_xticks(
            [
                x - 0.5 for x in plt.gca().get_xticks()][1:],
            minor='true')
        plt.gca().set_yticks(
            [
                y - 0.5 for y in plt.gca().get_yticks()][1:],
            minor='true')
        plt.gca().tick_params(which='minor', length=0, color='k')
        plt.grid(which='minor', color=gridcolor, lw=0.5)
    if title:
        plt.title(title, fontsize=fontsize+4)
    if size:
        plt.gcf().set_size_inches(size[0], size[1])
    if colorbar:
        plt.colorbar()
    if savefig:
        plt.gcf().savefig(filename,
                          dpi=120,
                          facecolor=fig.get_facecolor(),
                          edgecolor='none',
                          bbox_inches='tight')
        plt.tight_layout()
    if show:
        plt.show()
    else:
        return heatmap       
        
     
    
def plot_periodic_table(lw=1, c='k', alpha=1):
    """Create plot of periodic table lines for overlaying chemical data"""
    # coordinates for lines on the periodic table
    table_lines = [
        # horizontal lines
        [[0.5, 1.5], [0.5, 0.5]],
        [[0.5, 2.5], [1.5, 1.5]],
        [[0.5, 2.5], [2.5, 2.5]],
        [[17.5, 18.5], [0.5, 0.5]],
        [[12.5, 18.5], [1.5, 1.5]],
        [[12.5, 18.5], [2.5, 2.5]],
        [[0.5, 18.5], [3.5, 3.5]],
        [[0.5, 18.5], [4.5, 4.5]],
        [[0.5, 18.5], [5.5, 5.5]],
        [[0.5, 18.5], [6.5, 6.5]],
        [[0.5, 18.5], [7.5, 7.5]],
        # vertical lines
        [[0.5, 0.5], [0.5, 7.5]],    
        [[1.5, 1.5], [0.5, 7.5]],    
        [[2.5, 2.5], [1.5, 7.5]],  
        [[3.5, 3.5], [3.5, 7.5]],
        [[4.5, 4.5], [3.5, 7.5]],
        [[5.5, 5.5], [3.5, 7.5]],
        [[6.5, 6.5], [3.5, 7.5]],
        [[7.5, 7.5], [3.5, 7.5]],
        [[8.5, 8.5], [3.5, 7.5]],
        [[9.5, 9.5], [3.5, 7.5]],
        [[10.5, 10.5], [3.5, 7.5]],
        [[11.5, 11.5], [3.5, 7.5]],
        [[12.5, 12.5], [1.5, 7.5]],
        [[13.5, 13.5], [1.5, 7.5]],
        [[14.5, 14.5], [1.5, 7.5]],
        [[15.5, 15.5], [1.5, 7.5]],
        [[16.5, 16.5], [1.5, 7.5]],
        [[17.5, 17.5], [0.5, 7.5]],
        [[18.5, 18.5], [0.5, 7.5]],]
    for tl in table_lines:
        plt.plot(tl[0], tl[1], c=c, lw=lw, alpha=alpha)
    plt.xlim([0, 19])
    plt.ylim([8, 0])    
    
    
    
        
        
        

def export_df(df, filename, directory='data', export_index=True):
    """Export Pandas dataframe to CSV file"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(os.path.join(directory, filename),
              index=export_index)

    
def import_df(filename, directory, index_col=0):
    """Import Pandas dataframe from CSV file"""
    return pd.read_csv(
        os.path.join(directory, filename),
        index_col=index_col)


def get_rmse(errors, rounding=3):
    """Get the RMSE fo a sequence of errors"""
    errors = np.array(errors).astype(np.float)
    return np.round(
        np.sqrt(np.mean(np.square(errors))),
        decimals=rounding)


def normalize_vec(vec):
    """Normalize a 1D vector from 0 to 1"""
    return (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
    

def norm_df(df0, ignore_cols=[]):
    """Normalize all columns of a pandas dataframe. Ignore string columns and
    constant columns, and columns in ignore_cols argument."""
    df = df0.copy()
    # loop over each dataframe column
    for c in [cc for cc in df.columns if cc not in ignore_cols]:
        # if column is not constant
        if df[c].min() != df[c].max():
            # try normalizing
            try:
                df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
            # if it is a string column, ignore it
            except TypeError:
                pass
        # if column is constant, ignore it
        else:
            pass
    return df




def featurize(
    df,
    formula_col='formula',
    pbar=True,
    remove_nan_cols=True,
    remove_constant_cols=True,
    n_jobs=1,
    n_chunksize=None,
    fast=False,):
    """
    Featurization of chemical formulas for machine learning.
    Input a Pandas Dataframe with a column called formula_col,
    which contains chemical formulas (e.g. ['Mg', 'TiO2']).
    Other columns may contain additional descriptors.
    The chemical formulas are featurized according to methods
    from the matminer package. Returns the dataframe with chemical
    formulas and features, and a list of references
    to papers which describe the featurization methods used.
    
    Use 'fast' argument to run featurization with less
    features, but very quickly for large datasets.

    To prevent issues with multithreading, set n_jobs=1.

    To ignore certain features, comment them out in the
    'composition_features' or 'composition_ox_features' lists.

    Use the kwargs to return the list of references used,
    remove dataframe columns which are constant or
    remove dataframe columns which contain nans.

    ================= Useful links =======================
    Matminer summary table of features:
    https://hackingmaterials.lbl.gov/matminer/featurizer_summary
    Matminer Github repo:
    https://github.com/hackingmaterials/matminer
    Matminer notebook examples:
    https://github.com/hackingmaterials/matminer_examples
    """
    starttime = time.time()
    if formula_col not in list(df):
        raise KeyError(
            'Data does not contain {} column.'.format(formula_col))
    # drop duplicate chemical formulas
    feat = df.drop_duplicates(subset=formula_col)   
    print('Featurizing dataset...')
    # create composition column from the chemical formula
    stc = StrToComposition()
    stc.set_n_jobs(1)
    feat = stc.featurize_dataframe(
        feat,
        formula_col,
        ignore_errors=True,
        pbar=pbar)
    
    if not fast:
        # create oxide composition and add oxidation state
        # this line hangs on large molecules! (e.g. 'C60')
        ctoc = CompositionToOxidComposition()
        ctoc.set_n_jobs(1)
        feat = ctoc.featurize_dataframe(
            feat,
            "composition",
            ignore_errors=True,
            pbar=pbar)
        
    # add element property featurizer
    element_property = ElementProperty.from_preset(
        preset_name="magpie")

    # basic features
    feat_dict = {'composition': [
        element_property,
        BandCenter(),
        Stoichiometry(),
        AtomicOrbitals(),
        #TMetalFraction(),
        #ElementFraction(),
        #Miedema(),
        #YangSolidSolution(),
    ]}

    # add more features for slower featurization
    if not fast:
        feat_dict['composition'] += [  
            Meredig(), # slow
            IonProperty(), # slow
            CohesiveEnergy(),  # slow, hangs multithreading
            CohesiveEnergyMP(),  # slow, hangs multithreading
            AtomicPackingEfficiency(),
            ] #ValenceOrbital(),  # already used in AtomicOrbitals()
        feat_dict['composition_oxid'] = [OxidationStates(),
            ElectronegativityDiff(),
            ElectronAffinity(),
            DensityFeatures(),
        ]


    # loop over each feature and add it to the dataframe
    references = []
    for feat_type in feat_dict:
        for f in feat_dict[feat_type]:
            # n_jobs == 1 is required to prevent multithread hanging
            if n_jobs:
                f.set_n_jobs(n_jobs)
            if n_chunksize:
                f.set_chunksize(n_chunksize)

            # implement feature            
            feat = f.featurize_dataframe(
                feat, feat_type, pbar=pbar,
                ignore_errors=True)
            references.append(f.citations())

    # set dataframe index as chemical formula
    feat = feat.set_index(feat[formula_col])
    num_new_features = len(list(feat))
    # remove columns which are constant
    if remove_constant_cols:
        feat = feat.loc[:, (feat != feat.iloc[0]).any()]
    # change all inifinite values to nan
    feat = feat.replace([np.inf, -np.inf], np.nan)
    # remove columns containing a nan
    if remove_nan_cols:
        feat = feat.dropna(axis=1)
    # remove empty references
    references = [r[0] for r in references if r]
    
    print('Kept {} / {} new features for {} materials.'.format(
        len(list(feat)), num_new_features, len(feat)))
    print('Featurization time: {} min'.format(
        round((time.time() - starttime)/60, 2)))
    return feat, references




# for testing purposes
if __name__ == '__main__':

    df = pd.DataFrame({
        'formula': ['C12', 'Ti', '(TiO2)0.4(GaO)0.6'],
        'prop': [1.324, 3.45, 3.56],
    })

    feat, refs = featurize(df)

    print(feat.head())
    print(refs) 