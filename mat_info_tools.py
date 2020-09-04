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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    #'figure.dpi': dpi,
})

    
def plot_setup(
    xlabel=False,
    ylabel=False,
    title=False,         
    fsize=16, 
    legend_fontsize=12,
    limits=False,
    size=False,
    legend=True,
    save=False,
    filename='plot.jpg',):
    """Creates a custom plot configuration to make graphs look nice.
    This can be called with matplotlib for setting axes labels,
    titles, axes ranges, and the font size of plot labels.
    This should be called between plt.plot() and plt.show() commands."""
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
        fig.savefig(filename, dpi=250, bbox_inches='tight')
        plt.tight_layout()


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
  



def get_rmse(errors, round=3):
    """Get the RMSE fo a sequence of errors"""
    errors = np.array(errors).astype(np.float)
    return np.round(
    np.sqrt(np.mean(np.square(errors))),
    decimals=3)

    


def featurize(
    df,
    formula_col='formula',
    pbar=True,
    remove_nan_cols=True,
    remove_constant_cols=True,
    remove_nonnumeric_cols=True,
    return_references=False,):
    """
    Featurization of cheical formulas for machine learning.
    Input a Pandas Dataframe with a column called formula_col,
    which contains chemical formulas (e.g. ['Mg', 'TiO2']).
    Other columns may contain additional descriptors.
    The chemical formulas are featurized according to methods
    from the matminer package. Returns the dataframe with chemical
    formulas and features, and a list of references
    to papers which describe the featurization methods used.

    To prevent issues with multithreading, set n_jobs=1.

    To ignore certain features, comment them out in the
    'composition_features' or 'composition_ox_features' lists.

    Use the kwargs to return the list of references used,
    remove dataframe columns which are constant,
    remove dataframe columns which contain nans,
    or remove dataframe columns which are non-numeric.

    ================= Useful links =======================
    Matminer summary table of features:
    https://hackingmaterials.lbl.gov/matminer/featurizer_summary
    Matminer Github repo:
    https://github.com/hackingmaterials/matminer
    Matminer notebook examples:
    https://github.com/hackingmaterials/matminer_examples
    """
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
        pbar=pbar)
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

    # features related to composition and oxidation
    feat_dict = {
        'composition': [
            element_property,
            Meredig(),
            BandCenter(),
            IonProperty(),
            Stoichiometry(),
            AtomicOrbitals(),
            TMetalFraction(),
            CohesiveEnergy(),  # slow, hangs multithreading
            ElementFraction(),
            CohesiveEnergyMP(),  # slow, hangs multithreading
            Miedema(),
            YangSolidSolution(),  # slow, hangs multithreading
            AtomicPackingEfficiency(),
            #ValenceOrbital(),  # already used in AtomicOrbitals()
        ],
        'composition_oxid': [
            OxidationStates(),
            ElectronegativityDiff(),
            ElectronAffinity(),
            DensityFeatures(),  
        ]}

    # loop over each feature and add it to the dataframe
    references = []
    for feat_type in feat_dict:
        for f in feat_dict[feat_type]:
            # this required to prevent multithread hanging
            f.set_n_jobs(1)
            #f.set_chunksize(30)

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
    # remove non-numeric columns
    if remove_nonnumeric_cols:
        feat = feat.select_dtypes(include=[np.number])
    # remove empty references
    references = [r[0] for r in references if r]
    print('Kept {} / {} new features for {} materials.'.format(
        len(list(feat)), num_new_features, len(feat)))    
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