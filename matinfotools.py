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
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import stats
from scipy.interpolate import splrep
from scipy.interpolate import splev
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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
LINEWIDTH=1.5
TICKWIDTH=1.5
plt.rcParams.update({
    'xtick.labelsize': FONTSIZE,
    'ytick.labelsize': FONTSIZE,
    'axes.linewidth': LINEWIDTH,
    'xtick.minor.width': TICKWIDTH,
    'xtick.major.width': TICKWIDTH,
    'ytick.minor.width': TICKWIDTH,
    'ytick.major.width': TICKWIDTH,
    'font.family': 'helvetica',#'Arial',
    'figure.facecolor': 'w',
    #'figure.dpi': dpi,
})



def plot_correlations(corr, annotate=False):
    """Plot correlations between target variable and features,
    using the corrrelation dataframe"""
    xvar, yvar = "sr", "pr"
    # plot all correlations
    plt.scatter(corr[xvar]**2, corr[yvar]**2, c='cornflowerblue', s=60, edgecolor="k", lw=0.5, label="elements",)
    if annotate:
        for i in range(len(corr)):
            plt.annotate(corr.index[i], (corr[xvar].iloc[i]**2, corr[yvar].iloc[i]**2))
    # plot ox endmember features
    corr0 = corr[[i.startswith("EF-ox-endmember") for i in corr.index]]
    plt.scatter(corr0[xvar]**2, corr0[yvar]**2, c='limegreen', s=60, edgecolor="k", marker="s", lw=0.5, label="oxides",)
    # plot engineered features
    corr0 = corr[[i.startswith("EF--") and t in i for i in corr.index]]
    plt.scatter(corr0[xvar]**2, corr0[yvar]**2, c='tomato', s=60, edgecolor="k", marker="D", lw=0.5, label="engineered",)
    plt.xticks([0, 0.5, 1], [0, 0.5, 1])
    plt.ylabel("r$^2$", fontsize=24,)
    plt.yticks([0, 0.5, 1], [0, 0.5, 1])
    if t == 'Tg [K]':
        plt.legend(fontsize=14)
    plt.title(t, fontsize=24)
    mit.plot_setup(xlabel="r$_s^2$", fsize=24, limits=(0, 1, 0, 1), size=(4,4),)
    plt.show()



def plot_best_feature(df, corr, t, linear_fit=True, colorbar=False):
    """Plot the most linear feature with tthe target variable"""
    xvar, yvar = corr.index[0], t
    colors = [len(c) for c in df["comp"]]
    plt.scatter(
        df[xvar], df[yvar],
        #c='cornflowerblue',
        c=colors,
        cmap="plasma",
        vmin=1,
        vmax=12,
        s=40, edgecolor="w", lw=0.2,)
    if linear_fit:
        xmin, xmax = np.min(df[xvar]), np.max(df[xvar])
        # perform linear fit
        r2 = corr["r2"].iloc[0]
        print(f"r^2: {round(r2, 2)}")
        slope, intercept, _, _, lin_err = stats.linregress(df[xvar], df[yvar])
        xfit = np.linspace(xmin, xmax, 15)
        yfit = xfit * slope + intercept
        plt.plot(xfit, yfit, lw=1, c="k")
        plt.annotate("r$^2$: {}".format(round(r2, 2)), (0.65, 0.25), xycoords='figure fraction', fontsize=24)
    if colorbar:
        cbar = plt.colorbar()
        cbar.set_ticks([1, 4, 8, 12])
    mit.plot_setup(xlabel=xvar.replace("EF-ox-endmember", "EF-ox"), ylabel=t, fsize=24, size=(4,4),)    
    plt.show()

    

def remove_target_outliers(df, target, percentile=1):
    """Remove the outliers of a target variable"""
    # drrop nans
    df = df[df[t].notna()].dropna(axis=1)
    # sort by target value
    a = df[df[target].notna()][target].sort_values()
    # number of samples to clip
    clip_n_samples = int(percentile*len(a)/100)
    # keep indices inside the clip region
    keep_idx = a.index[clip_n_samples:-clip_n_samples]
    return df.loc[keep_idx]


def train_model(df, input_cols, target_col, splits, model_type, plot_pva=True, verbose=False):
    """
    Train an ML model using a dtafrme, target column,
    and dict of train-test splits.
    
    Inputs:
    df: dataframe containing the data
    input_cols: column names which should be useed as model input feautures
    target_col: column name to use as model output
    model_type: type of model. Choose one of ["RF", "linear"]
    splits: a dict of train-test splits to use for model cross-validation.
    splits should have the form
        splits {
            "split1": "train": train_idx, "test": test_idx},
            "split2": "train": train_idx, "test": test_idx},
        }
    where the keys in splits are the split names ( they can be anything) and 
    the train_idx and test_idx are indices of the dtaaframe to use
    for model training and testing.
    plot_pva: boolean whether to show PVA plot after model training
    verbose: boolean whether to show results of each train-test split
    
    Output:
    results: dataframe summarizing the results of model testing
    model: trained model
    """
    results = {k: [] for k in [
        'sample',
        'actual',
        'predicted',
        'std',
        'error',
        'system',
        'split',
        'source',
        'importances',
        'inputs',
        'targets',
        'class',
    ]}
    
    print(f"target: {target_col}, inputs: {len(input_cols)}, train-test splits: {len(splits)}, samples: {len(df)}")
    
    #model = LinearRegression()
    #model = ElasticNet()
    #model = RandomForestRegressorCov()
    #model = RFRlolo()
    #model = SVR()
    
    
    model_dict = {
        "RF": RandomForestRegressorCov(),
        "linear": LinearRegression(),
    }

    model = model_dict[model_type]
    
    starttime = time()
    # loop over each holdout set for cross-validation
    for si, s in enumerate(splits):
        # get inidices of training and testing rows
        test_idx = splits[s]["test"]
        train_idx = splits[s]["train"]
        # get training and testing inputs and outputs
        train_in = df.loc[train_idx][input_cols].values
        train_out = df.loc[train_idx][target_col].values
        test_in = df.loc[test_idx][input_cols].values
        test_out = df.loc[test_idx][target_col].values

        # fit model and make predictions on test samples
        model.fit(train_in, train_out)
        

        if model_type == "RF":
            # for custom random forest covariance models
            importances = list(model.feature_importances_)
            pred, std, _ = model.predict(test_in)
            
        elif model_type == "lolo":
            # for lolopy models
            importances = list(model.feature_importances_)
            pred, std = model.predict(test_in, return_std=True)
                        
        elif model_type == "linear":
            # for linear models
            importances = model.coef_
            pred, std = model.predict(test_in), [0]*len(test_in)
            
        else:
            pred, std = model.predict(test_in), [0]*len(test_in)
            importances = [0]*len(input_vars)

        # save results
        results['split'] += [s]*len(test_out)
        results['sample'] += list(test_idx)
        results['actual'] += list(test_out)
        results['predicted'] += list(pred)
        results['std'] += list(std)
        results['error'] += list(np.subtract(test_out, pred))
        results['system'] += list(np.full(len(test_out), s))
        results['source'] += list(df.loc[test_idx]['source'])
        results['class'] += list(df.loc[test_idx]['class'])
        results['importances'] += [importances for _ in range(len(test_out))]
        results['inputs'] += [input_cols for _ in range(len(test_out))]
        results['targets'] += [target_col for _ in range(len(test_out))]

        if verbose:
            ndme0 = round(mit.get_rmse(np.subtract(test_out, pred), rounding=6) / np.std(test_out), 3)
            print('{:2}/{:2}, {:10} NDME: {:5}, {:5} min'.format(
                si+1, len(splits), s, ndme0, round((time()-starttime)/60, 2)))
        
        if 0:  # to plot PVA for each holdout set
            plt.scatter(test_out, pred, s=3)
            plt.plot([np.min(test_out), np.max(test_out)], [np.min(test_out), np.max(test_out)], c="k")
            mit.plot_setup(size=(4,2), xlabel="Actual", ylabel="Predicted", title=s)
            plt.show()

    results = pd.DataFrame(results)
    results = results.set_index('sample')
    rmse = np.sqrt(np.mean(np.square(results['error'])))
    ndme = rmse / np.std(results['actual'])
    print('NDME: {}'.format(round(ndme, 3)))
    
    # if using a linear model, use bootstraping to estimate prediction uncertainties
    if model_type == "linear":

        # loop over each sample
        for ss in results.index.unique():
            # if the same sample was predicted in multiple train-test splits
            if len(results.loc[ss]) > 1:
                # use st dev in predictions for the prediction uncertainty
                results.loc[ss, "std"] = np.std(results.loc[ss, "predicted"])


    if plot_pva:
        plotlims = [results["actual"].min(), results["actual"].max()]
        plt.hist2d(results['actual'], results['predicted'], bins=100, cmap="gnuplot2_r", vmin=0, vmax=15)
        #plt.scatter(results['actual'], results['predicted'], s=3,)

        plt.plot(plotlims, plotlims, c="k", lw=1)
        mit.plot_setup(
            xlabel="Actual", ylabel="Predicted",
            title=target_col, size=(4,4), limits=[plotlims[0]*0.95, plotlims[1]*1.05]*2)
        plt.show()
    
    
    # finally, train model on entire dataset to export the fully-trained model
    model.fit(df[input_cols].values, df[target_col].values)
    
    return results, model


def bootstrap(arr, n_samples, n_trials):
    """
    Sample an array multiple times with dynamic
    probabilities to ensure that the array is sampled
    evenly over all the trials.
    
    Inputs
        arr: (array-like) sequence of elements to sample
        n_samples: (int) number of elements to sample per trial
        n_trials: (int) number of trials
    Returns
        list of lists, where each list is one trial of samples
    """
    if n_samples > len(arr):
        raise ValueError("'n_samples' must not be larger than the length of 'array_to_sample'")
    # make sure input array is a numpy array so it
    # can be indexed by another array
    arr = np.array(arr)
    # get indices of input array
    all_indices = range(len(arr))
    # initial probabilities are uniform
    prob = np.full(len(arr), 1) / len(arr)
    # the initial uniform probability for each sample
    prob0 = prob[0]
    # loop over each trial to get a new sample set
    samples = []
    for t in range(n_trials):
        # select some indices randomly by using probabilities
        s = np.random.choice(all_indices, size=n_samples, replace=False, p=prob)
        samples.append(list(arr[s]))
        # update probabilities so previously-chosen samples
        # are less likely to be chosen during the next round.
        # decrease probabilities of selected samples
        prob[s] -= prob0
        # increase probabilities of non-selected samples
        prob[~s] += prob0
    return samples


def get_bootstrap_splits(arr, n_samples, n_trials):
    """
    Sample an array multiple times with dynamic
    probabilities to ensure that the array is sampled
    evenly over all the trials.
    
    Inputs
        arr: (array-like) sequence of elements to sample
        n_samples: (int) number of elements to sample per trial
        n_trials: (int) number of trials
    Returns
        dict of train and test indices
    """
    if n_samples > len(arr):
        raise ValueError("'n_samples' must not be larger than the length of 'array_to_sample'")
    # make sure input array is a numpy array so it
    # can be indexed by another array
    arr = np.array(arr)
    # get indices of input array
    all_indices = range(len(arr))
    # initial probabilities are uniform
    prob = np.full(len(arr), 1) / len(arr)
    # the initial uniform probability for each sample
    prob0 = prob[0]
    # loop over each trial to get a new sample set
    splits = {}
    for t in range(n_trials):
        # select some indices randomly by using probabilities
        s = np.random.choice(all_indices, size=n_samples, replace=True)
        splits[t] = {"train": arr[s], "test": arr[[ss for ss in all_indices if ss not in s]]}
    return splits


def get_new_feats(corr, N_COMBO_FEATS=10, strategy="best"):
    """Get new features using combinations of existting features"""
    rand_feats = corr.index[np.random.randint(0, high=len(corr), size=N_COMBO_FEATS)]
    best_feats = [f for f in list(corr.index)[:N_COMBO_FEATS] if f not in TARGETS]
    half_feats = np.unique([rand_feats[:int(N_COMBO_FEATS/2)]] + [best_feats[:int(N_COMBO_FEATS/2)]])
    if strategy == "random":
        feats_to_permute = rand_feats
    elif strategy == "best":
        feats_to_permute = best_feats
    elif strategy == "half":
        feats_to_permute = half_feats
    else:
        raise Exception(f'Invalid stragegy. Try best, random, or half.')
    print('{} features to permute'.format(2*len(feats_to_permute)))
    mit.create_polynomial_features(
        df0,
        cols=feats_to_permute,
        target=t,
        order=4,
    )


def gaussian(x, amplitude):
    return amplitude * np.exp(-(x**2) / 2 )

class RandomForestRegressorCov(RandomForestRegressor):
    """
    Custom wrapper for sklearn RandomForest regressor
    which returns random forest predictions
    along with std and predictions by all estimators.
    """
    def predict(self, X):
        preds = RandomForestRegressor.predict(self, X)
        est_preds = np.empty((len(X), len(self.estimators_)))
        # loop over each tree estimator in forest and use it to predict
        for ind, est in enumerate(self.estimators_):
            est_preds[:,ind] = est.predict(X)
        if not np.allclose(np.mean(est_preds, axis=1), preds):
            print('Mean over estimators not equal to predictions')
        return preds, np.std(est_preds, axis=1), list(est_preds)


def get_correlations_w_target(df, tar_var, input_vars):
    """
    Get correlations between a target variable and other
    variables in a dataframe.
    Inputs:
    df: dataframe
    tar_var: column name of the target variable
    input_vars: column names with which to find correlations with target
    """
    input_vars =[i for i in input_vars if i != tar_var]
    # create dataframe with basic stats
    corr = pd.DataFrame({
            k: {
                'pr': stats.pearsonr(df[k], df[tar_var])[0],
                'sr': stats.spearmanr(df[k], df[tar_var])[0],
                'variance': np.var(df[k]),
                'std': np.std(df[k]),
                'n_unique': len(np.unique(df[k])),
                'unique_frac': len(np.unique(df[k])) / len(df)
            } for k in input_vars
        }).T
    # get total correlation column
    corr['r_tot'] = np.sqrt(np.square(corr['pr']) + np.square(corr['sr']))
    corr['r2'] = np.square(corr['pr'])
    # sort values
    corr = corr.sort_values('r2', ascending=False)
    return corr


def get_input_correlations(df, input_vars=None):
    """
    Get r^2 correlations between different input features.
    Inputs:
    df: dataframe with features
    input_vars: list of column names to use as input features
    """
    if input_vars is None:
        input_vars = [c for c in list(df) if pd.api.types.is_numeric_dtype(df[c])]
    # get all pairs of input feature combinations
    pairs = itertools.combinations(input_vars, 2)
    # loop over each pair of features and get r-squared correlation between them
    r2s = []
    for p in pairs:
        r2 = np.square(stats.pearsonr(df[p[0]], df[p[1]])[0])
        r2s.append([p[0], p[1], round(r2, 4)])
    r2df = pd.DataFrame(r2s, columns=['1', '2', 'r2']).sort_values(
        'r2', ascending=False, ignore_index=True)
    return r2df

def drop_redundant_cols(df):
    """
    Drop constant columns and duplicated
    columns from a Pandas datafame.
    """
    # drop columns in which all values are constant
    df0 = df[[c for c in df if not pd.api.types.is_numeric_dtype(df[c])]].copy()
    df = df[[c for c in df if pd.api.types.is_numeric_dtype(df[c])]]
    df = df.loc[:, (df != df.iloc[0]).any()]
    # drop duplicate columns, keeping only first column
    df = df.T.drop_duplicates().T
    df = pd.concat([df0, df], axis=1)
    return df


def write_json(data, filepath):
    """Write JSON data to a filepath"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(filepath):
    """Read a json file from its filepath"""
    with open(filepath) as f:
        data = json.load(f)
    return data



def get_spline(old_x, old_y, new_x, k=2, s=0):
    """
    Get a spline fit from an x and y array on a new x array.
    Inputs:
    old_x: original x data
    old_y: original y data
    new_x: new x data on which to evaluate spline
    k: degree of spine fit
    s: smoothing factor for spline
    Returns:
    spline fit evaluated on new x values
    """
    # if the arrays are not sorted, sort them
    #if np.all(np.diff(a) >= 0):
        
    # get spline parameters
    spline_params = splrep(old_x, old_y, k=int(k), s=s)
    # evaluate spline at new x values
    return splev(new_x, spline_params)




def drop_redundant_cols(df):
    """
    Drop constant columns and duplicated
    columns from a Pandas datafame.
    """
    # get non-numeric columns
    df0 = df[[c for c in df if not pd.api.types.is_numeric_dtype(df[c])]].copy()
    # get numeric columns
    df = df[[c for c in df if pd.api.types.is_numeric_dtype(df[c])]]
    # drop numeric columns in which all values are constant
    df = df.loc[:, (df != df.iloc[0]).any()]
    # drop duplicate numeric columns
    df = df.T.drop_duplicates().T
    # re-combine non-numeric and numeric columns
    df = pd.concat([df0, df], axis=1)
    return df

def create_polynomial_features(df, cols, target, order=3):
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
    df = drop_redundant_cols(df)
    cols = [c for c in cols if c in df]
    cols = [c for c in df if pd.api.types.is_numeric_dtype(df[c])]
    
    # get highest existing r^2 value to try to beat
    r2_goal = np.max([np.square(
                stats.pearsonr(df[c], df[target])[0]
            ) for c in list(df) if c != target and pd.api.types.is_numeric_dtype(df[c])])
    # instantiate polynomial feature model
    polyfeats = PolynomialFeatures(order, include_bias=False)
    
    # add inverse of columns to allow for division
    for c in cols:
        if 0 not in df[c].values:
            df["_inverse "+c] = 1/df[c].values
        
    # inputs for poly feats
    cols += [c for c in list(df) if "_inverse" in c]
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
            # only keep better than existing features
            if r2 > r2_goal:
                polyfeat_dict[n] = r2
    # sort polyfeature dict by correlations
    polyfeat_dict = {
        k: v for k, v in sorted(
            polyfeat_dict.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    print('Total runtime: {} min'.format(round((time.time() - starttime)/60, 2)))
    print('Previous highest r^2 value: {}'.format(round(r2_goal, 2)))
    print('{} new higher r^2 values:'.format(len(polyfeat_dict)))
    for k in list(polyfeat_dict)[:20]:
        print('{:6}   {}'.format(round(polyfeat_dict[k], 4), k))
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
    plt.rcParams.update({
        'xtick.labelsize': fsize,
        'ytick.labelsize': fsize,
    })
    
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



def bar_plot(x, y, n=50, alpha=1, label=None, rotation=90):
    """
    Create bar plot instance. To change color of bars:
    for b in bars:
        b.set_color('r')
    Use 'n' argument to limit the number of bars shown.
    """
    x, y = x[:n], y[:n]
    ha = 'center' if (rotation == 90) else 'right'
    bars = plt.bar(np.arange(len(x)), y, width=0.5, alpha=alpha, label=label)
    plt.gca().set_xticks(np.arange(len(x)))
    plt.gca().set_xticklabels(list(x), rotation=rotation, ha=ha)
    plt.gca().set_xlim([-1, len(x)])
    return bars


def bar_plot_h(x, y, n=30, alpha=1, label=None):
    """
    Create horizonatal bar plot instance. To change color of bars:
    for b in bars:
        b.set_color('r')
    Use 'n' argument to limit the number of bars shown.
    """
    x, y = x[:n], y[:n]
    bars = plt.barh(np.arange(len(x)), y[::-1], height=0.5, alpha=alpha, label=label)
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
        
     
    
def plot_periodic_table(lw=1, c='gray', alpha=1):
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
    # remove plot borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    
        
        
        

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