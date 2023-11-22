import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()

from constants import *
from helper import *
from bc_comparator import *

from sklearn.linear_model import LogisticRegression
#import xgboost as xgb
import seaborn as sns

def ps_distribution_visualization(ps_df, base, comp):
    
    bins = 40
    bin_cut = np.array(range(0,bins+1))/float(bins)

    fig, ax = plt.subplots()
    base_ps = ps_df.loc[ps_df[GRP_COL] == base, :]
    comp_ps = ps_df.loc[ps_df[GRP_COL] == comp, :]

    sns.histplot(base_ps, x="ps", bins = bin_cut, stat = "density", ax = ax, color = 'tab:blue', alpha = 0.5)
    sns.histplot(comp_ps, x="ps", bins = bin_cut, stat = "density", ax = ax, color = 'tab:orange', alpha = 0.5)
    ax.set_title(f'base: {base}, comparator: {comp}')
    ax.set_xlabel( f'propensity score\n({base}  <---->  {comp})')
    ax.legend([base, comp])

    return 

def estimate_ps_per_pair(baseline_var_df, 
                         base, 
                         comp, 
                         weights_details = False, 
                         visualization = False):

    s_df = baseline_var_df.loc[baseline_var_df[GRP_COL].isin([base, comp])].reset_index(drop = True)
    
    X_cols = remove_element_from_list(list(s_df.columns), [PID, 'D_start_date'] + [GRP_COL] + ['gender_Female', 'race_Chinese'])

    X = s_df[X_cols].values
    y = (s_df[GRP_COL] == comp).values #comp:1, base: 0
    
    class_weight = 'balanced' #'balanced' or None
    lr = LogisticRegression(random_state = 0, C = 0.5, max_iter=5000, class_weight = class_weight).fit(X, y)
    score = lr.score(X, y)
    probas = lr.predict_proba(X)
    print(f'---------------------------base:{base}, comparator:{comp}----------------------------')
    print(f'lr accuracy: {score*100:.2f}')

    ps_df = s_df[[PID, GRP_COL]]
    ps_df.loc[:, 'ps'] = probas[:,1]
    
    if weights_details:
        coefs = lr.coef_[0,:]
        weights = pd.Series(coefs, index = X_cols)
        
        weights_base = weights[weights<0].sort_values(ascending = False, key = lambda v: abs(v))
        weights_comp = weights[weights>0].sort_values(ascending = False, key = lambda v: abs(v))
        print('----------Contributors to base:----------\n', weights_base.head(5))
        print('----------Contributors to comparator:----------\n', weights_comp.head(5))
        print('\n')
        
    if visualization:
        ps_distribution_visualization(ps_df, base, comp)

    return ps_df

def get_percentile_range(arr, min_pctl = 2, max_pctl = 98):
    
    min_v = np.percentile(arr, min_pctl)
    max_v = np.percentile(arr, max_pctl)
    
    return (min_v, max_v)

def add_IPTW(ps_df, base, comp):
    
    '''
    Using the stabilized weights:
        - SW^A = f(A)/f(A|L)
        
    Debug: 
    - The weights assignment to df is corrent. Confirmed.
    '''
    
    b_ps_v = ps_df.loc[ps_df[GRP_COL] == base,'ps'].values
    c_ps_v = ps_df.loc[ps_df[GRP_COL] == comp,'ps'].values
    
    n, base_n, comp_n = len(ps_df), len(b_ps_v), len(c_ps_v)
    base_prob, comp_prob = float(base_n)/n, float(comp_n)/n
    
    ps_w_df = ps_df.copy()
    ps_w_df.loc[ps_w_df[GRP_COL] == base, 'ps_w'] =  base_prob/(1-b_ps_v)
    ps_w_df.loc[ps_w_df[GRP_COL] == comp, 'ps_w'] = comp_prob/c_ps_v
    
    return ps_w_df #(PID, GRP_COL, 'ps', 'ps_w'), GRP_COL in {base, comp}

def logit(p):
    
    if p < 0 or p > 1:
        raise ValueError('Incorrect input for logit()!')
    else:
        return np.log(p/(1-p))

def get_matched_list(ptn, str_list):
    
    import re
    
    matched_list = []
    
    for s  in str_list:
        if re.match(ptn,s):
            matched_list.append(s)
    
    return matched_list

def one_to_one_matching(base, comp, base_list, comp_list, caliper = 0.1):
     
    matched_list = [] #comp_id, base_id, comp_score, base_score
    
    np.random.seed(seed=666)
    np.random.shuffle(comp_list)
    comp_std = np.std(comp_list[:,1])
    s = caliper*comp_std

    for c_id, c_score in comp_list:
        
        idx = np.argmin(np.abs(base_list[:, 1] - c_score))#find the nearest
        b_id, b_score = base_list[idx, :]
        
        if abs(c_score - b_score) <= s: #score within the threshold 's'
            matched_list.append([c_id, b_id, c_score, b_score])
            base_list = np.delete(base_list, idx, 0)
            if len(base_list) == 0:
                break
        else:
            pass
        
    matched_df = pd.DataFrame(matched_list, columns = ['comp_id', 'base_id', 'comp_score', 'base_score'])
    pids = matched_df['comp_id'].values.tolist() + matched_df['base_id'].values.tolist()

    return pids, matched_df

def ps_one_to_one_matching(ps_df, base, comp, details = True):

    df_copy = ps_df.copy()
    df_copy['logit_ps'] = df_copy['propensity'].apply(lambda v: logit(v))
    base_list = df_copy.loc[df_copy[GRP_COL] == base, [PID, 'logit_ps']].values
    comp_list = df_copy.loc[df_copy[GRP_COL] == comp, [PID, 'logit_ps']].values
    
    pids, matched_df = one_to_onematching(base, comp, base_list, comp_list)
    
    matched_ps_df = ps_df.loc[ps_df[PID].isin(pids), :].reset_index()
    
    if details:
        print(f'Base: {base:20s}, comp: {comp:20s}')
        print(f'Before matching: #base: {str(len(base_list)):10s}, #comp: {str(len(comp_list)):10s}')
        print(f'After matching: {str(len(matched_df)):10s} patients matched!')
    
    return matched_ps_df

def add_stratum_label(ps_df, strata_num = 5):
    
    if ps_df.empty:
        return ps_df
    
    ps_stratum_df = ps_df.copy()
    ps_stratum_df.loc[:, 'stratum'] = pd.qcut(ps_stratum_df['ps'].values, strata_num, labels = False)
    
    return ps_stratum_df

def check_positivity(ps_df, base, comp, visualization = False):
    
    '''
    Check positivity: removing the sample whose ps not in 2-98 percentile.
    '''

    if ps_df.empty:
        return ps_df

    b_ps_v = ps_df.loc[ps_df[GRP_COL] == base, 'ps'].values
    c_ps_v = ps_df.loc[ps_df[GRP_COL] == comp, 'ps'].values

    b_min, b_max = get_percentile_range(b_ps_v)
    c_min, c_max = get_percentile_range(c_ps_v)
    range_min, range_max = max(b_min,c_min), min(b_max, c_max)

    mask = (ps_df['ps'] >= range_min) & (ps_df['ps'] <= range_max)
    ps_positivity_df = ps_df.loc[mask, :].reset_index(drop = True)

    if visualization:
        ps_distribution_visualization(ps_positivity_df, base, comp)

    return ps_positivity_df

def estimate_propensity_score(baseline_var_df, 
                              base_list = [], 
                              comp_list = [],
                              option = 'IPTW'):
    
    '''
    Aim: estimate propensity score for each grp-pair, and conduction the positiveity checking.
    
    Return: ps_dict. key = (base, comp), value = ps_XXX_df, based on the options specified.
    '''
    
    visualization = False
    
    all_grps = baseline_var_df[GRP_COL].unique()
    pair_list = generate_base_comp_pair_list(all_grps, base_list, comp_list, remove_repeated = True)
    
    ps_dict = {}
    for base, comp in pair_list:
        #estimation
        ps_df = estimate_ps_per_pair(baseline_var_df, 
                                     base, 
                                     comp, 
                                     visualization = visualization) #schema = (PID, GRP_COL, 'ps')
        
        #positivity checking
        ps_pos_df = check_positivity(ps_df, base, comp, visualization = False)
        
        #balancing based on option
        if option == 'IPTW':
            #schema = (PID, GRP_COL, 'ps', 'ps_w')
            ps_dict[(base, comp)] = add_IPTW(ps_pos_df, base, comp) 
        elif option == 'matching':
            #schema = (PID, GRP_COL, 'ps'), only the matched patients are preserved
            ps_dict[(base, comp)] = ps_one_to_one_matching(ps_pos_df, base, comp) 
        elif option == 'stratification':
            #schema = (PID, GRP_COL, 'ps', 'stratum')
            ps_dict[(base, comp)] = add_stratum_label(ps_df, strata_num = 10)
        else:
            raise ValueError('Incorrect option for ps balancing.')
        
    return ps_dict
