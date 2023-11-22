import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()

from constants import *
from helper import *

def get_SMD(base_mean, comp_mean, base_std, comp_std):
    
    if base_mean == comp_mean:
        return 0
    
    return (comp_mean - base_mean)/np.sqrt((comp_std**2 + base_std**2)/2)


def SMD_between_base_and_comparator(baseline_vars_df, base, comp):
  
    attr_list = remove_element_from_list(list(baseline_vars_df.columns), [PID, 'D_start_date'] + [GRP_COL])

    base_df = baseline_vars_df.loc[baseline_vars_df[GRP_COL] == base, attr_list]
    comp_df = baseline_vars_df.loc[baseline_vars_df[GRP_COL] == comp, attr_list]
    
    stats_dict = {'base_mean': base_df.mean(axis = 0).values,
              'comp_mean': comp_df.mean(axis = 0).values,
              'base_std': base_df.std(axis = 0).values,
              'comp_std': comp_df.std(axis = 0).values}
    
    stats_df = pd.DataFrame(stats_dict, index = attr_list)
    
    stats_df['SMD'] = stats_df.apply(lambda row: get_SMD(row['base_mean'],
                                                             row['comp_mean'],
                                                             row['base_std'],
                                                             row['comp_std']), axis = 1)

    return stats_df[['base_mean', 'comp_mean', 'SMD']]

def cohort_comparison(baseline_vars_df, details = False):

    all_grps = list(baseline_vars_df[GRP_COL].unique())
    pairs = generate_base_comp_pair_list(all_grps, remove_repeated = True)
    
    imbalanced_h = 0.2
    for base, comp in pairs:
        SMD_summary_df = SMD_between_base_and_comparator(baseline_vars_df, base, comp)
        n_imbalanced_rows = SMD_summary_df.loc[SMD_summary_df['SMD'].abs()>=imbalanced_h, :].shape[0]
        print(f'----------base: {base:10s}, comparator: {comp:10s}, #imbalanced_row: {n_imbalanced_rows}----------')
            
    return

def get_stat_series(df):
    
    attr_list = remove_element_from_list(list(df.columns), [PID, 'D_start_date'] + [GRP_COL])
    df_s = df.loc[:, attr_list]

    return df_s.mean(axis = 0)

def get_baseline_characteristics(baseline_vars_df):
    
    bc_df = baseline_vars_df.groupby(GRP_COL).apply(lambda df: get_stat_series(df)).T.round(2)
    
    return bc_df

#-----------------------------------------------------------------------#

def SMD_between_base_and_comparator_stratification(baseline_vars_df, ps_df, base, comp):
    
    joined_df = baseline_vars_df.merge(ps_df[[PID, 'stratum']], how = 'inner', left_on = PID, right_on = PID)
    labels = joined_df['stratum'].unique()
    
    index = None
    cols = None
    res = []
    for label in labels:
        s_df = joined_df.loc[joined_df['statum'] == label, ].drop(['stratum'], axis = 1)
        SMD_df = SMD_between_base_and_comparator(s_df, base, comparator)
        res.append(SMD_df.values.tolist())
        
        if index is None:
            index = SMD_df.index
        if cols is None:
            cols = SMD_df.columns
        
        
    res_np = np.array(res)
    res_mean = np.average(res_np, axis = 0)
    summary_df = pd.DataFrame(res_mean, index = index, columns = cols)
    
    return summary_df

def profile_comparison_stratification(baseline_vars_df, ps_dict):
    
    bc_comp_dict = {}
    for pair, ps_df in ps_dict.items():
        base, comp = pair
        
        before_summary = SMD_between_base_and_comparator(baseline_vars_df, base, comp).add_prefix('bef_')
        after_summary = SMD_between_base_and_comparator_stratification(baseline_vars_df, ps_df, base, comp).add_prefix('aft_')
    
        combined = pd.concat([before_summary, after_summary], axis = 1)
        bc_comp_dict[key] = combined
    
    return bc_comp_dict

def profile_comparison_matching(baseline_vars_df, ps_dict):
    
    bc_comp_dict = {}
    for pair, ps_df in ps_dict.items():
        base, comp = pair
    
        matched_pids = ps_df[PID].unique()
        baseline_var_df_matched = baseline_vars_df.loc[baseline_vars_df[PID].isin(matched_pids), :].reset_index()
        
        before_summary = SMD_between_base_and_comparator(baseline_vars_df, base, comp).add_prefix('bef_')
        after_summary = SMD_between_base_and_comparator(baseline_var_df_matched, base, comp).add_prefix('aft_')
        
        combined = pd.concat([before_summary, after_summary], axis = 1)
        bc_comp_dict[key] = combined
        
    return bc_comp_dict

def weighted_mean_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    
    Reference: 
    
    - Austin PC, Stuart EA. Moving towards best practice when using inverse probability of treatment weighting (IPTW) using the propensity score to estimate causal treatment effects in observational studies. Stat Med. 2015 Dec 10;34(28):3661-79. doi: 10.1002/sim.6607. Epub 2015 Aug 3. PMID: 26238958; PMCID: PMC4626409.
    """
    
    w_mean= np.average(values, axis = 0, weights=weights) #axis = 0: compute a mean for each column

    weights_2d = np.reshape(weights, (1, -1))
    weights_sum = np.sum(weights)
    w_variance = np.dot(weights_2d, (values - w_mean)**2)*(weights_sum)/(weights_sum**2 - np.sum(weights**2))
    w_variance = w_variance.ravel()

    return (w_mean, np.sqrt(w_variance))

def SMD_between_base_and_comparator_IPTW(baseline_vars_df, ps_df, base, comp):
    
    '''
    Using the stabilized weights:
        - SW^A = f(A)/f(A|L)
    '''
    
    joined_df = baseline_vars_df.merge(ps_df, how = 'inner', left_on = [PID, GRP_COL], right_on = [PID, GRP_COL])
    
    #confirmed: 
#     print(f'shape of baseline_vars_df: {baseline_vars_df.shape}')
#     print(f'shape of ps_df: {ps_df.shape}')
#     print(f'shape of joined_df: {joined_df.shape}')
    
    attr_list = remove_element_from_list(list(baseline_vars_df.columns), [PID, 'D_start_date', GRP_COL])
    
    base_data = joined_df.loc[joined_df[GRP_COL] == base, attr_list].values.astype('float')
    comp_data = joined_df.loc[joined_df[GRP_COL] == comp, attr_list].values.astype('float')
    
    base_w = joined_df.loc[joined_df[GRP_COL] == base, 'ps_w'].values
    comp_w = joined_df.loc[joined_df[GRP_COL] == comp, 'ps_w'].values
    
    base_mean, base_std = weighted_mean_and_std(base_data, base_w)
    comp_mean, comp_std = weighted_mean_and_std(comp_data, comp_w)

    df_dict = {'base_mean': base_mean,
               'comp_mean': comp_mean,
               'base_std': base_std,
               'comp_std': comp_std}

    summary_df = pd.DataFrame(df_dict, index = attr_list)
    summary_df['SMD'] = summary_df.apply(lambda row: get_SMD(row['base_mean'],
                                                         row['comp_mean'],
                                                         row['base_std'],
                                                         row['comp_std']), axis = 1)

    return summary_df[['base_mean', 'comp_mean', 'SMD']]

def profile_comparison_IPTW(baseline_vars_df, ps_dict):
    
    bc_comp_dict = {}
    
    for key, ps_df in ps_dict.items():
        base, comp = key
        print(f'base: {base}, comparator:{comp}')
        
        before_summary = SMD_between_base_and_comparator(baseline_vars_df, base, comp).add_prefix('bef_')
        after_summary = SMD_between_base_and_comparator_IPTW(baseline_vars_df, ps_df, base, comp).add_prefix('aft_')
    
        combined = pd.concat([before_summary, after_summary], axis = 1)
        
        bc_comp_dict[key] = combined

    return bc_comp_dict

def SMD_visualization(bc_comp_dict, followup_yrs = None):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(5,5)
#     if followup_yrs is not None:
#         ax.set_title(f'followup_len: {followup_yrs} yrs')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('SMD after weighting')
    ax.set_ylabel('SMD before weighting')
    x_ticks = [float(x)/10 for x in range(0,11)]
    ax.set_xticks(x_ticks)
    
    ax.axvline(x=0.1, color='k', linestyle='-')
    
    for key, summary_df in bc_comp_dict.items():
        base, comparator = key
        before_SMD = summary_df['bef_SMD'].abs().values
        after_SMD = summary_df['aft_SMD'].abs().values
        
        ax.scatter(after_SMD, before_SMD, label = base+' vs ' + comparator)

    ax.legend()
    
    return 