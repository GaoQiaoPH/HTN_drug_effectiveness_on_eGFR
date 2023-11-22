import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()

from constants import *
from helper import *
from ps_estimator import *
from bc_comparator import *

def get_interpolate_list(x1,x2,y1,y2):
    '''
    return the interpolated value list in the middle, without y1 and y2.
    '''
    
    delta = (y2-y1)/(x2-x1)
    v_list = []
    
    for i in range(1,x2-x1):
        v = y1+delta*i
        v_list.append(v)
        
    return v_list

def get_follow_up_list(index_date,
                       end_date,
                       values_np,
                       with_control,
                       unit_day_len,
                       baseline_avg):
    
    '''
    values_np = (date, value)
    
    Get follow-ups in [index_date, end_date]

    Special cases (with control == False): 
    - no valid index_date or end_date: return []
    - baseline value not found: return []
    - follow-up values not found: return []
    - otherwise: return [baseline, followup_1, follow_up2, ...]
    
    Special cases (with control == True): 
    - no valid index_date or end_date: return []
    - baseline value not found: return []
    - control not found: return []
    - follow-up values not found: return []
    - otherwise: return [control, baseline, followup_1, follow_up2, ...]

    '''
    
    #date type conversion
    if pd.isnull(index_date) or pd.isnull(end_date):
        return []
    else:
        index_date = np.datetime64(index_date)
        end_date = np.datetime64(end_date)
        values_np = np.array(values_np)

    #get baseline value: wind = (baseline_start, index_date]
    baseline_start = index_date - np.timedelta64(unit_day_len,'D')
    mask_baseline = (values_np[:, 0] > baseline_start) & (values_np[:, 0] <= index_date)
    pre_data = values_np[mask_baseline, :]
    if len(pre_data) == 0:#baseline not found
        return []
    if baseline_avg:
        baseline_v = np.average(pre_data[:, 1])
    else:
        baseline_v = pre_data[-1, 1]
    
    if with_control:
        #get control value: (control_start, control_end]
        control_start = index_date - np.timedelta64(unit_day_len*2,'D')
        control_end = index_date - np.timedelta64(unit_day_len,'D')
        mask_control = (values_np[:, 0] > control_start) & (values_np[:, 0] <= control_end)
        control_data = values_np[mask_control, :]
        
        if len(control_data) == 0:# control not found
            return []
        else:
            control = [np.average(control_data[:, 1])]
    else:
        control = []

    #find values after index date
    mask_study = (values_np[:, 0] > index_date) & (values_np[:, 0] <= end_date)
    post_data = values_np[mask_study, :]
#     print(post_values)
#     print(type(post_values))

    dates = pd.Series(post_data[:, 0]).values
    values = post_data[:, 1]
    n = len(dates)
    
    if len(dates) == 0: #no follow-ups
        return []

#     print(dates)
#     print(type(dates))
    
    #x_pre: unit_idx of each date
    x_pre = np.ceil((dates - index_date)/np.timedelta64(unit_day_len,'D')).astype('int')
    y_pre = values

    #assign value in values to each unit.
    #i,j used to find values with same unit_idx
    #x:unit_idx
    #y:avg_v
    i = 0
    j = i
    x = [0] 
    y = [baseline_v]
    while (i<len(x_pre)) and (j<len(x_pre)):
        while (j<len(x_pre)) and (x_pre[i] == x_pre[j]):
            j+=1
        avg = np.mean(y_pre[i:j])
        x.append(x_pre[i])#number of time window
        y.append(avg)
        i = j
        
    #interpolation: for each (x_k,y_k) and (x_{k+1}, y_{k+1}), interpolate the middle values.
    inter_list = []
    for k in range(0,len(y)-1):
        inter_list += [y[k]]
        sub_list = get_interpolate_list(x[k],x[k+1],y[k],y[k+1])
        inter_list += sub_list
    #add the last element
    inter_list += [y[-1]]

    return control + inter_list

def adding_followup_values(pid_date_df,
                           date_attr, 
                           end_date_attr,
                           value_df,
                           value_attr,
                           with_control = False,
                           unit_day_len = 365,
                           baseline_avg = False):
    
    '''
    Input: 
        pid_date_df: with columns (PID, index_date, baseline_value, ...)
        date_attr: column name of 'index_date' in pid_date_df
        value_df: with columns (PID, DATE, value)
        value_attr: column name of 'value' in value_df
        unit_day_len: unit len for follow-up values in days to extract a value after index date.
        
    Return:
        The same 'pid_date_df' as input, with an additional column 'follow-ups', containing a list of follow-up values
            of the same patient in time window [index_date, end_date]. One value is extracted from 'value_df' for every 'unit_day_len'
        Note that 'baseline_value' is the first element in this list.
        
    Processing: 
        From index_date (as day_0), find a follow-up value from 'value_df' for every unit (week-7d, month-30d, year-365d).
        If multiple values found for one unit, take average.
        If no value found for one unit, use linear interpolation to generate a value from adjancet units.
        
    '''
    
    #group multiple values of the same patient into a row.
    pid_date_df = pid_date_df.reset_index(drop = True)
    value_df = value_df.loc[value_df[PID].isin(pid_date_df[PID]),
                            [PID, DATE, value_attr]]
    
    trans_value_df = value_df.groupby(PID)\
                            .progress_apply(lambda df: df[[DATE, value_attr]].values)\
                            .reset_index().rename(columns = {0:'values_np'})

    joined_df = pid_date_df.merge(trans_value_df, how = 'inner', left_on = PID, right_on = PID)
    joined_df = joined_df[[PID, date_attr, end_date_attr, 'values_np']]
    
    #interpolation
    joined_df.loc[:, 'follow_ups'] = joined_df.progress_apply(lambda row: get_follow_up_list(row[date_attr],
                                                                           row[end_date_attr],
                                                                           row['values_np'], 
                                                                           with_control,
                                                                           unit_day_len,
                                                                           baseline_avg = baseline_avg), 
                                                                     axis = 1)
    
    return joined_df[[PID, date_attr, 'follow_ups']]

def get_CI95(values):
    
    from scipy.stats import sem, t
    import math
    
    confidence = 0.95
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)
    
    h = std * t.ppf((1 + confidence) / 2, n - 1)/math.sqrt(n)
    CI_start = round(mean-h,1)
    CI_end = round(mean+h,1)
    CI_tuple = (CI_start, CI_end)
    
    return CI_tuple, h

def get_diff_to_baseline(follow_ups, baseline_idx):
    
    follow_ups_diff = []
    
    for sub in follow_ups:
        if len(sub) >= 1:
            sub_diff = [i - sub[baseline_idx] for i in sub]
            follow_ups_diff.append(sub_diff)

    return follow_ups_diff

def get_stats_from_followups(followups, weights = []):

    '''
    followups.shape = (#patients, #follow-ups). All the follow-ups should have the same length.
    baseline_idx: which element in follow-ups indicate the baseline value.
    output_option: in ['raw', 'diff_to_baseline']
    '''
    
    from statsmodels.stats.weightstats import DescrStatsW

    use_diff = True
    if use_diff:
        followups = get_diff_to_baseline(followups, BASELINE_IDX)
    else:
        pass

    #get statistics
    followups = np.array(followups)
    n = followups.shape[0]
    
    if len(weights) == 0:
        weights = [1]*n
    
    d1 = DescrStatsW(followups, weights=weights)
    means = d1.mean
    stds = d1.std
    CI_lower, CI_upper = d1.tconfint_mean(alpha=0.05)
        
    stats_dict = {'sample_size': n,
                  'means': means,
                  'stds': stds,
                  'CI_upper': CI_upper,
                  'CI_lower': CI_lower,
                  'CI_upper_diff': CI_upper - means,
                  'CI_lower_diff': means - CI_lower,
                  'values': followups,
                  'weights': weights}

    return stats_dict

def get_pair_stats(followups_df, ps_df):
    
    '''
    grp_stats_dict: key = GRP_COL, values = stats_dict
    '''
    
    joined_df = followups_df.merge(ps_df, how = 'inner', left_on = [PID], right_on = [PID])
    grps = joined_df[GRP_COL].unique()
    
    pair_stats_dict = {}
    for grp in grps:
        s_df = joined_df.loc[joined_df[GRP_COL] == grp, :]
        followups = s_df.loc[:, FOLLOWUPS_COL].values.tolist()
        weights = s_df.loc[:, 'ps_w']
        pair_stats_dict[grp] = get_stats_from_followups(followups, weights = weights)

    return pair_stats_dict

def t_test_with_weights(stats_dict1, stats_dict2):
    
    from statsmodels.stats.weightstats import ttest_ind

    f_len = len(stats_dict1['means'])
    
    v1 = stats_dict1['values']
    w1 = stats_dict1['weights']
    
    v2 = stats_dict2['values']
    w2 = stats_dict2['weights']
    
    p_list = []
    for i in range(0, f_len):
        _, p, _ = ttest_ind(v1[:, i], v2[:, i], weights = (w1, w2))
        p_list.append(p)
        print(f'Time_{i}\t p={round(p,4)}')

    return p_list
    
def followups_visualization(grp_stats_dict,
                            errorbar = False,
                            ctrl_expolation = False,
                            add_patient_num = False,
                            additional_title = '',
                            v_name = BIO_COL):
    '''
    Setting for eGFR:
    - figure_size = (5,5)
    - ylim = (-20, 10)
    - ctrl_expolation = True (set in the caller)
    - add_patient_num = True (set in the caller)
    - legend_pos = 'lower left'
    
    Setting for BMI:
    - figure_size = (5,2)
    - ylim = (-2, 2)
    - ctrl_expolation = False (set in the caller)
    - add_patient_num = False (set in the caller)
    - legend_pos = 'upper right'
    
    '''

    
    #obtain the p_list of pairwise comparison
    if len(grp_stats_dict) == 2:
        grps_list = list(grp_stats_dict.values())
        p_list = t_test_with_weights(grps_list[0], grps_list[1])
    else: 
        p_list = []

    x_len = len(list(grp_stats_dict.values())[0]['means']) #length of data
    
    sample_size_str = '#patients:\n'
    for grp, stats_dict in grp_stats_dict.items():
        sample_size_str =  sample_size_str + ' ' + grp + ':' + str(stats_dict['sample_size'])

    #visualization
    fig, ax = plt.subplots()
    fig.set_size_inches(5,5)
    
    title = 'Sample size'
    ylabel = v_name + ' difference\n to baseline'
    ax.set_ylim(-20, 10)

    title += '\n'
    title += sample_size_str
    
    if add_patient_num:
        ax.set_title(sample_size_str)
    ax.set_xlabel('Time(yrs)')
    ax.set_ylabel(ylabel)
        

    ax.set_xlim(0,x_len)
    xticks = range(0,x_len)
    xtick_labels = [str(tick - BASELINE_IDX) for tick in xticks]
    xtick_labels[0] = 'control'
    xtick_labels[1] = 'baseline'
    #add an asterisk(*) if p < 0.05
    if len(p_list) > 0:
        xtick_labels = [xtick_labels[i]+'*' 
                        if (len(p_list)>i and p_list[i]<0.05) else xtick_labels[i] 
                        for i in range(0,len(xtick_labels))]
        
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    label_list = []
    colors = ['tab:blue',
             'tab:orange',
             'tab:green',
             'tab:red',
             'tab:purple']
    color_idx = 0
    for grp, stats_dict in grp_stats_dict.items():
        n = len(stats_dict['means'])
        ax.axvline(x=BASELINE_IDX, color = 'k')
        if errorbar:
            yerr = [stats_dict['CI_lower_diff'], stats_dict['CI_upper_diff']]
        else:
            yerr = None

        #real curve
        x = list(range(0,x_len))
        y = stats_dict['means']
        ax.errorbar(x = x,
                    y = y,
                    yerr = yerr,
                    label = grp + ' (real)',
                    color = colors[color_idx],
                    capsize = 2,
                    marker = 'o')
        
        #control curve
        if ctrl_expolation:
            c_x = list(range(BASELINE_IDX, x_len))
            c_slope = y[BASELINE_IDX] - y[BASELINE_IDX-1]
            c_y = [(element - BASELINE_IDX) * c_slope for element in c_x]
            ax.plot(c_x, 
                    c_y, 
                    color = colors[color_idx],
                    marker = 'x',
                    linestyle = '--',
                    label = (grp +' (control)'))
        
        color_idx += 1

    ax.legend(loc = 'lower right')
    ax.grid()

    return

def obtain_CI_and_std_for_baseline_eGFR(raw_baseline_vars_df, grp_list = GRP_LIST):
    
    eGFR_col = 'eGFR'
    
    mean_df = raw_baseline_vars_df.groupby(grp_list).apply(lambda df: df[eGFR_col].mean())
    std_df = raw_baseline_vars_df.groupby(grp_list).apply(lambda df: df[eGFR_col].std())
    CI_df = raw_baseline_vars_df.groupby(grp_list).apply(lambda df: get_CI95(df[eGFR_col].values))
    
    concat_df = pd.concat([mean_df.round(2), std_df.round(2), CI_df], axis = 1)
    concat_df.columns = ['mean', 'std', '95_CI']
    
    return concat_df

def obtain_num_of_patient_with_condition(raw_baseline_vars_df, grp_list = GRP_LIST):
    
    eGFR_col = 'eGFR'
    
    size_df = raw_baseline_vars_df.groupby(grp_list).size()
    
    eGFR_below60_df = raw_baseline_vars_df.loc[raw_baseline_vars_df[eGFR_col]<60, :]
    cond_size_df = eGFR_below60_df.groupby(grp_list).size()
    
    cond_pct_df = raw_baseline_vars_df\
                            .groupby(grp_list)\
                            .apply(lambda df: (df[eGFR_col]<60).sum()/len(df)*100)
    
    concat_df = pd.concat([size_df, cond_size_df, cond_pct_df.round(2)], axis = 1)
    concat_df.columns = ['#patient','#eGFR<60','pct(%)']
    
    return concat_df

def select_patient_with_followup_yrs(followups_df, 
                                     baseline_vars_df, 
                                     followup_yrs):
    
    '''
    followups_df = (PID, f_col)
    '''    
    
    followup_len = followup_yrs + BASELINE_IDX + 1
    mask = followups_df[FOLLOWUPS_COL].apply(lambda v: True if len(v) >= followup_len else False)
    followups_df_s = followups_df.loc[mask, :].reset_index(drop = True)
    followups_df_s[FOLLOWUPS_COL] = followups_df_s[FOLLOWUPS_COL].apply(lambda v: v[0:followup_len])
    
    pids = followups_df_s[PID].unique()
    mask = baseline_vars_df[PID].isin(pids) 
    baseline_vars_df_s = baseline_vars_df.loc[mask, :].reset_index(drop = True)

    return followups_df_s, baseline_vars_df_s

def pairwise_mean_eGFR_comparison(selected_followup_df, matched_dict, baseline_idx):
    
    for key, matched in matched_dict.items():
        base, comparator = key
        print(key)
        base_pids = matched['base_id'].to_list()
        comp_pids = matched['comp_id'].to_list()
        pid_list = base_pids + comp_pids
        print(f'sample of base_pids: {np.random.choice(base_pids, size = 5)}')
        print(f'sample of comp_pids: {np.random.choice(comp_pids, size = 5)}')
        
        mask = selected_followup_df[PID].isin(pid_list)
        selected_followup_df2 = selected_followup_df.loc[mask, :]

        sample_size = len(matched)
        additional_title = f'(sample_size:{sample_size})'
        grps_dict, size_df = follow_ups_visualization_by_group(selected_followup_df2, 
                                                      grp_list = GRP_LIST, 
                                                      output_option = 'diff_to_baseline',
                                                      errorbar = True,
                                                      baseline_idx = baseline_idx,
                                                      class_marker = False,
                                                      additional_title = additional_title)
    
    return 


def eGFR_mean_curve_entire_population(eGFR_df):
    
    value_attr = 'eGFR'
    trans_value_df = eGFR_df.groupby(PID)\
                            .progress_apply(lambda df: df[[DATE, value_attr]].values)\
                            .reset_index().rename(columns = {0:'values_np'})
    
    inter_v_df = trans_value_df.copy()
    inter_v_df.loc[:, 'follow_ups'] = trans_value_df.progress_apply(lambda row: get_follow_up_list_simplified(np.array(row['values_np']), unit_day_len = 365),
                                              axis = 1)
    inter_v_df.loc[:, 'grp'] = 'entire_population'
    
    for output_option in ['raw', 'diff_to_baseline']:
        grps_dict, size_df = follow_ups_visualization_by_group(inter_v_df, 
                                                  grp_list = ['grp'], 
                                                  output_option = output_option,
                                                  errorbar = False,
                                                  class_marker = False)
        
        print(f'Sample size of each data point in {output_option.upper()} curve:')
        print(size_df)

    return inter_v_df

def get_followups_with_control(drug_class_df, biomarker_df):
    
    #col_name = 'follow_ups'
    followups_ctrl_df = adding_followup_values(drug_class_df,
                           date_attr = 'D_start_date', 
                           end_date_attr = 'D_end_date',
                           value_df = biomarker_df,
                           value_attr = BIO_COL,
                           with_control = True,
                           unit_day_len = 365).rename(columns = {'follow_ups': FOLLOWUPS_COL})
    
    return followups_ctrl_df[[PID, FOLLOWUPS_COL]]

def get_followups_with_control_addtional_traj(drug_class_df, traj_col):
    
    traj_df = pd.read_pickle(PRO_PATH+traj_col+'_df.pklz', compression = 'gzip')
    followups_col = traj_col+'_'+FOLLOWUPS_COL
    
    #col_name = 'follow_ups'
    followups_ctrl_df = adding_followup_values(drug_class_df,
                           date_attr = 'D_start_date', 
                           end_date_attr = 'D_end_date',
                           value_df = traj_df,
                           value_attr = traj_col,
                           with_control = True,
                           unit_day_len = 365,
                           baseline_avg = True).rename(columns = {'follow_ups': followups_col})
    
    return followups_ctrl_df[[PID, followups_col]]
