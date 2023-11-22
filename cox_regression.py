import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()

from constants import *
from helper import *

from lifelines import CoxPHFitter

def Cox_for_multiple_base_classes(cox_vars_df, grp_list, dummies_prefixes):
        
    cox_vars_df_copy = generate_grp_tuple_col(cox_vars_df, grp_list)
    
    class_list = list(cox_vars_df_copy['grp_tuple'].unique())
    
    cox_weights_dict = {}
    comp_df_list = []
    
    for base_idx in range(0, len(class_list)):
        base_class = class_list[base_idx]
        print(f'base class: {base_class}')
        weights_df, comparator_df = Cox_for_single_base_class(cox_vars_df_copy, class_list, base_idx, dummies_prefixes)
        
        cox_weights_dict[base_class] = weights_df
        comp_df_list.append(comparator_df)
        
    res_df = pd.concat(comp_df_list, axis = 0).reset_index(drop = True)
    res_df_s = res_df.loc[res_df['P'] < 0.05,:]
    
    return cox_weights_dict, res_df


def select_rows_with_same_class(cox_res_df):
    
    '''
    cox_res_df = (base, comparator, HR, 95%_CI, P)
    '''
    
    res_df_copy = cox_res_df.copy()
    res_df_copy['b_class'] = res_df_copy['base'].apply(lambda s: s.split('_')[1])
    res_df_copy['c_class'] = res_df_copy['comparator'].apply(lambda s: s.split('_')[1])
    
    mask = (res_df_copy['b_class'] == res_df_copy['c_class'])
    s_df = cox_res_df.loc[mask, :].reset_index(drop = True)
    
    return s_df

def cox_with_strata(strata_dict, cohort_df, dummies_prefixes, event_params):
    
    res_list = []
    
    event_dur_df = obtain_event_dur_df_with_parameters(cohort_df, cohort_df, event_params)
    
    for key, df in strata_dict.items():
        base, comparator = key
        
        print(f'----------base:{base}, comparator:{comparator}----------')

        surv_df =  adding_dur_event_variables_from_surv_df(df, event_dur_df)
        surv_df['comparator'] = (surv_df['grp_tuple'] == comparator).astype(int)
        
#         event_summary = surv_df.groupby(['comparator']).apply(lambda df: df['event'].sum()/len(df))
#         print('event_summary\n', event_summary)
        
#         event_mean_dur = surv_df.groupby(['comparator']).apply(lambda df: df.loc[df['event']==1, 'dur'].mean())
#         print('event_mean_dur\n', event_mean_dur)

        surv_df_copy = drop_cols_before_fitting_model(surv_df, 
                                                   id_attrs = [PID, 'D_start_date'], 
                                                   dummies_prefixes = dummies_prefixes,
                                                   other_attrs = ['grp_tuple', 'propensity'],
                                                   preserved_attrs = ['event', 'dur'])
        
        cph = CoxPHFitter(penalizer=0.2)
        cph.fit(surv_df_copy, duration_col = 'dur', event_col = 'event', strata = ['stratum'])

        weights_df = cph.summary
        weights_formatted = formatting_weights_df(weights_df)

        res_series = weights_formatted.loc[weights_formatted['covariate'] == 'comparator', ['HR', '95%_CI', 'P']]
        res_series['base'] = base
        res_series['comparator'] = comparator
        res_list.append(res_series)

    final_df = pd.concat(res_list, axis = 0).reset_index(drop = True)
    final_df = final_df[['base', 'comparator', 'HR', '95%_CI', 'P']]
    
    res_same_class = select_rows_with_same_class(final_df)
    res_p005 = final_df.loc[final_df['P']<0.05, :].reset_index(drop = True)
    
    #sorting
    res_same_class = res_same_class.sort_values('HR', ascending = False).reset_index(drop = True)
    res_p005 = res_p005.sort_values('HR', ascending = False).reset_index(drop = True)
    
    event = event_params['eGFR_event']
    print(f'-------------------eGFR_event: {event.upper()}----------------')
    
    print('Dosage intensity comparison')
    print(res_same_class.to_string(), '\n')

    print(final_df.to_string(), '\n')
    
    return 


def cox_with_matching(ps_dict, cohort_df, matched_dict, dummies_prefixes, event_params):
    
    res_list = []
    event_dur_df = obtain_event_dur_df_with_parameters(cohort_df, cohort_df, event_params)
    
    for key, cox_var_df in ps_dict.items():
        base, comparator = key
        if (base, comparator) in matched_dict:
            matche_df = matched_dict[(base, comparator)]
        else:
            matche_df = matched_dict[(comparator, base)]
            
        matched_pids = set(matche_df['comp_id'].values)|set(matche_df['base_id'].values)
        
        print(f'----------base:{base}, comparator:{comparator}----------')
        
        cox_var_df = cox_var_df.loc[cox_var_df[PID].isin(matched_pids), :]
        surv_df =  adding_dur_event_variables_from_surv_df(cox_var_df, event_dur_df)
        surv_df['comparator'] = (surv_df['grp_tuple'] == comparator).astype(int)


        surv_df_copy = drop_cols_before_fitting_model(surv_df, 
                                                       id_attrs = [PID, 'D_start_date'], 
                                                       dummies_prefixes = dummies_prefixes,
                                                       other_attrs = ['grp_tuple', 'propensity', 'logit_ps'],
                                                       preserved_attrs = ['event', 'dur'])
            
        
        cph = CoxPHFitter(penalizer=0.2)
        cph.fit(surv_df_copy, duration_col = 'dur', event_col = 'event')

        weights_df = cph.summary
        weights_formatted = formatting_weights_df(weights_df)
        
        res_series = weights_formatted.loc[weights_formatted['covariate'] == 'comparator', ['HR', '95%_CI', 'P']]
        res_series['base'] = base
        res_series['comparator'] = comparator
        res_list.append(res_series)

        
    final_df = pd.concat(res_list, axis = 0).reset_index(drop = True)
    final_df = final_df[['base', 'comparator', 'HR', '95%_CI', 'P']]
    
    res_same_class = select_rows_with_same_class(final_df)
    res_p005 = final_df.loc[final_df['P']<0.05, :].reset_index(drop = True)
    
    #sorting
    res_same_class = res_same_class.sort_values('HR', ascending = False).reset_index(drop = True)
    res_p005 = res_p005.sort_values('HR', ascending = False).reset_index(drop = True)
    
    event = event_params['eGFR_event']
    print(f'-------------------eGFR_event: {event.upper()}----------------')
    
    print('Dosage intensity comparison')
    print(res_same_class.to_string(), '\n')
    
    print('Rows with P<0.05')
    print(res_p005.to_string(), '\n')
    
    return