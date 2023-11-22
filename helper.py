import pandas as pd
import numpy as np


from constants import *

#compute the VIF of each column in df
def get_VIF_of_df(df):
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    threshold = 5
    
    df_np = df.values.astype('float64')
    count = 0
    for i in range(0,df.shape[1]):
        vif = variance_inflation_factor(df_np,i)
        if vif >= threshold:
            count += 1
        print(f'{df.columns[i]:20s} {vif:.2f}')
        
    print(f'\n{count} out of {df.shape[1]} ({count/df.shape[1]:2f}) columns have VIF >= {threshold}.')
    
    return

#covariance matrix visualization
def visualize_covariance_mtrx(df):
    
    import seaborn as sns
    import matplotlib.pyplot as plt


    
    columns = list(df.columns)
    df = df.astype(float)
    
    #normalize: max-min
    for col in columns:
        max_v = df[col].max()
        min_v = df[col].min()
        df.loc[:, col] = (df[col] - min_v)/(max_v - min_v)
    
    cov = df.cov()
    
    fig, ax = plt.subplots(figsize=(10,12))
    sns.heatmap(cov, center=0, cmap="YlGnBu")
    
    return cov

def remove_element_from_list(orig_list, r_list, by_set = False):
    '''
    for elements in `r_list`, if it is in `orig_list`, remove it.
    
    Complexity: O(3m+n)
        - m = len(orig_list)
        - n = len(r_list)
    '''
    if (not isinstance(orig_list, list)) or (not isinstance(r_list, list)):
        raise TypeError("Please pass two lists as parameter.")
    
    if by_set:
        #fast, but will change the element order
        return list(set(orig_list) - set(r_list))
    else:
        for r in r_list:
            if r in orig_list:
                orig_list.remove(r)
        return orig_list
    
    
def get_matched_list(ptn, str_list):
    
    import re
    
    matched_list = []
    
    for s in str_list:
        if re.match(ptn,str(s)):
            matched_list.append(s)
    
    return matched_list


def row_to_grp_str(row, grp_list):
    
    s = str(row[grp_list[0]])
    
    if len(grp_list) > 1:
        for i in range(1, len(grp_list)):
            s = s+'_'+str(row[grp_list[i]])
        
    return s

def generate_grp_tuple_col(df, grp_list):
    
    df_cols = list(df.columns)
    for col in grp_list:
        if col not in df_cols:
            raise ValueError(f'The input df did not contain column {col}.')
            
    if GRP_COL in df.columns:
        return df
    
    if len(GRP_LIST) == 1:
        df_copy = df.rename(columns = {GRP_LIST[0]: GRP_COL})
        return df_copy
            
    df_copy = df.copy()
    df_copy[GRP_COL] = df_copy.progress_apply(lambda row: row_to_grp_str(row, grp_list), axis = 1)
    df_copy.drop(grp_list, axis = 1, inplace = True)
    
    return df_copy

def get_event_dur(row):
    
    index_date_col = 'D_start_date'
    event_date_col = 'B_worse_date'
    end_date_col = 'D_end_date'
        
    if pd.isnull(row[event_date_col]):
        #event not found in the dataset: censoring
        event = False
        dur = (row[end_date_col] - row[index_date_col]).days/365
    elif row[event_date_col] <= row[end_date_col]:
        #event found
        event = True
        dur = (row[event_date_col] - row[index_date_col]).days/365
    else:
        #event not found in the defined study window: censoring
        event = False
        dur = (row[end_date_col] - row[index_date_col]).days/365

    data_dict = {PID: row[PID],
                'event': event,
                'dur': dur}
    
    return pd.Series(data_dict)

def load_lab_dict(lab_list):
    
    lab_dict={}
    for lab in lab_list:
        lab_dict[lab]=pd.read_pickle(PRO_PATH+lab+'_df.pklz',compression='gzip')
        print(lab,'loaded!')

    return lab_dict

def get_pids_taking_certain_drugs(med_df, drug_list):
    
    all_drug_list = list(med_df.columns[2:])
    drug_cnt_df = med_df.groupby(PID).progress_apply(lambda df: (df[all_drug_list]!=0).sum(axis=0)).reset_index()
    
    mask = np.array([False]*len(drug_cnt_df))
    for drug in drug_list:
        mask = mask | (drug_cnt_df[drug]!=0)
        
    pids = drug_cnt_df.loc[mask, PID].values
    
    return pids

def generate_base_comp_pair_list(all_grps, base_list = [], comp_list = [], remove_repeated = True):
  
    if len(all_grps) == 0:
        return []
    
    if not (set(base_list).issubset(set(all_grps)) and set(comp_list).issubset(set(all_grps))):
        raise ValueError('base_list or comp_list not in all_grps.')

    if len(base_list) == 0:
        base_list = all_grps
        
    if len(comp_list) == 0:
        comp_list = all_grps
   
    pair_list = []
    for base in base_list:
        for comp in comp_list:
            pair = (base, comp)
            if remove_repeated and base == comp:
                pass
            else:
                pair_list.append(pair)

    return list(set(pair_list))

def bc_comp_df_postprocessing(df, base, comp):
    
    num_variables = ['Age', 'BP_S', 'BP_D', 'HbA1c', 'LDL', 'BMI']
    cat_variables = ['gender_Female', 'gender_Male', 
                     'race_Chinese', 'race_Indian','race_Malay', 'race_OthersX',
                     'DM_diag', 'HLD_diag',
                     'Macrovascular_diag', 'Eye_diag', 'Foot_diag', 'Kidney_diag',
                     'DM_biguanides', 'DM_sulfonylureas', 'DM_dpp4_inhibitors',
                     'DM_sglt2_inhibitors', 'DM_alpha_glucosidase_inhibitors', 'DM_insulin', 
                     'HLD_hmgcoa_reductase_inhibitors', 'HLD_fibric_acid_derivatives', 
                     'HLD_cholesterol_absorption_inhibitors', 'HLD_bile_acid_sequestrants',
                     'eGFR_stage_1', 'eGFR_stage_2', 'eGFR_stage_3', 'eGFR_stage_4', 'eGFR_stage_5']
    
    var_seq = ['Age', 'gender_Male', 'gender_Female',
              'race_Chinese', 'race_Malay', 'race_Indian', 'race_OthersX',
              'HbA1c', 'LDL', 'BP_S', 'BP_D','BMI',
              'HLD_diag', 'DM_diag',
              'Macrovascular_diag', 'Kidney_diag',  'Eye_diag', 'Foot_diag',
              'DM_biguanides', 'DM_sulfonylureas', 'DM_dpp4_inhibitors',
              'DM_alpha_glucosidase_inhibitors', 'DM_insulin', 'DM_sglt2_inhibitors',
              'HLD_hmgcoa_reductase_inhibitors', 'HLD_fibric_acid_derivatives', 
              'HLD_cholesterol_absorption_inhibitors', 'HLD_bile_acid_sequestrants',
              'eGFR_stage_1', 'eGFR_stage_2', 'eGFR_stage_3', 'eGFR_stage_4', 'eGFR_stage_5']
 
    df_copy = df.copy()
    SMD_cols = ['bef_SMD', 'aft_SMD']
    mean_cols = ['bef_base_mean', 'bef_comp_mean', 'aft_base_mean', 'aft_comp_mean']
    
    # for SMD_cols , round(v, 2)
    df_copy.loc[:, ['bef_SMD', 'aft_SMD']] = df.loc[:, ['bef_SMD', 'aft_SMD']].round(2)
    
    # for mean_cols
    # - for num_variables, round(v, 1)
    # - for cat_varaibles, round(v*100, 1)
    df_copy.loc[df_copy.index.isin(num_variables), mean_cols] = \
                    df.loc[df.index.isin(num_variables), mean_cols].applymap(lambda v: round(v,1))
    
    df_copy.loc[df_copy.index.isin(cat_variables), mean_cols] = \
                    df.loc[df.index.isin(cat_variables), mean_cols].applymap(lambda v: round(v*100,1))
    
    df_copy.index = pd.Categorical(df_copy.index, var_seq)
    df_copy.sort_index(inplace = True)
    
    df_copy.rename(columns = {'bef_base_mean': 'bef_' + base,
                             'bef_comp_mean': 'bef_' + comp,
                             'aft_base_mean': 'aft_' + base,
                             'aft_comp_mean': 'aft_' + comp}, inplace = True)

    return df_copy

def df_dict_2_excel(df_dict, file_name):
    
    with pd.ExcelWriter(file_name) as writer:
        for key, df in df_dict.items():
            df_f = bc_comp_df_postprocessing(df, key[0], key[1])
            df_f.to_excel(writer, sheet_name = str(key)) 
    
    return
