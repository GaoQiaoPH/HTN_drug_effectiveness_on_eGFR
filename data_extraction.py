import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()

from constants import *
from basic_feature_extraction import *
from helper import *

def extract_drug_class_per_patient(pat_df):
    '''
    Return the 1st HTN drug prescription for each patient. Including drug_class, drug_name, time period and drug_num.
    - combine the consecutive same prescription into a sinlge long-range prescription.
    - Identify drug discontinutation if no refill within 365 days.
    '''

    data_dict = {'drug_class': np.nan,
                 'drug_name': np.nan,
                 'D_start_date': pd.NaT,
                 'D_end_date': pd.NaT,
                 'drug_num': np.nan
    }
    
    
    dates = pat_df.loc[:, DATE].values
    drugs = pat_df.loc[:, DRUG_LIST].values
    
    first_pre_vec = drugs[0,:]
    first_pre_idxs = np.where(first_pre_vec!=0)[0]
    first_pre_drug_names = [DRUG_LIST[idx] for idx in first_pre_idxs]
    first_pre_drug_classes = [DRUG2CLASS[drug] for drug in first_pre_drug_names]
    
    first_pre_drug_names.sort()
    first_pre_drug_classes.sort()
    
    drug_name_str = first_pre_drug_names[0]
    for drug in first_pre_drug_names[1:]:
        drug_name_str = drug_name_str + '_' + drug

    if ('AB' in first_pre_drug_classes) or ('OutOfMainClasses' in first_pre_drug_classes):
        drug_class = 'OutOfSelectedHTNClasses'
    else:
#         drug_class = class_1st_line[0]
#         for c in class_1st_line[1:]:
#             drug_class = drug_class + '_' + c
        if len(first_pre_drug_classes) == 1:
            drug_class = first_pre_drug_classes[0]
        else: 
            drug_class = str(len(first_pre_drug_names)) + '_drugs'

    data_dict['drug_class'] = drug_class
    data_dict['drug_name'] = drug_name_str
    data_dict['D_start_date'] = dates[0]
    data_dict['drug_num'] = len(first_pre_drug_names)

    #only one record for medication
    if len(dates) == 1:
        data_dict['D_end_date'] = data_dict['D_start_date'] + np.timedelta64(365-1,'D')
        return  pd.Series(data_dict)
        
    #multiple records for medication
    pre_date = dates[0]
    for i in range(1, len(dates)):
        pre_date = dates[i-1]
        curr_date = dates[i]
        curr_drug = drugs[i,:]

        if ((curr_date - pre_date)/np.timedelta64(1,'D')) > 365: #discontinuation
            data_dict['D_end_date'] = pre_date + np.timedelta64(365-1,'D')
            break
        elif not np.array_equal(first_pre_vec!=0, curr_drug!=0): #change medication, ignoring dosage change
            data_dict['D_end_date'] = curr_date - np.timedelta64(1,'D')
            break
        elif i == len(dates)-1:#last record
            data_dict['D_end_date'] = curr_date + np.timedelta64(365-1,'D')

    return pd.Series(data_dict)

def extract_init_drug_class(med_df):
    
    drug_class_df = med_df.groupby(PID)\
                        .progress_apply(lambda pat_df: extract_drug_class_per_patient(pat_df))\
                        .reset_index()

    return drug_class_df

def treatment_history_processing(intensity_df, drug_list, add_end_row = True):
    
    '''
     - merge same prescription, 
     - insert drug discontiutation
    '''
    
    data = intensity_df[[PID, DATE] + drug_list].values
    pids = data[:, 0]
    dates = data[:, 1]
    drugs = data[:, 2:]
    
    res = []
    res.append(data[0,:])
    
    for i in tqdm(range(1,len(data))):
        if pids[i] == pids[i-1]:
            if (dates[i] - dates[i-1]) <= np.timedelta64(365,'D') and (drugs[i,:] == drugs[i-1,:]).all():#same treatment
                pass
            elif (dates[i] - dates[i-1]) > np.timedelta64(365,'D'): #discontinuation
                vec = [0]*(len(drug_list)+2)
                vec[0] = pids[i-1]
                vec[1] = dates[i-1] + np.timedelta64(365,'D')
                res.append(np.array(vec))
                res.append(data[i,:])
            else: #difference drugs
                res.append(data[i,:])
        else:
            if add_end_row:
                #add the ending row
                vec = [0]*(len(drug_list)+2)
                vec[0] = pids[i-1]
                vec[1] = dates[i-1] + np.timedelta64(365,'D')
                res.append(np.array(vec))
            res.append(data[i,:])
            
    if add_end_row:
        #add the ending row
        vec = [0]*(len(drug_list)+2)
        vec[0] = pids[-1]
        vec[1] = dates[-1] + np.timedelta64(365,'D')
        res.append(np.array(vec))

    hist_df = pd.DataFrame(res, columns = [PID, DATE] + drug_list)
    
    return hist_df

def merge_same_prescription(df, drug_list):
    
    '''
     - merge same prescription, without considering the interval
    '''
    
    data = df[[PID, DATE] + drug_list].values
    pids = data[:, 0]
    dates = data[:, 1]
    drugs = data[:, 2:]
    
    res = []
    res.append(data[0,:])
    
    for i in tqdm(range(1,len(data))):
        if pids[i] == pids[i-1]:
            if (drugs[i,:] == drugs[i-1,:]).all():#same treatment
                pass
            else: #difference treatment
                res.append(data[i,:])
        else:
            res.append(data[i,:])

    hist_df = pd.DataFrame(res, columns = [PID, DATE] + drug_list)
    
    return hist_df

def map_intensity(med, dosage, max_dose, intensity_pct = 0.5):

    if dosage == 0 or max_dose == 'NA':
        return dosage
    else:
        return 'low' if dosage <= intensity_pct*max_dose else 'high'


def extract_HTN_treatment_history(med_df):
    
    #map drug dosage to drug intensity
    intensity_df = med_df.copy()
    for med, max_dose in MAX_DOSE.items():
        intensity_df[med] = intensity_df[med].progress_apply(lambda dosage: map_intensity(med,dosage, max_dose))

    treatment_hist_df = treatment_history_processing(intensity_df, DRUG_LIST)
    
    return treatment_hist_df

def extract_drug_class_and_intensity_per_patient(pat_df, drug_list):

    data_dict = {'drug_class': np.nan,
                 'drug_name': np.nan,
                 'drug_intensity': np.nan, 
                 'D_start_date': pd.NaT,
                 'D_end_date': pd.NaT,
    }
    
    
    dates = pat_df.loc[:, DATE].values
    drugs = pat_df.loc[:, drug_list].values
    
    drug_cnt = (drugs[0,:] != 0).sum()
    
    #not start with monotherapy
    if drug_cnt != 1:
        data_dict['drug_class'] = 'combined'
        return pd.Series(data_dict)
    
    #start with monotherapy
    drug_vec = drugs[0,:]
    index = np.where(drug_vec!=0)[0][0]
    drug_name = drug_list[index]
    data_dict['drug_name'] = drug_name
    data_dict['drug_intensity'] = drug_vec[index]
    data_dict['drug_class'] = DRUG2CLASS[drug_name]
    data_dict['D_start_date'] = dates[0]
    data_dict['D_end_date'] = dates[1] - np.timedelta64(1, 'D')

    return pd.Series(data_dict)

def extract_init_drug_class_with_intensity_hist(treatment_hist_df, drug_list):
    
    drug_class_df = treatment_hist_df.groupby(PID)\
                        .progress_apply(lambda pat_df: extract_drug_class_and_intensity_per_patient(pat_df, drug_list))\
                        .reset_index()

    return drug_class_df

def drug_vec_2_tuple_list(drug_vec, drug_list):
    
    tuple_list = []
    
    for i in range(0, len(drug_vec)):
        if drug_vec[i] != 0:
            drug_class = DRUG2CLASS[drug_list[i]]
            drug_tuple = (drug_class, drug_vec[i])
            tuple_list.append(drug_tuple)

    return tuple_list

def drug_tuple_list_comparison(pre_tuple_list, curr_tuple_list):
    
    if pre_tuple_list is None:
        return 'drug_initialization' #did not taking drugs before in history
    
    if len(pre_tuple_list) == 0:
        return 'drug_starting' #start from discontinuation
    
    if len(curr_tuple_list) == 0:
        return 'drug_stopping'
    
    if (len(pre_tuple_list) == 1) and (len(curr_tuple_list) == 1) and (pre_tuple_list[0][0] == curr_tuple_list[0][0]):
        return 'single_drug_titration'
    
    
    if (len(pre_tuple_list) == len(curr_tuple_list)):
        return 'same_drug_num' 
    
    if len(pre_tuple_list) < len(curr_tuple_list):
        return 'adding_drugs'
    
   
    if len(pre_tuple_list) > len(curr_tuple_list):
        return 'removing_drugs'

    return 'Others'

def extract_prescription_history_in_class_intensity_level_per_patient(pat_df, drug_list):
    
    '''
    res_df = (PID, D_start_date, D_end_date, drug_tuple_list, drug_num, change_remark)
    '''
    
    dates = pat_df.loc[:, DATE].values
    drugs = pat_df.loc[:, drug_list].values
    
    res_list = []
    one_day = np.timedelta64(1, 'D')
        
    pre_tuple_list = None
    res = []
    for i in range(0, len(pat_df)-1):
        drug_vec = drugs[i, :]
        curr_tuple_list = drug_vec_2_tuple_list(drug_vec, drug_list)
        change_remark = drug_tuple_list_comparison(pre_tuple_list, curr_tuple_list)
        row = [dates[i], dates[i+1] - one_day, curr_tuple_list, len(curr_tuple_list), change_remark]
        
        pre_tuple_list = curr_tuple_list
        res.append(row)
        
        
    res_df = pd.DataFrame(res, columns = ['D_start_date', 'D_end_date', 'drug_tuple_list', 'drug_num', 'change_remark'])
    
    return res_df


def extract_prescription_history_in_class_intensity_level(treatment_hist_df, drug_list):
    
    prep_hist_df = treatment_hist_df.groupby(PID)\
                        .progress_apply(lambda pat_df: extract_prescription_history_in_class_intensity_level_per_patient(pat_df, drug_list))\
                        .reset_index()
    
    
    
    
    return prep_hist_df.drop(['level_1'], axis = 1)




def basic_analysis_eGFR(eGFR_df):
    
    mean = eGFR_df['eGFR'].mean()
    std = eGFR_df['eGFR'].std()
    
    diff = eGFR_df.groupby(PID).progress_apply(lambda df: df.diff(axis = 0)).reset_index()
    diff_mean = diff['eGFR'].mean()
    diff_std = diff['eGFR'].std()
    
    print(f'For all eGFR values: mean = {mean:.2f}, std = {std: .2f}')
    print(f'For adjacent value difference of same patient: mean = {diff_mean:.2f}, std = {diff_std: .2f}')
    
    return

def event_checking(init_v, candidate_v, threshold, larger_better = False):
    '''
    larger_better = True (eGFR value): larger value -> getting better   
    
    larger_better = False (for eGFR stage): larger value -> getting worse
    '''
    
    if (abs(init_v - candidate_v) >= threshold):
        found = True
        if larger_better:
            remark = 'better' if candidate_v > init_v else 'worse' #larger value suggests 'better'
        else:
            remark = 'better' if candidate_v < init_v else 'worse' #larger value suggests 'better'
    else:
        found = False
        remark = 'no change'

    return found, remark

def find_eGFR_event(event_values, dates, init_idx, option, pct, event_str = 'worse'):
    
    '''
    return date, value, remark, for the found event
    '''
    
    if option == 'stage':
        value_threshold = 1
        larger_better = False
    elif option == 'value':
        value_threshold = 5
        larger_better = True
    else: #option == 'pct'
        value_threshold = pct * event_values[init_idx]
        larger_better = True
    
    init_value = event_values[init_idx]
    for i in range(init_idx+1, len(dates)-1):
        #set the found flag, and event string
        found1, remark1 = event_checking(init_value, event_values[i], value_threshold, larger_better = larger_better)
        found2, remark2 = event_checking(init_value, event_values[i+1], value_threshold, larger_better = larger_better)
        
        found = (found1 and found2)
        if found and (remark1 == event_str) and (remark2 == event_str):
            return dates[i+1], event_values[i+1], event_str
        else:
            pass
    
    return pd.NaT, np.nan, 'not_found'


def get_eGFR_stage_changing_per_patient_independent(pat_df, option, pct):
    
    if option not in ['stage', 'value', 'pct']:
        raise ValueError('Please specify a correct option in [\'stage\', \'value\', \'pct\'].')
        
    if option == 'stage':
        event_values = pat_df.loc[:, 'stage'].values
    elif (option == 'value') or (option == 'pct'):
        event_values = pat_df.loc[:, BIO_COL].values
    else: 
        raise ValueError('Please specify a correct option in [\'stage\', \'value\', \'pct\'].')
  
    D_start_date = pat_df.loc[:, 'D_start_date'].values[0]
    dates = pat_df.loc[:, DATE].values
    
    data_dict = {'B_init_date': pd.NaT,
                 'B_init_value': np.nan,
                 'B_better_date': pd.NaT,
                 'B_better_value': np.nan,
                 'B_worse_date': pd.NaT,
                 'B_worse_value': np.nan,
                 'B_remark': 'no change'
    }
    
#     stages = pat_df.loc[:, 'stage'].values
#     eGFRs = pat_df.loc[:, 'eGFR'].values
    
    #single record
    if len(pat_df) == 1:
        data_dict['B_remark'] = 'single record'
        return pd.Series(data_dict)
    
    #find baseline eGFR one year before the drug_start_date
    #time window: [drug_start - 365 + 1, drug_start]
    idx = 0
#     find_init = False
    wind_start = D_start_date - np.timedelta64(365-1, 'D')
    wind_end = D_start_date 
    candidate_idxs = []
    while (idx < len(pat_df)):
        #find the initial stage (nearest and prior to the index date)
        if (dates[idx] >= wind_start) and (dates[idx] <= wind_end): 
            candidate_idxs.append(idx)
        idx += 1
        
    if len(candidate_idxs) == 0: #no baseline value
        data_dict['B_remark'] = 'no baseline eGFR'
        return pd.Series(data_dict)
    
    init_idx = candidate_idxs[-1]
    data_dict['B_init_date'] = dates[init_idx] 
    data_dict['B_init_value'] = event_values[init_idx]
    
    if init_idx  == (len(pat_df) - 1): #the baseline value is the last value
        data_dict['B_remark'] = 'no follow-up eGFR'
        return pd.Series(data_dict)
    
    
    b_date, b_value, b_remark = find_eGFR_event(event_values, dates, init_idx, option, pct, event_str = 'better')
    w_date, w_value, w_remark = find_eGFR_event(event_values, dates, init_idx, option, pct, event_str = 'worse')
    
    data_dict['B_better_date'] = b_date
    data_dict['B_better_value'] = b_value
    data_dict['B_worse_date'] = w_date
    data_dict['B_worse_value'] = w_value
    
    if (b_remark == 'not_found') and (w_remark == 'not_found'):
        data_dict['B_remark'] = 'no change'
    elif (b_remark != 'not_found') and (w_remark != 'not_found'):
        data_dict['B_remark'] = 'both_better_worse'
    elif (b_remark != 'not_found'):
        data_dict['B_remark'] = 'better'
    else:
        data_dict['B_remark'] = 'worse'

    return pd.Series(data_dict)

def extract_eGFR_stage_changing(eGFR_df, drug_class_df, option = 'stage'):
    
    print(f'eGFR option: {option}')
    if option == 'pct':
        print(f'eGFR_PCT: {eGFR_PCT}')
    
    drug_start_df = drug_class_df.loc[:, [PID, 'D_start_date']]
    joined_df = eGFR_df.merge(drug_start_df, how='inner', left_on=[PID], right_on=[PID])
    
    stage_changing_df = joined_df.groupby(PID)\
                                .progress_apply(lambda pat_df: 
                                                get_eGFR_stage_changing_per_patient_independent(pat_df,
                                                                                                option,
                                                                                                pct = eGFR_PCT))\
                                .reset_index()
    
    return stage_changing_df

def get_HTN_diagnostic_date(DHL_date_df):
    
    HTN_date_df = DHL_date_df.loc[:, [PID, 'diagnosed_HLD_date']]\
                                .rename(columns = {'diagnosed_HLD_date':'HTN_date'})
    
    
    return HTN_date_df



