import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from constants import *
from helper import *

tqdm.pandas()

def demo_for_pid(pid):
    '''
    return gender, race, DOB 
    '''
    
    patientBased = pd.read_pickle(RAW_PATH + 'vector_PatientBased.pklz', compression = 'gzip')
    gender_race_df = patientBased[[PID, 'genderdesc','racedesc1']]
    
    raw_demo = pd.read_pickle(RAW_PATH + 'raw_demo1.pklz', compression = 'gzip')
    DOB_df = raw_demo[[PID,'birthdateyyyymmdd1']]
    
    gender = gender_race_df.loc[gender_race_df[PID] == pid, 'genderdesc'].iloc[0]
    race = gender_race_df.loc[gender_race_df[PID] == pid, 'racedesc1'].iloc[0]
    DOB = DOB_df.loc[DOB_df[PID] == pid, 'birthdateyyyymmdd1'].iloc[0]
    age_at_2012Jan = (pd.to_datetime('2012-01-01') - DOB).days/365
    
    demo_dict = {'gender': gender, 
                'race': race,
                'age_at_2012Jan': round(age_at_2012Jan,1)}
    
    return demo_dict

def extract_DHL_CC_dates(pid):
    
    DHL_date = pd.read_pickle(RAW_PATH + 'patType_diagnosticDate.pklz', compression = 'gzip')
    CC_date = pd.read_pickle(PRO_PATH + 'all_CC_dates.pkl')
    
    res_dict = {'DM_diag':[],
                'HLD_diag': [],
                'HTN_diag': [],
                'Macrovascular_diag': [],
                'Kidney_diag': [], 
                'Eye_diag': [], 
                'Foot_diag': []
               }
    
    
    #DHL date
    pat_DHL = DHL_date.loc[DHL_date[PID] == pid, :]
    if pat_DHL.empty:
        pass
    else:
        for DHL_cat in ['DM', 'HLD', 'HTN']:
            if pd.notna(pat_DHL['diagnosed_' + DHL_cat + '_date'].iloc[0]):
                res_dict[DHL_cat+'_diag'] += list(pat_DHL['diagnosed_' + DHL_cat + '_date'].values)
    
    #CC date
    pat_CC = CC_date.loc[CC_date[PID] == pid, :]
    if pat_CC.empty:
        pass
    else:
        for CC_cat in ['Macrovascular', 'Kidney', 'Eye', 'Foot']:
            if pd.notna(pat_CC[CC_cat + '_date'].iloc[0]):
                res_dict[CC_cat+'_diag'] += list(pat_CC[CC_cat + '_date'].values)

    return res_dict

def extract_treatment_dates(pid):
    
    res_dict = {'DM_drugs':[],
                'HLD_drugs': [],
                'HTN_drugs': []
               }
    
    for DHL_cat in ['DM', 'HLD', 'HTN']:
        df = pd.read_pickle(PRO_PATH + 'med'+DHL_cat+'_class_df.pklz', compression = 'gzip')
        pat_df = df.loc[df[PID] == pid,:]
        if pat_df.empty:
            pass
        else:
            res_dict[DHL_cat+'_drugs'] = list(pat_df[DATE].values)

    return res_dict

def extract_biomarker_dates(pid):
    
    lab_list = ['BP_S', 'BP_D', 'HbA1c', 'LDL', 'BMI']
    
    res_dict = {lab:[] for lab in lab_list}
    
    lab_dict = load_lab_dict(lab_list)
    
    for lab, df in lab_dict.items():
        pat_df = df.loc[df[PID] == pid, :]
        if pat_df.empty:
            pass
        else:
            res_dict[lab] = list(pat_df[DATE].values)
            
    #eGFR
    eGFR_df = pd.read_pickle(PRO_PATH+'eGFR_df.pklz', compression = 'gzip')
    pat_df = eGFR_df.loc[eGFR_df[PID] == pid, :]
    
    if pat_df.empty:
        res_dict['eGFR'] = []
    else:
        res_dict['eGFR'] = list(pat_df[DATE].values)
            
    return res_dict

def date_visualization_per_patient(pid):
    
    '''
    Date of DHL diagnosis
    Date of CC diagnosis
    Date of drug prescription
    Date of biomarker measurement
    
    Data structure: dict = {'label': [list of dates]}
    '''
    print('Extracting dates...')
    DHL_CC_dict = extract_DHL_CC_dates(pid)
    treatment_dict = extract_treatment_dates(pid)
    biomarker_dict = extract_biomarker_dates(pid)
    comb_dict = DHL_CC_dict | treatment_dict | biomarker_dict

    total_len = len(DHL_CC_dict) + len(treatment_dict) + len(biomarker_dict)
    
    print('Generating figures...')
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    ax.set_title(f'Dates visualization for PID = {pid}')
    ax.set_xlabel('Time')
    ax.set_xlim(pd.to_datetime('2009-01-01'),pd.to_datetime('2022-12-31'))
    ax.set_ylim(0, total_len+1)
    ax.set_ylabel('No meaning')
    
    y_idx = total_len
    for label, dates in DHL_CC_dict.items():
        ax.scatter(dates, [y_idx]*len(dates), label = label, marker = '*')
        y_idx -= 1

    for label, dates in treatment_dict.items():
        ax.scatter(dates, [y_idx]*len(dates), label = label)
        y_idx -= 1

    for label, dates in biomarker_dict.items():
        ax.scatter(dates, [y_idx]*len(dates), label = label, marker = 'x')
        y_idx -= 1

    ax.legend()

    return comb_dict

def traj_visualization(pid_list, value_df, value_attr, marked_dates = None, legend = True):
    
    fig,ax = plt.subplots()
    fig.set_size_inches(15,3)
    
    ax.set_title(f'Trajectory of {value_attr}')
    
    ax.set_xlim(pd.to_datetime('2009-01-01'), pd.to_datetime('2022-12-31'))
    ax.set_ylabel(value_attr)
    ax.set_xlabel('Time')
    
    for pid in pid_list:
        data_df = value_df.loc[value_df[PID] == pid, :]
        if not data_df.empty:
            X = data_df[DATE]
            Y = data_df[value_attr]
            ax.plot(X, Y, marker = 'o', label = pid)
    
    if len(marked_dates) > 0:
        for date in marked_dates:
            ax.axvline(x = date, color = 'r')
    
    if legend:
        ax.legend()
    
    return

def traj_visualization_per_patient(pid, marked_dates):
       
    lab_list = ['eGFR','SCr','LDL', 'HbA1c', 'BP_S', 'BP_D', 'BMI']
    
    lab_dict = {}
    
    for lab in lab_list:
        lab_dict[lab] = pd.read_pickle(PRO_PATH+lab+'_df.pklz', compression = 'gzip')
        
    for lab in lab_list:
        traj_visualization([pid], lab_dict[lab], lab, marked_dates)

    return

def get_med_from_row(row):
    
    drug_list = []
    drug_names = row.index[2:]
    
    for drug in drug_names:
        if row[drug] != 0:
            drug_list.append((drug, row[drug]))
            
    res_dict = {PID: row[PID],
               DATE: row[DATE],
               'drug_list': drug_list}
    
    return pd.Series(res_dict)

def drug_history_per_patient(pid):
    
    DHL_list = ['HTN', 'DM', 'HLD']
    
    drug_dict = {}
    
    for DHL_cat in DHL_list:
        drug_dict[DHL_cat] = pd.read_pickle(PRO_PATH+'med'+DHL_cat+'_df.pklz', compression = 'gzip')
        
    for DHL_cat in DHL_list:
        df = drug_dict[DHL_cat]
        df_s = df.loc[df[PID] == pid,:]
        df_mapping = df_s.apply(lambda row: get_med_from_row(row), axis =1)
        print(f'----------------{DHL_cat} drugs:----------------')
        print(df_mapping, '\n')
        
    return