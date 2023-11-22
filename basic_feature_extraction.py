import pandas as pd
import numpy as np
from tqdm import tqdm
from constants import *
from helper import *

tqdm.pandas()

def adding_gender_race_variables(pid_date_df, 
                              date_attr = 'change_date', 
                              with_dummies = True,
                              drop_ref = True):
    
    patientBased = pd.read_pickle(RAW_PATH + 'vector_PatientBased.pklz', compression = 'gzip')
    
    gender_race_df = patientBased[[PID, 'genderdesc','racedesc1']]

    joined_df = pid_date_df.merge(gender_race_df, how='left', left_on=[PID], right_on=[PID]).reset_index(drop = True)
    
    if not with_dummies:
        return joined_df
    else:
        #create dummies
        dumpy_gender = pd.get_dummies(joined_df['genderdesc'], prefix='gender')
        dumpy_race = pd.get_dummies(joined_df['racedesc1'], prefix='race')
        joined_df = pd.concat([joined_df, dumpy_gender, dumpy_race], axis=1)
        
        #drop reference columns
        if drop_ref:
            joined_df.drop(['gender_Female', 'race_Chinese'], axis = 1, inplace = True)
        else:
            pass
        
        #drop columns
        return joined_df.drop(['genderdesc', 'racedesc1'], axis=1)

def add_age_at_date(pid_date_df, 
                    date_attr = 'change_date'): 
                                  
    raw_demo = pd.read_pickle(RAW_PATH + 'raw_demo1.pklz', compression = 'gzip')
    DOB_df = raw_demo[[PID,'birthdateyyyymmdd1']]
                                  
    joined_df = pid_date_df.merge(DOB_df,how='left',left_on=[PID],right_on=[PID])
    
    #Age
    joined_df['Age'] = (joined_df[date_attr]-joined_df['birthdateyyyymmdd1']).apply(lambda x:round(x.days/365,1))
    
    return joined_df.drop(['birthdateyyyymmdd1'],axis=1)

def add_age_band_at_date(pid_date_df, 
                        date_attr = 'change_date',
                        with_dummies = True,
                        drop_ref = True): 
    
    band_len = 10
    
    raw_demo = pd.read_pickle(RAW_PATH + 'raw_demo1.pklz', compression = 'gzip')
    DOB_df = raw_demo[[PID,'birthdateyyyymmdd1']]
    
    pid_date_df.reset_index(drop = True, inplace = True)                              
    joined_df = pid_date_df.merge(DOB_df,how='left',left_on=[PID],right_on=[PID])
    
    #Age
    ages = (joined_df[date_attr]-joined_df['birthdateyyyymmdd1']).apply(lambda x:round(x.days/365,0))
    age_bands = ((ages//band_len)*band_len).apply(lambda v: str(int(v))+'_to_'+str(int(v+band_len-1)))
    
    if not with_dummies:
        joined_df['age_band'] = age_bands
        return joined_df[[PID, date_attr, 'age_band']]
    else:
        dumpy_age_band = pd.get_dummies(age_bands, prefix='age')
        
        if drop_ref:
            first_col = dumpy_age_band.columns[0]
            dumpy_age_band.drop([first_col], axis=1, inplace = True)
            print(f'Generating dummies for `age_band`, {first_col} column is dropped.')
        else:
            pass
        
        
        dummy_df = pd.concat([pid_date_df, dumpy_age_band], axis = 1)
        return dummy_df

def adding_DHL_diagnostic_variables(pid_date_df, 
                                    date_attr = 'change_date', 
                                    DHL_list = ['DM','HLD','HTN']):
    '''
    True - with diagnose at 'change_date'
    False - without diagnose at 'change_date'
    
    condition: diag = (date is not null) and (date<=change_date)
    '''
    
    if DHL_list and not set(DHL_list).issubset({'DM','HLD','HTN'}):
        raise ValueError('Incorrect parameter DHL_list')
        return 
    
    DHL_date = pd.read_pickle(RAW_PATH + 'patType_diagnosticDate.pklz', compression = 'gzip')

    date_cols = ['diagnosed_' + DHL_cat + '_date' for DHL_cat in DHL_list]
    DHL_date_selected = DHL_date[[PID] + date_cols]
    DHL_joined = pid_date_df.merge(DHL_date_selected,
                                      how = 'left',
                                      left_on = PID,
                                      right_on = PID)
    for DHL_cat in DHL_list:
        DHL_joined[DHL_cat+'_diag'] = (DHL_joined['diagnosed_' + DHL_cat + '_date'].notna())\
                                       &(DHL_joined['diagnosed_' + DHL_cat + '_date'] <= DHL_joined[date_attr])

    DHL_joined.drop(date_cols, axis=1, inplace=True)
    
    return DHL_joined

def adding_CC_diagnostic_variables(pid_date_df,
                                   date_attr = 'change_date', 
                                   CC_list = ['Macrovascular','Kidney','Eye','Foot']):
    '''
    True - with diagnose at 'change_date'
    False - without diagnose at 'change_date'
    
    condition: diag = (date is not null) and (date<=change_date)
    '''
    
    if CC_list and not set(CC_list).issubset({'Macrovascular', 'Kidney', 'Eye', 'Foot'}):
        raise ValueError('Incorrect parameter CC_list')
        return 
    
    CC_date = pd.read_pickle(PRO_PATH + 'all_CC_dates.pkl')

    #CC
    CC_cols = [CC+'_date' for CC in CC_list]
    CC_date_selected = CC_date[[PID] + CC_cols]
    CC_joined = pid_date_df.merge(CC_date_selected,
                                     how = 'left',
                                     left_on = PID,
                                     right_on = PID)
    for CC in CC_list:
        CC_joined[CC + '_diag']= (CC_joined[CC + '_date'].notna())\
                                &(CC_joined[CC + '_date'] <= CC_joined[date_attr])
    CC_joined.drop(CC_cols,
                   axis=1,
                   inplace=True)
    
    return CC_joined

def preprocess_lab_dict(lab_dict,pid_list,min_date,max_date):
    '''
    Reduce the size of 'lab_df' by 
    (1) removing rows with date out of the range [min_date,max_date], and 
    (2) remove rows with pid not in 'pid_list'
    '''
    
    new_lab_dict = {}
    
    for lab_name, lab_df in lab_dict.items():
        
        mask = ((lab_df[PID].isin(pid_list))&
                (lab_df[DATE] >= min_date)&
                (lab_df[DATE] <= max_date))
        lab_df = lab_df.loc[mask, :].reset_index(drop = True)
        
        new_lab_dict[lab_name] = lab_df

    return new_lab_dict

def get_first_day(date,unit):
    
    if unit == 'week':
        return date-pd.Timedelta(date.dayofweek, unit='day')
    elif unit == 'month':
        return date - pd.Timedelta(date.day-1, unit='day')
    else:
        return date


def adding_baseline_lab_values_from_interpolated_df(lab_inter_dict, change_date_df, lab_list=None):
    '''
    Find the interpolated lab value in the same [week] as `change_date`.
    
    #need to change it to month.
    '''
    
    unit = 'month'
    
    if lab_list is None:
        lab_list = list(lab_inter_dict.keys())
 
    #reduce size of 'lab_df'
    max_date = change_date_df['change_date'].max()
    min_date = change_date_df['change_date'].min()
    min_first_date = get_first_day(min_date,unit)
   
    pid_list = change_date_df[PID].unique()
    lab_dict = preprocess_lab_dict(lab_inter_dict,pid_list,min_first_date,max_date)
    
    #baseline lab values
    print('Extracting baseline lab values...')
    baseline_dict = {}
    for lab_name in lab_list:
        lab_df = lab_dict[lab_name]
        print(f'Associating baseline lab values for {lab_name}.')
        pid_np = lab_df[PID].values
        date_np = lab_df[DATE].values
        X_pid_date_v = change_date_df[[PID,'change_date']].values
        baseline_v = []
    
        for i in tqdm(range(0,len(change_date_df))):

            pid,change_date = X_pid_date_v[i]
            pre_lab_date = get_first_day(change_date,unit)
            pre_lab_date_np = np.datetime64(pre_lab_date)
            
            bool_f = (pid_np==pid)&(date_np==pre_lab_date_np)
            s = lab_df.loc[bool_f,lab_name]

            if s.empty==True:
                v = None
            else:
                v = s.iloc[0]
                
            baseline_v.append(v)
        baseline_dict['baseline_'+lab_name] = baseline_v  
    baseline_df = pd.DataFrame(baseline_dict)
    final_df = pd.concat([change_date_df,baseline_df],axis=1)

    return final_df

def adding_baseline_lab_values_from_lab_df_opt(pid_date_df, 
                                           lab_list, 
                                           date_attr = 'change_date', 
                                           wind_size=90,
                                           fillna = True):
    '''
    Find the real lab value within [wind_size] before the `change_date`.
    
    Idea: join the lab_df with window_df, and select the last row whose date within the window.
    '''
    
    lab_dict = load_lab_dict(lab_list)
    
    window_df = pid_date_df[[PID]].copy()
    window_df['end'] = pid_date_df[date_attr] #inclusive
    window_df['start'] = window_df['end'] - pd.Timedelta(wind_size-1,unit='day')
    
    df_list = []
    for lab_name, lab_df in lab_dict.items():
        print(f'Associating baseline lab values for {lab_name}.')
        lab_wind_df = lab_df.merge(window_df, 
                                     how = 'inner', 
                                     left_on = PID, 
                                     right_on = PID)
        s_lab_wind_df = lab_wind_df.loc[(lab_wind_df[DATE] >= lab_wind_df['start']) &
                                          (lab_wind_df[DATE] <= lab_wind_df['end']),
                                          [PID, DATE, lab_name]]
        s_lab_wind_df.sort_values([PID, DATE], inplace = True)
        baseline_lab_df = s_lab_wind_df.groupby(PID).apply(lambda df: df.iloc[-1]).reset_index(drop = True)
        df_list.append(baseline_lab_df.drop([DATE], axis = 1))
#         print(baseline_lab_df.head())

    #join the results
    final_df = pid_date_df.copy()
    for df in df_list:
        final_df = final_df.merge(df, how = 'left', left_on = PID, right_on = PID)
        
    print(final_df.head())
    if fillna:
        for lab_name, lab_df in lab_dict.items():
            lab_mean = lab_df[lab_name].mean()
            final_df.loc[:, lab_name].fillna(lab_mean, inplace = True)
        return final_df
    else:
        return final_df

def adding_baseline_drug_variables_opt(pid_date_df, 
                                       DHL_list = ['DM', 'HLD', 'HTN'],
                                       date_attr = 'change_date', 
                                       wind_size = 90):


    '''
    Find the real drug prescription within [wind_size] before the `change_date`.
    '''
    
    #load data
    drug_dict = {}
    for DHL_cat in DHL_list:
        drug_dict[DHL_cat] = pd.read_pickle(PRO_PATH + 'med' + DHL_cat + '_class_df.pklz', compression = 'gzip')
        
    window_df = pid_date_df[[PID]].copy()
    window_df['end'] = pid_date_df[date_attr] #inclusive
    window_df['start'] = window_df['end'] - pd.Timedelta(wind_size-1,unit='day')
        
    df_list = []
    for DHL_cat, drug_df in drug_dict.items():
        
        print(f'Associating baseline drug classes for {DHL_cat}.')
        classes = list(drug_df.columns[2:])
        
        data_wind_df = drug_df.merge(window_df, 
                                     how = 'inner', 
                                     left_on = PID, 
                                     right_on = PID)
        s_df = data_wind_df.loc[(data_wind_df[DATE] >= data_wind_df['start']) &
                                          (data_wind_df[DATE] <= data_wind_df['end']),
                                          [PID, DATE] + classes]
        s_df.sort_values([PID, DATE], inplace = True)
        baseline_df = s_df.groupby(PID).apply(lambda df: df.iloc[-1]).reset_index(drop = True)
        baseline_df.loc[:, classes] = (baseline_df.loc[:, classes] != 0).astype(int)
        
        class_prefixing = {c: DHL_cat+'_'+c for c in classes}
        baseline_df.rename(columns = class_prefixing, inplace = True)
        
        df_list.append(baseline_df.drop([DATE], axis = 1))
        
    #join the results
    final_df = pid_date_df.copy()
    for df in df_list:
        final_df = final_df.merge(df, how = 'left', left_on = PID, right_on = PID)
        
    final_df.fillna(0, inplace = True)

    return final_df

def impute_missing_lab_values_with_MICE(pid_date_df, baseline_labs_df, df_list, lab_list, date_attr = 'change_date'):
    
    print('Imputing missing lab values with MICE......')
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    
    baseline_lab_list = ['baseline_'+lab for lab in lab_list]
    
    joined_df = join_df_list_with_change_date(pid_date_df,
                                              df_list+[baseline_labs_df],
                                              date_attr = date_attr)
    
    input_df = joined_df.drop([PID, date_attr], axis = 1)
    
    imp_mean = IterativeImputer(random_state=0)
    imputed = imp_mean.fit_transform(input_df)
    imputed_df = pd.DataFrame(imputed,columns = input_df.columns)
    
    imputed_labs = imputed_df[baseline_lab_list].copy()
    imputed_labs.loc[:, PID] = joined_df[PID]
    imputed_labs.loc[:, date_attr] = joined_df[date_attr]

    return imputed_labs

def impute_missing_lab_values_with_grand_mean(baseline_labs_df, lab_list):
    
    lab_dict = load_lab_dict(lab_list)
    
    for lab_name in lab_list:
        grand_mean = lab_dict[lab_name].loc[:, lab_name].mean()
        baseline_labs_df.loc[:,'baseline_'+lab_name] = baseline_labs_df['baseline_'+lab_name].fillna(grand_mean)

    return baseline_labs_df

def adding_CC_survial_variables(change_date_df,CC_multi_dates,CC_name):
    '''
    Return: sur_df = (PID,change_date,event,dur)
    '''
    
    #select dates for single CC
    #filter out dates before 2015-01-01
    #take the first date after 2015-01-01
    single_CC_date =  CC_multi_dates.loc[(CC_multi_dates['compType']==CC_name)&\
                                          (CC_multi_dates['date']>pd.to_datetime('2015-01-01'))\
                                          ,[PID,'date']]\
                                    .groupby(PID,as_index=False).apply(lambda df:df.iloc[0])
    
    #those with CC in next five years
    single_CC_date['dur'] = (single_CC_date['date']-pd.to_datetime('2015-01-01')).apply(lambda x:x.days/365)
    single_CC_date['event'] = True
    
    #those without CC in next five years
    sur_df = change_date_df.merge(single_CC_date[[PID,'dur','event']],how='left',left_on=PID,right_on=PID)\
                            .sort_values(PID)\
                            .reset_index(drop=True)
    sur_df['dur'] = sur_df['dur'].fillna(5)
    sur_df['event'] = sur_df['event'].fillna(False)
    
    return sur_df

def join_df_list_with_change_date(pid_date_df,
                                  df_list,
                                  date_attr = 'change_date'):
    
    joined_df = pid_date_df.copy()
    
    for df in df_list:
        joined_df = joined_df.merge(df, 
                                    how = 'left', 
                                    left_on = [PID, date_attr],
                                    right_on = [PID, date_attr])

    return joined_df

def join_df_list_on_same_key(df_list, 
                             key_list, 
                             how='outer'):
    
    joined_df = df_list[0].loc[:,key_list]
    
    for df in df_list:
        joined_df = joined_df.merge(df, 
                                    how = how, 
                                    left_on = key_list, 
                                    right_on = key_list)

    return joined_df

def drop_cols_before_fitting_model(input_df, 
                                   id_attrs = [], 
                                   dummies_prefixes = [],
                                   other_attrs = [],
                                   preserved_attrs = [],
                                   DUMMIES_DROP_FIRST = True):
    
    input_cols = list(input_df.columns)
    input_df_copy = input_df.copy()
    drop_list = []
    drop_list += other_attrs
    
    #id columns
    drop_list += id_attrs
    
    #dummies
    if not DUMMIES_DROP_FIRST:
        for prefix in dummies_prefixes:
            ptn = prefix+'_.*'
#             print(ptn)
            pfx_cols = get_matched_list(ptn, input_cols)
    
            if len(pfx_cols) > 1:
                drop_list.append(pfx_cols[0])
            else:
                pass

    #columns with zero variance
    var_s = input_df.drop(id_attrs + other_attrs, axis = 1).astype('float64').var(axis = 0)
    zero_var_cols = list(var_s[var_s == 0].index)
    drop_list += zero_var_cols
    
    #drop lists
    remained_cols = remove_element_from_list(input_cols, drop_list+preserved_attrs)
    remained_cols += preserved_attrs
    
    return input_df[remained_cols]
