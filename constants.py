# -*- coding: utf-8 -*-

import pandas as pd

RAW_PATH = ''
PRO_PATH = ''

#general parameters
PID = 'patientcode'
DATE = 'Date'
BASE_DATE = 'baseVisitDate'
VISIT_DATE = DATE
CC_DATE = 'dateCC'
LAB = 'LDL'
CC_yrs = 'CC_yrs_after_baseVisit'
CC_list = ['Kidney','Eye','Foot','Macrovascular']

#for HTN drug comparison

HTN_drug_classes = ['ACEI',
                    'ARB',
                   'BB',
                   'CCB',
                   'D',
                   'AB',
                   'OutOfMainClasses']

#maximum recommended dose in mg
MAX_DOSE = {'Enalapril': 40,
            'Lisinopril': 40, 
            'Perindopril': 16, 
            'Captopril': 150, 

            'Losartan': 100, 
            'Valsartan': 320, 
            'Telmisartan': 80, 
            'Candesartan': 32, 
            'Olmesartan': 40,
 
            'Atenolol': 100, 
            'Bisoprolol': 5,
            'Propranolol': 640, 
            
            'Amlodipine': 10, 
            'Nifedipine LA': 120, 
            
            'Hydrochlorothiazide': 50, 
            'Spironolactone': 50, 
            'Indapamide': 5, 
            
            'Terazosin': 'NA', 
            'Prazosin': 'NA',

            'Hydralazine': 'NA', 
            'Methyldopa': 'NA',
            'Nifetex': 'NA', 
            'Hydrochlorothiazide_Amiloride': 'NA'
}

DRUG2CLASS = {'Enalapril': 'ACEI',
            'Lisinopril': 'ACEI', 
            'Perindopril': 'ACEI', 
            'Captopril': 'ACEI', 

            'Losartan': 'ARB', 
            'Valsartan': 'ARB', 
            'Telmisartan': 'ARB', 
            'Candesartan': 'ARB', 
            'Olmesartan': 'ARB',
 
            'Atenolol': 'BB', 
            'Bisoprolol': 'BB',
            'Propranolol': 'BB', 
            
            'Amlodipine': 'CCB', 
            'Nifedipine LA': 'CCB', 
            
            'Hydrochlorothiazide': 'D', 
            'Spironolactone': 'D', 
            'Indapamide': 'D', 
            
            'Terazosin': 'AB', 
            'Prazosin': 'AB',

            'Hydralazine': 'OutOfMainClasses', 
            'Methyldopa': 'OutOfMainClasses',
            'Nifetex': 'OutOfMainClasses', 
            'Hydrochlorothiazide_Amiloride': 'OutOfMainClasses'
}

CLASS_MAPPING_DICT = {'ACEI': 'ACEIs',
                      'ARB': 'ARBs',
                      'BB': 'OTHERS',
                      'CCB': 'OTHERS',
                      'D': 'OTHERS'
}

DUMMIES_DROP_FIRST = False
DRUG_LIST = list(DRUG2CLASS.keys())
GRP_LIST = ['drug_class']
#GRP_LIST = ['drug_class','drug_intensity']
GRP_COL = 'grp_name'
FOLLOWUPS_COL = 'follow_ups_with_control'
BASELINE_IDX = 1

SELECTED_CLASSES = ['class_ACEI',
                    'class_ARB',
                   'class_BB',
                   'class_CCB',
                   'class_D']

eGFR_OPTION = 'stage' # in ['stage', 'value', 'pct']
eGFR_PCT = None

cohort_DM = None
cohort_CKD = None

BIO_COL = 'eGFR'
TRAJ_COL = 'BP_S'

