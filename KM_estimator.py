import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()

import itertools
from constants import *
from helper import *

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test,logrank_test

def KM_fit_and_visualization(KM_input_dict):

    fig,ax = plt.subplots()
    fig.set_size_inches(6, 6)
    
    title = f'KM for drug effectiveness on kidney protection.'
    ax.set_title(title)

    ax.set_xlabel('Time (yrs)',fontsize=12)
    ax.set_ylabel('Percent of patients',fontsize=12)
    ax.set_xlim(0, 11)
    ax.set_xticks(range(0, 11))
    ax.set_ylim(0, 0.5)
    
    for grp, grp_dict in KM_input_dict.items():
        kmf = KaplanMeierFitter()
        kmf.fit(grp_dict['durations'], grp_dict['events'], label = grp, weights = grp_dict['weights'])
        kmf.plot_cumulative_density(ci_show = True, at_risk_counts = False, ax=ax)

    #p-value
    pair_list = itertools.combinations(list(KM_input_dict.keys()), 2)
    p_list = []
    for pair in pair_list:
        logrank_res = logrank_test(KM_input_dict[pair[0]]['durations'],
                                   KM_input_dict[pair[1]]['durations'],
                                   KM_input_dict[pair[0]]['events'],
                                   KM_input_dict[pair[1]]['events'],
                                   weights_A = KM_input_dict[pair[0]]['weights'], 
                                   weights_B = KM_input_dict[pair[1]]['weights'])
        p_value = logrank_res.p_value
        p_list.append(p_value)
        print(f'{pair[0]} vs. {pair[1]}: p = {p_value:.4f}')
        
    if len(p_list) == 1 and p_list[0] < 0.05:
        ax.set_xlabel('Time (yrs) *')

    ax.legend()

    return