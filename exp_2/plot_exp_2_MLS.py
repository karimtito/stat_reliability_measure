import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from stat_reliability_measure.dev.utils import float_to_file_float,get_sel_df
from stat_reliability_measure.home import ROOT_DIR
csv_path=os.path.join(ROOT_DIR,'logs/mnist_tests/agg_res.csv')
csv_res=pd.read_csv(csv_path)
csv_res['mean_log10_est']=csv_res['mean_log_est']/np.log(10)
csv_res['std_log10_est']=csv_res['std_log_est']/np.log(10)
csv_res=csv_res.round({'epsilon':2,'mean_calls':0})
csv_res=csv_res[(csv_res['model_name']!='dnn2_mnist') | (csv_res['epsilon']==0.15) | (csv_res['epsilon']==0.18)]
csv_res=csv_res[csv_res['n_rep']==100]
T_range=[5,10,20,50,100,200,500]
e_range=[0.85,]
N_range=[32,64,128,256,512,1024]
ratio_range=[0.1,0.6]
min_rate_smc=0.2
min_rate_ams=0.51
v_min_opt=True
amls_filtr= (csv_res['method']=='MLS_SMC')  
smc_filtr= ((csv_res['method']=='H_SMC') | 
            csv_res['method']=='MALA_SMC' | 
            csv_res['method']=='RW_SMC')
csv_res_webb=csv_res[amls_filtr]

csv_res_smc=csv_res[smc_filtr]

################################################################################
####################### Plot HMC-SMC results ######################################
################################################################################

log10=True
plt.figure(figsize=(16,9))
min_rate=0.4
N_max=1024
eps=0.15
ess_ref=0.85
L_ref=5
x_min=0
T_max=50
ref_df=get_sel_df(df=csv_res_smc,triplets=[('N',N_max,'='),('epsilon',eps,'='),('T',T_max,'='),(
        'ess_alpha',ess_ref,' ='),('L',L_ref,')')])
log_str='log10' if log10 else 'ln'
y_col='mean_log10_est' if log10 else 'mean_log_est' 
y_std_col='std_log10_est' if log10 else 'std_log_est' 
x_col='mean_calls'
ref_log_proba=ref_df.iloc[0][y_col]
N_=np.linspace(start=x_min,stop=int(1.5e6),num=10)
ref=ref_log_proba*np.ones(shape=(10,))
plt.plot(N_,ref,'--',label=f'Ref. {log_str} proba.')
x_max=int(1e6)

ess_alpha=0.85

L=5

only_duplicated=True
GV_opt=False
kernel='RW' if GV_opt else 'H'

triplets=[('method','smc','cont'),('epsilon',eps,'='),(x_col,x_max,'<='),
             ('L',L,'='),('only_duplicated',only_duplicated,'='),
            
                ('ess_alpha',ess_alpha,'='),('v_min_opt',v_min_opt,'='),('GV_opt',GV_opt,'=')]
method_str=f"{kernel}-SMC"
T_range=[1,10,]

for T in T_range:
    csv_smc_T = get_sel_df(df=csv_res_smc, cols=[x_col,'T'],vals=[20,T],conds=['>=','='],
                    triplets=triplets)[:6].sort_values(by=x_col)
    l = len(csv_smc_T)
    N_ = csv_smc_T[x_col].values

    print(l)
    
    N_ = csv_smc_T[x_col].values

    plt.errorbar(x = N_, y = csv_smc_T[y_col].values,yerr=csv_smc_T[y_std_col].values,label =f"{method_str}, T={T}, alpha={ess_alpha}",
                )

    




########################################################################
##################### print AMLS results ###############################
########################################################################
x_max=int(1.1e6)

img_idx=0
eps=0.15
ratio=0.1
T_range = [20,50,100,200,500,1000]
triplets=[('image_idx',img_idx,'='),('epsilon',eps,'='),('n_rep',50,'>='),
         (x_col,x_max,'<='),('ratio',ratio,'=')]
method_str='MLS-SMC'

for T in T_range:
    csv_ams_T = get_sel_df(df=csv_res_webb, cols=['N','T'],vals=[16,T],conds=['>=','='],
                    triplets=triplets)[:5].sort_values(by=x_col)
    l = len(csv_ams_T)
    print(l)
    N_ = csv_ams_T[x_col].values

    plt.errorbar(x = N_, y = csv_ams_T[y_col].values,yerr=csv_ams_T[y_std_col].values,label =f"{method_str}, T= {T}, survival rate={ratio}")


    
    
plt.legend(loc='lower center', mode=None, fontsize=15)

plt.xlabel(xlabel=x_col,fontsize=13)
axes=plt.gca()
axes.xaxis.set_ticks(np.linspace(start=x_min,stop=x_max,num=11))
plt.ylabel(ylabel=f'mean {log_str} proba est.',fontsize=15)

axes.xaxis.set_tick_params(labelsize=16)
axes.yaxis.set_tick_params(labelsize=16)
x_max=int(5e5)
x_min=5000
axes=plt.gca()

axes.xaxis.set_ticks(np.linspace(start=x_min,stop=x_max,num=11))
plt.xlim(right=x_max)
plt.xlim(left=x_min)

plt.legend(loc='lower right', mode=None, fontsize=16)

plt.xlabel(xlabel=x_col,fontsize=15)
axes=plt.gca()
axes.xaxis.set_ticks(np.linspace(start=x_min,stop=x_max,num=11))
plt.ylabel(ylabel=f'mean {log_str} proba est.',fontsize=15)

axes.xaxis.set_tick_params(labelsize=17)
axes.yaxis.set_tick_params(labelsize=17)
x_max=int(1.1e6)
x_min=0
axes=plt.gca()

axes.xaxis.set_ticks(np.arange(start=x_min,stop=900001,step=100000))
plt.xlim(right=x_max)
plt.xlim(left=x_min)
PLOTS_DIR=os.path.join(ROOT_DIR,'exp_2/')
fig_path=os.path.join(PLOTS_DIR,
                      f'mnist_{log_str}_eps_{float_to_file_float(eps)}_{method_str}.pdf')
print(f'Saving figure to:{fig_path}')
plt.savefig(fig_path)