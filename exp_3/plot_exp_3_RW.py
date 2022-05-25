import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from dev.utils import float_to_file_float,path_to_stdlog10,path_to_meanlog10,get_sel_df
from home import ROOT_DIR
csv_path=os.path.join(ROOT_DIR,'logs/imagenet_tests/aggr_res.csv')
csv_res=pd.read_csv(csv_path)
csv_res['mean_log10_est']=csv_res['mean_log_est']/np.log(10)
csv_res['std_log10_est']=csv_res['std_log_est']/np.log(10)
T_range=[10,20,50,100,200,500]
e_range=[0.85,0.7,0.5]
N_range=[32,64,128,256,512,1024]
ratio_range=[0.1,0.6]
min_rate_smc=0.2
min_rate_ams=0.51
csv_res =csv_res[csv_res['n_rep']==50]
csv_res['mean_log10_est']=csv_res['mean_log_est']/np.log(10)
csv_res['std_log10_est']=csv_res['std_log_est']/np.log(10)
csv_res=csv_res[((csv_res['epsilon']==0.01)&(csv_res['model_name']=='torchvision_resnet18'))
                 
                 |((csv_res['epsilon']==0.13)&(csv_res['model_name']=='torchvision_mobilenet_v2'))
                
                    
                 ]

amls_filtr=(csv_res['method']=='webb_ams') 

smc_filtr=  (csv_res['method']=='smc_pyt_killing_adjusted') 
csv_res_webb=csv_res[amls_filtr]

csv_res_smc=csv_res[smc_filtr]

csv_res_webb.loc[:,'mean_log10_est'] = csv_res_webb.loc[:,'est_path'].apply(path_to_meanlog10)
csv_res_webb.loc[:,'std_log10_est'] = csv_res_webb.loc[:,'est_path'].apply(path_to_stdlog10)

plt.figure(figsize=(16,9))
log10=True
y_col='mean_log10_est' if log10 else 'mean_log10_est' 
std_col='std_log10_est' if log10 else 'std_log10_est'
log_str='log10' if log10 else 'ln'
x_col='mean_calls' 
x_max=int(5e5)
x_min=0
y_max=1.2*csv_res[y_col].max()
y_min=0.8*csv_res[y_col].min()
img_idx=0
eps=0.01



################################################################################
####################### Plot HMC-SMC results ######################################
################################################################################


ess_alpha=0.85
model_name='torchvision_mobilenet_v2'
only_duplicated=True
GV_opt=False

L=5
kernel='RW' if GV_opt else 'H'
if not GV_opt and L==1:
    kernel='MALA'
method_str=f"{kernel}-SMC"
T_range = [10,]
triplets=[('image_idx',img_idx,'='),('n_rep',50,'>='),
          ('only_duplicated',only_duplicated,'='),('model_name',model_name,'='),
         (x_col,x_max,'<='),('L',L,'='),('ess_alpha',ess_alpha,'='),('GV_opt',GV_opt,'=')]

for T in T_range:
    csv_smc_T = get_sel_df(df=csv_res_smc, cols=['N','T'],vals=[20,T],conds=['>=','='],
                    triplets=triplets).sort_values(by=x_col)
    l = len(csv_smc_T)
    print(l)
    N_ = csv_smc_T[x_col].values

    plt.errorbar(x = N_, y = csv_smc_T[y_col].values,yerr=csv_smc_T[std_col].values,label =f"{method_str}, T= {T}, alpha={ess_alpha}")
    
################################################################################
####################### Plot MALA-SMC results ######################################
################################################################################


ess_alpha=0.85
model_name='torchvision_mobilenet_v2'
only_duplicated=True
GV_opt=False

L=1
kernel='RW' if GV_opt else 'H'
if not GV_opt and L==1:
    kernel='MALA'
method_str=f"{kernel}-SMC"
T_range = [10]
triplets=[('image_idx',img_idx,'='),('n_rep',50,'>='),
          ('only_duplicated',only_duplicated,'='),('model_name',model_name,'='),
         (x_col,x_max,'<='),('L',L,'='),('ess_alpha',ess_alpha,'='),('GV_opt',GV_opt,'=')]

for T in T_range:
    csv_smc_T = get_sel_df(df=csv_res_smc, cols=['N','T'],vals=[20,T],conds=['>=','='],
                    triplets=triplets).sort_values(by=x_col)
    l = len(csv_smc_T)
    print(l)
    N_ = csv_smc_T[x_col].values

    plt.errorbar(x = N_, y = csv_smc_T[y_col].values,yerr=csv_smc_T[std_col].values,label =f"{method_str}, T= {T}, alpha={ess_alpha}")
    
################################################################################
####################### Plot RW-SMC results ######################################
################################################################################


ess_alpha=0.7
model_name='torchvision_mobilenet_v2'
only_duplicated=True
GV_opt=True

L=1
kernel='RW' if GV_opt else 'H'
if not GV_opt and L==1:
    kernel='MALA'
method_str=f"{kernel}-SMC"
T_range = [10,20,50]
triplets=[('image_idx',img_idx,'='),('n_rep',50,'>='),
          ('only_duplicated',only_duplicated,'='),('model_name',model_name,'='),
         (x_col,x_max,'<='),('L',L,'='),('ess_alpha',ess_alpha,'='),('GV_opt',GV_opt,'=')]

for T in T_range:
    csv_smc_T = get_sel_df(df=csv_res_smc, cols=['N','T'],vals=[20,T],conds=['>=','='],
                    triplets=triplets).sort_values(by=x_col)
    l = len(csv_smc_T)
    if T==10:
        csv_smc_T=csv_smc_T.iloc[1:]
    print(l)
    N_ = csv_smc_T[x_col].values

    plt.errorbar(x = N_, y = csv_smc_T[y_col].values,yerr=csv_smc_T[std_col].values,label =f"{method_str}, T= {T}, alpha={ess_alpha}")
    




plt.xlabel(xlabel=x_col)
axes=plt.gca()
axes.xaxis.set_ticks(np.linspace(start=x_min,stop=x_max,num=11))
plt.ylabel(ylabel=y_col)


plt.legend(loc='lower right', mode=None, fontsize=18)

plt.xlabel(xlabel=x_col,fontsize=13)
axes=plt.gca()
axes.xaxis.set_ticks(np.linspace(start=x_min,stop=x_max,num=11))
plt.ylabel(ylabel=f'mean {log_str} proba est.',fontsize=15)

axes.xaxis.set_tick_params(labelsize=16)
axes.yaxis.set_tick_params(labelsize=18)
x_max=int(8e5)
x_min=0
axes=plt.gca()

axes.xaxis.set_ticks(np.linspace(start=x_min,stop=x_max,num=11))
plt.xlim(right=480000)
plt.xlim(left=x_min)
PLOTS_DIR=os.path.join(ROOT_DIR,'exp_3/')
fig_path=os.path.join(PLOTS_DIR,
                      f'imagenet_{log_str}_eps_{float_to_file_float(eps)}_{method_str}.pdf')
print(f'Saving figure to:{fig_path}')
plt.savefig(fig_path)