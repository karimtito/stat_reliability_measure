import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from dev.utils import get_sel_df
from home import ROOT_DIR
HOME_DIR=ROOT_DIR
csv_path=os.path.join(HOME_DIR,'logs/linear_gaussian_tests/aggr_res.csv')
csv_res=pd.read_csv(csv_path)
csv_res=csv_res[csv_res['n_rep']==200]
min_rate_smc=0.2
min_rate_ams=0.3
amls_filtr=(csv_res['min_rate']==min_rate_ams) & (csv_res['method']=='MLS_SMC')  & (csv_res['last_particle']==False)
smc_filtr=(csv_res['min_rate']==min_rate_smc) & ((csv_res['method']=='H_SMC') | 
                                                csv_res['method']=='MALA_SMC' | csv_res['method']=='RW_SMC')
csv_res_amls=csv_res[amls_filtr]
csv_res_smc=csv_res[smc_filtr]
plt.figure(figsize=(18,10))
plt.subplot(2,1,1)
min_rate=0.3
p_t = 1e-6
csv_res_pt=csv_res[csv_res['p_t']==p_t]
class config:
    T=20
    T_range=[]

if len(config.T_range)==0:
    T_range=[config.T]
x_max=int(1e6)
y_max=2
N_max=2048
x_min=0
alpha = 0.002
y_col='mean_rel_error'
x_col='mean_calls'

y_min=0.9*csv_res_pt[y_col].min()
#y_max=0.*csv_res_pt[y_col].max()





################################################################################
####################### Plot AMLS results ######################################
################################################################################
ratio=0.1

min_rate_ams=0.3
for T in T_range:


    csv_amls_T = get_sel_df(df=csv_res_amls, cols=[x_col,'T'],vals=[20,T],conds=['>=','='],
                    triplets=[('method','amls_pyt','='),('p_t',p_t,'='),('n_rep',100,'>='),
                    ('N',N_max,'<='),(x_col,x_min,'>='), ('min_rate',min_rate_ams,'='),
                             ('ratio',ratio,'=')]).sort_values(by=x_col)[:10]
    
    N_ = csv_amls_T[x_col].values
    print(len(N_))
    plt.plot(N_,csv_amls_T[y_col].values,
    label = f"MLS-SMC, T= {T}, survival rate={ratio}")



plt.legend(loc='upper center', mode=None)
plt.xlabel(xlabel=x_col)
axes=plt.gca()
#axes.xaxis.set_ticks(csv_amls_T[x_col].values)
plt.ylabel(ylabel=y_col)




    
################################################################################
####################### Plot HMC-SMC results ######################################
################################################################################
ess_alpha=0.95
only_duplicated=True
v_min_opt=True
min_rate=0.18
GV_opt=False

L=10
kernel='RW' if GV_opt else 'H'
if not GV_opt and L==1:
    kernel='MALA'
triplets=[('method','smc','cont'),('p_t',p_t,'='),('n_rep',100,'>='),
               ('L',L,'='),('only_duplicated',only_duplicated,'='),
                ('ess_alpha',ess_alpha,'='),('v_min_opt',v_min_opt,'='),('GV_opt',GV_opt,'='),
                 (x_col,x_max,'<=')]


for T in T_range:
    csv_smc_T = get_sel_df(df=csv_res_smc, cols=['N','T'],vals=[20,T],conds=['>=','='],
                    triplets=triplets).sort_values(by=x_col)
    csv_smc_T['MSE_rel']=csv_smc_T['MSE']/csv_smc_T['p_t']**2
    l = len(csv_smc_T)
    N_ = csv_smc_T[x_col].values
        
    plt.plot(N_,csv_smc_T[y_col].values,
    label = f"{kernel}-SMC, T= {T}, alpha={ess_alpha}")
    

################################################################################
####################### Plot MALA-SMC results ######################################
################################################################################
ess_alpha=0.95
only_duplicated=True
v_min_opt=True
min_rate=0.18
GV_opt=False

L=1
kernel='RW' if GV_opt else 'H'
if not GV_opt and L==1:
    kernel='MALA'
triplets=[('method','smc','cont'),('p_t',p_t,'='),('n_rep',100,'>='),
               ('L',L,'='),('only_duplicated',only_duplicated,'='),
                ('ess_alpha',ess_alpha,'='),('v_min_opt',v_min_opt,'='),('GV_opt',GV_opt,'='),
                 (x_col,x_max,'<=')]


for T in T_range:
    csv_smc_T = get_sel_df(df=csv_res_smc, cols=['N','T'],vals=[20,T],conds=['>=','='],
                    triplets=triplets).sort_values(by=x_col)
    csv_smc_T['MSE_rel']=csv_smc_T['MSE']/csv_smc_T['p_t']**2
    l = len(csv_smc_T)
    N_ = csv_smc_T[x_col].values
        
    plt.plot(N_,csv_smc_T[y_col].values,
    label =f"{kernel}-SMC, T= {T}, alpha={ess_alpha}")
    

    




################################################################################
####################### Plot RW-SMC results ######################################
################################################################################
ess_alpha=0.95
only_duplicated=True
v_min_opt=True
min_rate=0.2
GV_opt=True

L=1
triplets=[('method','smc','cont'),('p_t',p_t,'='),('n_rep',200,'='),
               ('L',L,'='),('only_duplicated',only_duplicated,'='),('min_rate',min_rate,'='),
                ('ess_alpha',ess_alpha,'='),('v_min_opt',v_min_opt,'='),('GV_opt',GV_opt,'='),
                 (x_col,x_max,'<=')]



kernel='RW' if GV_opt else 'H'
for T in T_range:
    csv_smc_T = get_sel_df(df=csv_res, cols=['N','T'],vals=[20,T],conds=['>=','='],
                    triplets=triplets).sort_values(by=x_col)
    csv_smc_T['MSE_rel']=csv_smc_T['MSE']/csv_smc_T['p_t']**2
    l = len(csv_smc_T)
    N_ = csv_smc_T[x_col].values
    print(len(N_))
        
    plt.plot(N_, csv_smc_T[y_col].values,
    label = f"{kernel}-SMC, T= {T}, alpha={ess_alpha}")

plt.legend(loc='upper right', mode=None,fontsize=17)
plt.xlabel(xlabel=x_col,fontsize=15)

plt.ylabel(ylabel=y_col,fontsize=16)
plt.ylim(top=1)
plt.ylim(bottom=0)
x_max=100000
x_min=20000
axes=plt.gca()
axes.xaxis.set_tick_params(labelsize=20)
axes.yaxis.set_tick_params(labelsize=22)
axes.xaxis.set_ticks(np.arange(x_min,x_max+1,step=20000))
plt.xlim(right=x_max)
plt.xlim(left=x_min)
    
plt.subplot(2,1,2)  
p_t = 1e-12
csv_res_pt=csv_res[csv_res['p_t']==p_t]
T=20
T_range=[T]
x_max=int(1e6)
y_max=2
N_max=2048
x_min=0
alpha = 0.002
y_col='mean_rel_error'
x_col='mean_calls'

y_min=0.9*csv_res_pt[y_col].min()
#y_max=0.*csv_res_pt[y_col].max()





################################################################################
####################### Plot AMLS results ######################################
################################################################################
ratio=0.1

min_rate_ams=0.3
for T in T_range:


    csv_amls_T = get_sel_df(df=csv_res_amls, cols=[x_col,'T'],vals=[20,T],conds=['>=','='],
                    triplets=[('method','amls_pyt','='),('p_t',p_t,'='),('n_rep',100,'>='),
                    ('N',N_max,'<='),(x_col,x_min,'>='), ('min_rate',min_rate_ams,'='),
                             ('ratio',ratio,'=')]).sort_values(by=x_col)[:10]
    
    N_ = csv_amls_T[x_col].values
    print(len(N_))
    plt.plot(N_,csv_amls_T[y_col].values,
    label = f"MLS-SMC, T= {T}, survival rate={ratio}")







    
################################################################################
####################### Plot HMC-SMC results ######################################
################################################################################
ess_alpha=0.95
only_duplicated=True
v_min_opt=True
min_rate=0.18
GV_opt=False

L=10
kernel='RW' if GV_opt else 'H'
if not GV_opt and L==1:
    kernel='MALA'
triplets=[('method','smc','cont'),('p_t',p_t,'='),('n_rep',100,'>='),
               ('L',L,'='),('only_duplicated',only_duplicated,'='),
                ('ess_alpha',ess_alpha,'='),('v_min_opt',v_min_opt,'='),('GV_opt',GV_opt,'='),
                 (x_col,x_max,'<=')]


for T in T_range:
    csv_smc_T = get_sel_df(df=csv_res_smc, cols=['N','T'],vals=[20,T],conds=['>=','='],
                    triplets=triplets).sort_values(by=x_col)
    csv_smc_T['MSE_rel']=csv_smc_T['MSE']/csv_smc_T['p_t']**2
    l = len(csv_smc_T)
    N_ = csv_smc_T[x_col].values
        
    plt.plot(N_,csv_smc_T[y_col].values,
    label = f"{kernel}-SMC, T= {T}, alpha={ess_alpha}")
    

################################################################################
####################### Plot MALA-SMC results ######################################
################################################################################
ess_alpha=0.95
only_duplicated=True
v_min_opt=True
min_rate=0.18
GV_opt=False

L=1
kernel='RW' if GV_opt else 'H'
if not GV_opt and L==1:
    kernel='MALA'
triplets=[('method','smc','cont'),('p_t',p_t,'='),('n_rep',100,'>='),
               ('L',L,'='),('only_duplicated',only_duplicated,'='),
                ('ess_alpha',ess_alpha,'='),('v_min_opt',v_min_opt,'='),('GV_opt',GV_opt,'='),
                 (x_col,x_max,'<=')]


for T in T_range:
    csv_smc_T = get_sel_df(df=csv_res_smc, cols=['N','T'],vals=[20,T],conds=['>=','='],
                    triplets=triplets).sort_values(by=x_col)
    csv_smc_T['MSE_rel']=csv_smc_T['MSE']/csv_smc_T['p_t']**2
    l = len(csv_smc_T)
    N_ = csv_smc_T[x_col].values
        
    plt.plot(N_,csv_smc_T[y_col].values,
    label = f"{kernel}-SMC, T= {T}, alpha={ess_alpha}")
    

    




################################################################################
####################### Plot RW-SMC results ######################################
################################################################################
ess_alpha=0.95
only_duplicated=True
v_min_opt=True
min_rate=0.2
GV_opt=True

L=1
triplets=[('method','smc','cont'),('p_t',p_t,'='),('n_rep',200,'='),
               ('L',L,'='),('only_duplicated',only_duplicated,'='),('min_rate',min_rate,'='),
                ('ess_alpha',ess_alpha,'='),('v_min_opt',v_min_opt,'='),('GV_opt',GV_opt,'='),
                 (x_col,x_max,'<=')]



kernel='RW' if GV_opt else 'H'
for T in T_range:
    csv_smc_T = get_sel_df(df=csv_res, cols=['N','T'],vals=[20,T],conds=['>=','='],
                    triplets=triplets).sort_values(by=x_col)
    csv_smc_T['MSE_rel']=csv_smc_T['MSE']/csv_smc_T['p_t']**2
    l = len(csv_smc_T)
    N_ = csv_smc_T[x_col].values
    print(len(N_))
        
    plt.plot(N_, csv_smc_T[y_col].values,
    label = f"{kernel}-SMC, T= {T}, alpha={ess_alpha}")

plt.legend(loc='upper right', mode=None,fontsize=18)
plt.xlabel(xlabel=x_col)
axes=plt.gca()
#axes.xaxis.set_ticks(csv_amls_T[x_col].values)
plt.ylabel(ylabel=y_col)

plt.xlabel(xlabel=x_col,fontsize=15)

plt.ylabel(ylabel=y_col,fontsize=16)
plt.ylim(top=2.)
plt.ylim(bottom=0)
x_max=150000
x_min=40000
axes=plt.gca()
axes.xaxis.set_tick_params(labelsize=20)
axes.yaxis.set_tick_params(labelsize=22)
axes.xaxis.set_ticks(np.arange(x_min,x_max+1,step=20000))
plt.xlim(right=x_max)
plt.xlim(left=x_min)
PLOTS_DIR=os.path.join(HOME_DIR,'exp_1/')
fig_path=os.path.join(PLOTS_DIR,
                      f'toy_model_{y_col}_T_{T}_min_{x_min}_max_{x_max}.pdf')
print(f'Saving figure to:{fig_path}')
plt.savefig(fig_path)