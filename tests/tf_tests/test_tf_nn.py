import scipy.stats as stat
import numpy as np
import eagerpy as ep
from tqdm import tqdm
from time import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.layers import Conv2D
import GPUtil
import foolbox as fb
import cpuinfo
import pandas as pd
from stat_reliability_measure.dev.utils import str2bool,str2floatList,str2intList,float_to_file_float,dichotomic_search,datasets_dims,datasets_num_c
from scipy.special import betainc
from importlib import reload
from stat_reliability_measure.home import ROOT_DIR
from stat_reliability_measure.dev.tf_utils import dic_in_shape_tf
from stat_reliability_measure.dev.tf_arch import CNN_custom_tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

import stat_reliability_measure.dev.smc.smc_ep as smc_ep
import stat_reliability_measure.dev.ep_utils as e_u
import stat_reliability_measure.dev.smc.smc_pyt as smc_pyt
import stat_reliability_measure.dev.tf_utils as tf_u
import stat_reliability_measure.dev.utils as u
import stat_reliability_measure.dev.tf_arch as t_a

reload(smc_ep)
reload(tf_u)
reload(e_u)
reload(u)
reload(t_a)




method_name="smc_ep"

class config:
    dataset='mnist'
    N=20
    N_range=[]
    T=10
    T_range=[]
    L=5
    L_range=[]
    min_rate=0.2
    
    alpha=0.2
    alpha_range=[]
    ess_alpha=0.875
    e_range=[]
   
    n_rep=10
    
    save_config=False 
    print_config=True
    
    x_min=0
    x_max=1
    x_mean=0
    x_std=1

    epsilons = None
    eps_max=0.1
    eps_min=0.01
    eps_num=5
    model_arch='CNN_custom'
    model_path=None
    export_to_onnx=False
    use_attack=False
    attack='PGD'
    lirpa_bounds=False
    download=True
    train_model=False
    
    
    noise_dist='uniform'
    d=None
    verbose=0
    log_dir=None
    aggr_res_path = None
    update_aggr_res=False
    sigma=1
    v1_kernel=True
    torch_seed=None
    gpu_name=None
    cpu_name=None
    cores_number=None
    track_gpu=False
    track_cpu=False
    
    n_max=10000 
    allow_multi_gpu=True
    tqdm_opt=True
    allow_zero_est=True
    track_accept=True
    track_calls=False
    mh_opt=False
    adapt_dt=False
    adapt_dt_mcmc=False
    target_accept=0.574
    accept_spread=0.1
    dt_decay=0.999
    dt_gain=None
    dt_min=1e-5
    dt_max=0.7
    v_min_opt=True
    ess_opt=False
    only_duplicated=True
    np_seed=None
    lambda_0=0.5
    test2=False

    s_opt=False
    s=1
    clip_s=True
    s_min=1e-3
    s_max=3
    s_decay=0.95
    s_gain=1.0001

    track_dt=False
    mult_last=True
    linear=True

    track_ess=True
    track_beta=True
    track_dt=True
    track_v_means=True
    track_ratios=False

    kappa_opt=True

    adapt_func='ESS'
    M_opt = False
    adapt_step=True
    FT=True
    sig_dt=0.02

    batch_opt=True
    track_finish=False
    lirpa_cert=False
    robust_model=False
    robust_eps=0.1
    load_batch_size=100 
    nb_epochs= 10
    adversarial_every=1
    data_dir=ROOT_DIR+"/data"
    p_ref_compute=False
    input_start=0
    input_stop=None

    gaussian_latent=True

    model_dir=None 
    L_min=1
    GK_opt=False
    GV_opt=False
    g_target=0.8
    skip_mh=False
    force_train=False
    killing=True


if config.model_dir is None:
    config.model_dir=os.path.join(ROOT_DIR+"/models/",config.dataset)
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)

config.d = u.datasets_dims[config.dataset]
color_dataset=config.dataset in ('cifar10','cifar100','imagenet') 
#assert config.adapt_func.lower() in smc_pyt.supported_beta_adapt.keys(),f"select adaptive function in {smc_pyt.supported_beta_adapt.keys}"
#adapt_func=smc_pyt.supported_beta_adapt[config.adapt_func.lower()]

if config.adapt_func.lower()=='simp_ess':
    adapt_func = lambda beta,v : smc_pyt.nextBetaSimpESS(beta_old=beta,v=v,lambda_0=config.lambda_0,max_beta=1e6)
elif config.adapt_func.lower()=='simp':
    adapt_func = lambda beta,v: smc_pyt.SimpAdaptBetaPyt(beta,v,config.g_target,v_min_opt=config.v_min_opt)
prblm_str=config.dataset




if len(config.e_range)==0:
    config.e_range= [config.ess_alpha]

if config.input_stop is None:
    config.input_stop=config.input_start+1
else:
    assert config.input_start<config.input_stop,"/!\ input start must be strictly lower than input stop"
if len(config.N_range)==0:
    config.N_range= [config.N]

if config.noise_dist is not None:
    config.noise_dist=config.noise_dist.lower()

if config.noise_dist not in ['uniform','gaussian']:
    raise NotImplementedError("Only uniform and Gaussian distributions are implemented.")

if len(config.T_range)==0:
    config.T_range= [config.T]

if len(config.L_range)==0:
    config.L_range= [config.L]
if len(config.alpha_range)==0:
    config.alpha_range= [config.alpha]







if config.np_seed is None:
    config.np_seed=int(time())




if config.track_gpu:
    gpus=GPUtil.getGPUs()
    if len(gpus)>1:
        print("Multi gpus detected, only the first GPU will be tracked.")
    config.gpu_name=gpus[0].name

if config.track_cpu:
    config.cpu_name=cpuinfo.get_cpu_info()[[key for key in cpuinfo.get_cpu_info().keys() if 'brand' in key][0]]
    config.cores_number=os.cpu_count()



d=config.d
#epsilon=config.epsilon

if config.log_dir is None:
    config.log_dir=os.path.join(ROOT_DIR+'/logs',config.dataset+'_tests')
if not os.path.exists(ROOT_DIR+'/logs'):
    os.mkdir(ROOT_DIR+'/logs')  
if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)

results_path=os.path.join(config.log_dir,'results.csv')
if os.path.exists(results_path):
    results_g=pd.read_csv(results_path)
else:
    results_g=pd.DataFrame(columns=['mean_est','mean_time','mean_err','stdtime','std_est','T','N','rho','alpha','n_rep','min_rate','method'])
    results_g.to_csv(results_path,index=False)
raw_logs = os.path.join(config.log_dir,'raw_logs/')
if not os.path.exists(raw_logs):
    os.mkdir(raw_logs)
raw_logs_path=os.path.join(config.log_dir,'raw_logs/'+method_name)
if not os.path.exists(raw_logs_path):
    os.mkdir(raw_logs_path)

loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
log_name=method_name+'_'+'_'+loc_time
log_name=method_name+'_'+'_'+loc_time
exp_log_path=os.path.join(raw_logs_path,log_name)
if os.path.exists(path=exp_log_path):
    exp_log_path = exp_log_path+'_'+str(np.random.randint(low=0,high=9))
os.mkdir(path=exp_log_path)

# if config.aggr_res_path is None:
#     aggr_res_path=os.path.join(config.log_dir,'agg_res.csv')
# else:
#     aggr_res_path=config.aggr_res_path

if config.dt_gain is None:
    config.dt_gain=1/config.dt_decay



if config.epsilons is None:
    log_min,log_max=np.log(config.eps_min),np.log(config.eps_max)
    log_line=np.linspace(start=log_min,stop=log_max,num=config.eps_num)
    config.epsilons=np.exp(log_line)

param_ranges = [config.N_range,config.T_range,config.L_range,config.e_range,config.alpha_range]
param_lens=np.array([len(l) for l in param_ranges])
nb_runs= np.prod(param_lens)

mh_str="adjusted" 
method=method_name
save_every = 1
x_min=0
x_max=1
#adapt_func= smc_pyt.ESSAdaptBetaPyt if config.ess_opt else smc_pyt.SimpAdaptBetaPyt
num_classes=u.datasets_num_c[config.dataset.lower()]
print(f"Running reliability experiments on architecture {config.model_arch} trained on {config.dataset}.")
print(f"Testing uniform noise pertubatin with epsilon in {config.epsilons}")




(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)



def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)



model_name = config.model_arch
#if model_name.lower()=='cnn_custom':
#    model = t_a.CNN_custom_tf()
#else:
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(10)
])



model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)




history =model.fit(
    ds_train, 
    epochs=6,
    validation_data=ds_test
)


model.summary()


for X_test,y_test in ds_test.__iter__():
    print(X_test.shape)
    break
log_pred = model.predict(X_test)
y_pred = tf.argmax(log_pred,axis=1)
correct = tf.equal(y_pred,y_test)
X_correct, y_correct= X_test[correct], y_test[correct]


del X_test, y_test


inp_indices=np.arange(start=config.input_start,stop=config.input_stop)
normal_dist=tfp.distributions.Normal(loc=0, scale=1.)
run_nb=0
iterator= tqdm(range(config.n_rep))
exp_res=[]
clip_min=0
clip_max=1



for l in inp_indices:
    x_0,y_0 = X_correct[l],tf.cast(y_correct[l],dtype=tf.int32) 

   
    for idx in range(len(config.epsilons)):
        
        
        epsilon = config.epsilons[idx]
        p_l,p_u=None,None
        if config.lirpa_bounds:
            from stat_reliability_measure.dev.lirpa_utils import get_lirpa_bounds
            # Step 2: define perturbation. Here we use a Linf perturbation on input image.
            p_l,p_u=get_lirpa_bounds(x_0=x_0,y_0=y_0,model=model,epsilon=epsilon,
            num_classes=num_classes,noise_dist=config.noise_dist,a=config.a,device=config.device)
            p_l,p_u=p_l.item(),p_u.item()
        lirpa_safe=None
        if config.lirpa_cert:
            assert config.noise_dist.lower()=='uniform',"Formal certification only makes sense for uniform distributions"
            from stat_reliability_measure.dev.lirpa_utils import get_lirpa_cert
            lirpa_safe=get_lirpa_cert(x_0=x_0,y_0=y_0,model=model,epsilon=epsilon,
            num_classes=num_classes,device=config.device)

        
        
        if config.gaussian_latent:
            gen = lambda N: tf.random.normal(shape=(N,d))
        else:
            gen= lambda N: (2*tf.random.uniform(shape=(N,d))-1)
            
        low=tf.maximum(x_0-epsilon, tf.constant([float(x_min)]))
        low_ep = ep.maximum(ep.astensor(x_0-epsilon), x_min)
        high_ep = ep.minimum(ep.astensor(x_0+epsilon), x_max)
        high=tf.minimum(x_0+epsilon, tf.constant([float(x_max)]))  
        model_shape=dic_in_shape_tf['mnist']
        V_ep = lambda X: e_u.V_ep(X, model=model,low=ep.astensor(low), high = ep.astensor(high),target_class=y_0, input_shape = model_shape,gaussian_latent=config.gaussian_latent)
    
        gradV_ep = lambda X: e_u.gradV_ep(X,model=model,input_shape=model_shape, low=ep.astensor(low), high = ep.astensor(high), target_class= y_0,  gaussian_latent=config.gaussian_latent)
        assert tf.math.reduce_all(low_ep.raw==low), "/!\ lower bounds are not the same"
        assert tf.math.reduce_all(high_ep.raw==high), "/!\ higher bounds are not the same"


del X_correct,y_correct


for ess_t in config.e_range:
            if config.adapt_func.lower()=='ess':
                adapt_func = lambda beta,v : smc_ep.nextBetaESS(beta_old=beta,v=v,ess_alpha=ess_t,max_beta=1e6)
            for T in config.T_range:
                for L in config.L_range:
                    for alpha in config.alpha_range:       
                        for N in config.N_range:
                            loc_time= datetime.today().isoformat().split('.')[0].replace('-','_').replace(':','_')
                            log_name=method_name+'_'+'_'+loc_time
                            log_name=method_name+f'_N_{N}_T_{T}_L_{L}_a_{float_to_file_float(alpha)}_ess_{float_to_file_float(ess_t)}'+'_'+loc_time.split('_')[0]
                            log_path=os.path.join(exp_log_path,log_name)
                            if os.path.exists(log_path):
                                log_path = log_path + '_'+str(np.random.randint(low=0,high =10))
                            
                            
                            os.mkdir(path=log_path)
                            run_nb+=1
                            print(f'Run {run_nb}/{nb_runs}')
                            times=[]
                            ests = []
                            calls=[]
                            finished_flags=[]
                            iterator= tqdm(range(config.n_rep)) if config.tqdm_opt else range(config.n_rep)
                            print(f"Starting {method} simulations with model:{model_name} img_idx:{l},eps={epsilon},ess_t:{ess_t},T:{T},alpha:{alpha},N:{N},L:{L}")
                            for i in iterator:
                                t=time()
                                p_est,res_dict,=smc_ep.SamplerSMC(gen=gen,V= V_ep,gradV=gradV_ep,adapt_func=adapt_func,min_rate=config.min_rate,N=N,T=T,L=L,
                                alpha=alpha,n_max=config.n_max,L_min=config.L_min,
                                verbose=config.verbose, track_accept=config.track_accept,track_beta=config.track_beta,track_v_means=config.track_v_means,
                                track_ratios=config.track_ratios,track_ess=config.track_ess,kappa_opt=config.kappa_opt
                                ,gaussian =True,accept_spread=config.accept_spread, 
                                adapt_dt=config.adapt_dt, dt_decay=config.dt_decay,
                                dt_gain=config.dt_gain,dt_min=config.dt_min,dt_max=config.dt_max,
                                v_min_opt=config.v_min_opt,
                                track_dt=config.track_dt,M_opt=config.M_opt,adapt_step=config.adapt_step,FT=config.FT,
                                sig_dt=config.sig_dt, skip_mh=config.skip_mh,GV_opt=config.GV_opt
                                )
                                t1=time()-t
                                if config.verbose>=2:
                                    print(p_est)
                                #finish_flag=res_dict['finished']
                                
                                if config.track_accept:
                                    accept_rates_mcmc=res_dict['accept_rates_mcmc']
                                    np.savetxt(fname=os.path.join(log_path,f'accept_rates_mcmc_{i}.txt')
                                    ,X=accept_rates_mcmc,)
                                    x_T=np.arange(len(accept_rates_mcmc))
                                    plt.plot(x_T,accept_rates_mcmc)
                                    plt.savefig(os.path.join(log_path,f'accept_rates_mcmc_{i}.png'))
                                    plt.close()
                                    

                                if config.track_dt:
                                    dts=res_dict['dts']
                                    np.savetxt(fname=os.path.join(log_path,f'dts_{i}.txt')
                                    ,X=dts)
                                    x_T=np.arange(len(dts))
                                    plt.plot(x_T,dts)
                                    plt.savefig(os.path.join(log_path,f'dts_{i}.png'))
                                    plt.close()
                                
                                
                                times.append(t1)
                                ests.append(p_est)
                                calls.append(res_dict['calls'])
                            times=np.array(times)
                            ests = np.array(ests)
                            calls=np.array(calls)
                        
                            mean_calls=calls.mean()
                            std_est=ests.std()
                            mean_est=ests.mean()
                            std_rel=std_est/mean_est
                            std_rel_adj=std_rel*mean_calls
                            print(f"Native PyTorch peformance")
                            print(f"mean est:{ests.mean()}, std est:{ests.std()}")
                            print(f"mean calls:{calls.mean()}")
                            print(f"std. rel.:{std_rel}")
                            print(f"std. rel. adj.:{std_rel*mean_calls}")
                            print(f"mean time:{times.mean()}, std. time:{times.std()}")


