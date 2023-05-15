from stat_reliability_measure.home import ROOT_DIR
import numpy as np
class base_config:
    method_name="MLS_SMC"
    dataset="mnist"
    log_dir=ROOT_DIR+"/logs/mnist_tests"
    model_dir=ROOT_DIR+"/models/mnist"
    model_arch="dnn2"
    epsilons = [0.15]
    N_range=[32,64,128,256,512,1024]
    N_range_alt=[] 
    T_range=[1,10,20,50,100,200,500,1000]
    ratio_range=[0.1]
    n_rep=100
    a=0
    verbose=0
    min_rate=0.51
    clip_s=True
    s_min=8e-3
    s_max=3
    n_max=2000
    x_min=0
    x_max=1
    x_mean=0
    x_std=1
    allow_zero_est=True
    N=40
    T=1
    ratio=0.6
    s=1
    s_range = [1.]
    track_accept=False
    d = 784
    epsilon = 0.1
    n_max=2000
    tqdm_opt=True
    eps_max=0.3
    eps_min=0.1 
    eps_num=5
    allow_zero_est=True
    save_config=True
    print_config=True
    update_aggr_res=True
    aggr_res_path=None
    gaussian_latent=True
    project_kernel=True
    allow_multi_gpu=True
    input_start=0
    input_stop=None
    g_target=None
    track_gpu=False
    track_cpu=False
    gpu_name=None
    cpu_name=None
    cores_number=None
    device=None
    torch_seed=0
    np_seed=0
    tf_seed=None
    model_path=None
    export_to_onnx=False
    use_attack=False
    attack='PGD'
    lirpa_bounds=False
    download=True
    train_model=False
    noise_dist='uniform'
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
    force_train=False
    last_particle=False
    repeat_exp=True

def update_config(config):
    if len(config.epsilons)==0:
        log_eps=np.linspace(start=np.log(config.eps_min),stop=np.log(config.eps_max),num=config.eps_num)
        config.epsilons=np.exp(log_eps)

    if config.input_stop is None:
        config.input_stop=config.input_start+1
    if len(config.N_range)==0:
        config.N_range=[config.N]
    nb_runs*=len(config.N_range)
    if len(config.T_range)==0:
        config.T_range=[config.T]
    nb_runs*=len(config.T_range)
    if len(config.ratio_range)==0:
        config.ratio_range=[config.ratio]
    nb_runs*=len(config.ratio_range)
    if len(config.s_range)==0:
        config.s_range=[config.s]
    nb_runs*=len(config.s_range)
    
    return config