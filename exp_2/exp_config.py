from stat_reliability_measure.home import ROOT_DIR

class exp_config:
    dataset='mnist'
    data_dir=ROOT_DIR+"/data"
    model_arch=None  
    model_dir=None 
    epsilons = [0.15]
    save_config=False 
    print_config=True
    eps_max=0.3
    eps_min=0.2
    eps_num=5

    model_path=None
    export_to_onnx=False
    use_attack=False
    attack='PGD'
    lirpa_bounds=False
    download=True
    train_model=False
    train_model_epochs=10
    input_start=0
    input_stop=None

    gaussian_latent=True

    
    noise_dist='uniform'
    d=None
    verbose=0
    log_dir=None
    aggr_res_path = None
    update_aggr_res=True
    batch_opt=True
    track_finish=False
    lirpa_cert=False
    robust_model=False
    robust_eps=0.1
    load_batch_size=100 
    nb_epochs= 15
    adversarial_every=1
    

