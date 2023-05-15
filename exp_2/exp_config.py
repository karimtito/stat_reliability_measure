

class exp_config:
    dataset='mnist'
    model_arch=None  
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

    model_dir=None 
    noise_dist='uniform'
    d=None
    verbose=0
    log_dir=None
    aggr_res_path = None
    update_aggr_res=True
