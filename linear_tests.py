import argparse
import subprocess

from dev.utils import str2bool

class config:
    log_dir="./logs/linear_tests"
    n_rep='10'
    N='10,40'
    verbose=0
    min_rate=0.40
    T='1'
    rho='90'
    alpha='0.025'
    n_max='5000'
    tqdm_opt=True
    p_t='1e-15'
    d = 1024
    epsilon = 1
    save_config = True
    print_config=True
    update_aggr_res=True
    method="langevin_base"
    script_name="linear_test_"
    gaussian_latent='False'

parser=argparse.ArgumentParser()
parser.add_argument('--log_dir',default=config.log_dir)
parser.add_argument('--n_rep',type=int,default=config.n_rep)
parser.add_argument('--N',type=str,default=config.N)
parser.add_argument('--verbose',type=float,default=config.verbose)
parser.add_argument('--d',type=int,default=config.d)
parser.add_argument('--p_t',type=str,default=config.p_t)
parser.add_argument('--min_rate',type=str,default=config.min_rate)
parser.add_argument('--alpha',type=str,default=config.alpha)
parser.add_argument('--n_max',type=int,default=config.n_max)
parser.add_argument('--epsilon',type=float, default=config.epsilon)
parser.add_argument('--tqdm_opt',type=bool,default=config.tqdm_opt)
parser.add_argument('--T',type=str,default=config.T)
parser.add_argument('--save_config', type=bool, default=config.save_config)
parser.add_argument('--print_config',type=bool , default=config.print_config)
parser.add_argument('--update_aggr_res', type=bool,default=config.update_aggr_res)
parser.add_argument('--method',type=str,default=config.method)
parser.add_argument('--rho',type=str,default=config.rho)
parser.add_argument('--gaussian_latent',type=str, default=config.gaussian_latent)



args=parser.parse_args()

for k,v in vars(args).items():
    setattr(config, k, v)

# def str_to_type_list(in_string,out_type=int,split_char=','):
#     l = in_string.strip('[').strip(']').split(split_char)
#     return [out_type(e) for e in l]


# T_list=str_to_type_list(config.T,out_type=int)
# p_t_list=str_to_type_list(config.alpha, out_type=float)
# alpha_list=str_to_type_list(config.alphat, out_type= float)
# rho_list=str_to_type_list(config.rho,out_type=float)

gaussian_latent= True if config.gaussian_latent.lower() in ('true','yes','y') else False
assert type(gaussian_latent)==bool, "The conversion of string gaussian_latent failed"

T_list=(config.T).split(',')
p_t_list=(config.p_t).split(',')
alpha_list=(config.alpha).split(',')
N_list=(config.N).split(',')
rho_list=(config.rho).split(',')

if config.method in ("langevin_base","langevin_adapt"):
    script_name=config.script_name+config.method+".py"
else:
    raise RuntimeError("Input method must be chosen in ('langevin_base','langevin_adapt'")

for p in p_t_list:
    for t in T_list:
        for N in N_list: 
            if "langevin" in config.method:
                for a in alpha_list: 
                    if 'base' in config.method:
                        for r in rho_list:
                            cmd_list = f"python {script_name} --min_rate {config.min_rate} --epsilon {config.epsilon} --n_max {config.n_max} --T {t} --p_t {p} --alpha {a} --rho {r} --N {N} --n_rep {config.n_rep} --verbose {config.verbose} --d {config.d}".split(' ')
                            subprocess.run(cmd_list)
                    else:
                        cmd_list = f"python {script_name} --min_rate {config.min_rate} --epsilon {config.epsilon} --n_max {config.n_max} --T {t} --p_t {p} --alpha {a} --N {N} --n_rep {config.n_rep} --verbose {config.verbose} --d {config.d}".split(' ')
                        subprocess.run(cmd_list)
            else:
                cmd_list = f"python {script_name} --epsilon {config.epsilon} --p_t {p} --N {N} --n_rep {config.n_rep} --verbose {config.verbose} --d {config.d}".split(' ')
                subprocess.run(cmd_list)