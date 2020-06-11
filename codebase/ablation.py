import sys
import os
import numpy as np
from subprocess import call

def load_config(cfg_file):
    cfg = {}

    with open(cfg_file,'r') as f:
        for l in f:
            l = l.strip()
            if len(l)>0 and not l[0] == '#':
                vs = l.split('=')
                if len(vs)>0:
                    k,v = (vs[0], eval(vs[1]))
                    if not isinstance(v,list):
                        v = [v]
                    cfg[k] = v
    return cfg

def sample_config(configs):
    cfg_sample = {}
    for k in configs.keys():
        opts = configs[k]
        c = np.random.choice(len(opts),1)[0]
        cfg_sample[k] = opts[c]
    return cfg_sample

def cfg_string(cfg):
    ks = sorted(cfg.keys())
    cfg_str = ','.join(['%s:%s' % (k, str(cfg[k])) for k in ks])
    return cfg_str.lower()

def is_used_cfg(cfg, used_cfg_file):
    cfg_str = cfg_string(cfg)
    used_cfgs = read_used_cfgs(used_cfg_file)
    return cfg_str in used_cfgs

def read_used_cfgs(used_cfg_file):
    used_cfgs = set()
    with open(used_cfg_file, 'r') as f:
        for l in f:
            used_cfgs.add(l.strip())

    return used_cfgs

def save_used_cfg(cfg, used_cfg_file):
    with open(used_cfg_file, 'a') as f:
        cfg_str = cfg_string(cfg)
        f.write('%s\n' % cfg_str)

def run(cfg_file, num_runs, ablation_var1):
    configs = load_config(cfg_file)
    outdir = configs['outdir'][0]
    used_cfg_file = '%s/used_configs.txt' % outdir

    # list of possible values for the ablation variable
    ablation_values1 = configs[ablation_var1]
   

    if not os.path.isfile(used_cfg_file):
        f = open(used_cfg_file, 'w')
        f.close()
    
    for i in range(num_runs):
        cfg = sample_config(configs)
        for ablation_val1 in ablation_values1:
            cfg[ablation_var1] = ablation_val1
            if is_used_cfg(cfg, used_cfg_file):
                print ('Configuration used, skipping')
                continue

            save_used_cfg(cfg, used_cfg_file)

            print ('------------------------------')
            print ('Run %d of %d:' % (i+1, num_runs))
            print ('------------------------------')
            print ('\n'.join(['%s: %s' % (str(k), str(v)) for k,v in cfg.iteritems() if len(configs[k])>1]))

            flags = (' '.join('--%s %s' % (k,str(v)) for k,v in cfg.iteritems()))

            call('python cfr_net_train.py %s' % flags, shell=True)
            # with this, you get a fair ablation!
            # note: here "num_runs" is actually multiplied by number of possible values of ablation_var

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print ('Usage: python cfr_param_search.py <config file> <num runs> <ablation var>')
    else:
        run(sys.argv[1], int(sys.argv[2]),str(sys.argv[3]))