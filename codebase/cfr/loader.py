import os
import numpy as np

from logger import Logger as Log

def load_result_file(file,use_e=False):
    arr = np.load(file)
    if(use_e):
        D = dict([(k, arr[k]) for k in arr.keys()])
    else:
        D = dict([(k, arr[k]) for k in arr.keys() if k!='e'])

    return D

def load_config(cfgfile):
    """ Parses a configuration file """

    cfgf = open(cfgfile,'r')
    cfg = {}
    for l in cfgf:
        ps = [p.strip() for p in l.split(':')]
        if len(ps)==2:
            try:
                cfg[ps[0]] = float(ps[1])
            except ValueError:
                cfg[ps[0]] = ps[1]
                if cfg[ps[0]] == 'False':
                    cfg[ps[0]] = False
                elif cfg[ps[0]] == 'True':
                    cfg[ps[0]] = True
    cfgf.close()
    return cfg

def load_single_result(result_dir, filters=None):
    # print(result_dir)
    if Log.VERBOSE:
        print 'Loading %s...' % result_dir

    config_path = '%s/config.txt' % result_dir
    # print (config_path)
    has_config = os.path.isfile(config_path)
    if not has_config:
        print 'WARNING: Could not find config.txt for %s. Skipping.' % os.path.basename(result_dir)
        config = None
    else:
        config = load_config(config_path)

    if filters:
        for k, v in filters.items():
            if k in config:
                if config[k] != v:
                    return None
            else:
                print ('{} not a config key.'.format(k))

    
    train_path = '%s/result.npz' % result_dir
    test_path = '%s/result.test.npz' % result_dir

    has_test = os.path.isfile(test_path)

    try:
        train_results = load_result_file(train_path,use_e=True)
    except:
        print('WARNING: Couldnt load result file. Skipping')
        return None
    
    n_rep = np.max([config['repetitions'], config['experiments']])

    if len(train_results['pred'].shape) < 4 or train_results['pred'].shape[2] < n_rep:
        print 'WARNING: Experiment %s appears not to have finished. Skipping.' % result_dir
        return None

    if has_test:
        test_results = load_result_file(test_path,use_e=True)
    else:
        test_results = None

    return {'train': train_results, 'test': test_results, 'config': config}

def load_results(output_dir, filters=None):

    if Log.VERBOSE:
        print 'Loading results from %s...' % output_dir

    ''' Detect results structure '''
    # Single result
    if os.path.isfile('%s/results.npz' % output_dir):
        #@TODO: Implement
        pass

    # Multiple results
    files = ['%s/%s' % (output_dir, f) for f in os.listdir(output_dir)]
    # print(files)
    exp_dirs = [f for f in files if os.path.isdir(f)
                    if os.path.isfile('%s/result.npz' % f)]

    if Log.VERBOSE:
        print 'Found %d experiment configurations.' % len(exp_dirs)

    # Load each result folder
    results = []
    for dir in exp_dirs:
        dir_result = load_single_result(dir, filters)
        if dir_result is not None:
            results.append(dir_result)

    return results

def load_data(datapath):
    """ Load dataset """
    arr = np.load(datapath)
    xs = arr['x']

    HAVE_TRUTH = False
    SPARSE = False

    if len(xs.shape)==1:
        SPARSE = True

    ts = arr['t']
    yfs = arr['yf']
    try:
        es = arr['e']
    except:
        es = None
    try:
        ate = np.mean(arr['ate'])
    except:
        ate = None
    try:
        ymul = arr['ymul'][0,0]
        yadd = arr['yadd'][0,0]
    except:
        ymul = 1
        yadd = 0
    try:
        ycfs = arr['ycf']
        mu0s = arr['mu0']
        mu1s = arr['mu1']
        HAVE_TRUTH = True
    except:
        print 'Couldn\'t find ground truth. Proceeding...'
        ycfs = None; mu0s = None; mu1s = None
    
    if ate:
        data = {'x':xs, 't':ts, 'e':es, 'yf':yfs, 'ycf':ycfs, \
            'mu0':mu0s, 'mu1':mu1s, 'ate':ate, 'YMUL': ymul, \
            'YADD': yadd, 'ATE': ate.tolist(), 'HAVE_TRUTH': HAVE_TRUTH, \
            'SPARSE': SPARSE}
    else:
        data = {'x':xs, 't':ts, 'e':es, 'yf':yfs, 'ycf':ycfs, \
            'mu0':mu0s, 'mu1':mu1s, 'ate':ate, 'YMUL': ymul, \
            'YADD': yadd, 'ATE': ate, 'HAVE_TRUTH': HAVE_TRUTH, \
            'SPARSE': SPARSE}

    return data
