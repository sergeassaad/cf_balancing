import tensorflow as tf
import numpy as np

SQRT_CONST = 1e-10
SMALL = 1e-7
FLAGS = tf.app.flags.FLAGS

def validation_split(D_exp, val_fraction):
    """ Construct a train/validation split """
    n = D_exp['x'].shape[0]

    if val_fraction > 0:
        n_valid = int(val_fraction*n)
        n_train = n-n_valid
        I = np.random.permutation(range(0,n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = range(n)
        I_valid = []

    return I_train, I_valid

def log(logfile,str):
    """ Log a string in a file """
    with open(logfile,'a') as f:
        f.write(str+'\n')
    print str

def save_config(fname):
    """ Save configuration """
    flagdict = tf.app.flags.FLAGS.flag_values_dict()
    s = '\n'.join(['%s: %s' % (k,str(flagdict[k])) for k in sorted(flagdict.keys())])
    f = open(fname,'w')
    f.write(s)
    f.close()

def load_data(fname):
    """ Load data set """
    if fname[-3:] == 'npz':
        data_in = np.load(fname)
        data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf']}
        try:
            data['ycf'] = data_in['ycf']
        except:
            data['ycf'] = None
    else:
        if FLAGS.sparse>0:
            data_in = np.loadtxt(open(fname+'.y',"rb"),delimiter=",")
            x = load_sparse(fname+'.x')
        else:
            data_in = np.loadtxt(open(fname,"rb"),delimiter=",")
            x = data_in[:,5:]

        data['x'] = x
        data['t'] = data_in[:,0:1]
        data['yf'] = data_in[:,1:2]
        data['ycf'] = data_in[:,2:3]

    data['HAVE_TRUTH'] = not data['ycf'] is None

    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]

    return data

def load_sparse(fname):
    """ Load sparse data set """
    E = np.loadtxt(open(fname,"rb"),delimiter=",")
    H = E[0,:]
    n = int(H[0])
    d = int(H[1])
    E = E[1:,:]
    S = sparse.coo_matrix((E[:,2],(E[:,0]-1,E[:,1]-1)),shape=(n,d))
    S = S.todense()

    return S

def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    x_inf = tf.cast(tf.is_inf(x), dtype=tf.float32)
    x_corrected = (1 - x_inf)*x + x_inf*tf.reduce_max(x+tf.constant(1e7), keepdims=True)
    return tf.sqrt(tf.clip_by_value(x_corrected, lbound, np.inf)) #tf.sqrt(tf.clip_by_value(x, lbound, 1e7)) #tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

def lindisc(X,p,t,weights=None):
    ''' Linear MMD '''	

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    if weights is not None:
	Wc = tf.gather(weights,ic)/(tf.reduce_sum(tf.gather(weights,ic))+tf.constant(1e-7))
	Wt = tf.gather(weights,it)/(tf.reduce_sum(tf.gather(weights,it))+tf.constant(1e-7))

    if weights is not None:
	mean_control = tf.reduce_sum(tf.tile(tf.reshape(Wc, shape=(tf.shape(Xc)[0], 1)), multiples=(1, tf.shape(Xc)[1]))*Xc, axis=0)
	mean_treated = tf.reduce_sum(tf.tile(tf.reshape(Wt, shape=(tf.shape(Xt)[0], 1)), multiples=(1, tf.shape(Xt)[1]))*Xt, axis=0)
    else:
	mean_control = tf.reduce_mean(Xc,reduction_indices=0)
	mean_treated = tf.reduce_mean(Xt,reduction_indices=0)

    c = tf.square(2*p-1)*0.25
    f = tf.sign(p-0.5)

    mmd = tf.reduce_sum(tf.square(p*mean_treated - (1-p)*mean_control))
    mmd = f*(p-0.5) + safe_sqrt(c + mmd)

    return mmd

def diff_list(list1,list2):
    diff_list = []
    for var in list1:
        if var not in list2:
            diff_list.append(var)
    return diff_list
    
def mmd2_lin(X,t,p,weights=None):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    if weights is not None:
        Wc = tf.gather(weights,ic)/(tf.reduce_sum(tf.gather(weights,ic))+tf.constant(1e-7))
        Wt = tf.gather(weights,it)/(tf.reduce_sum(tf.gather(weights,it))+tf.constant(1e-7))

    if weights is not None:
        mean_control = tf.reduce_sum(tf.tile(tf.reshape(Wc, shape=(tf.shape(Xc)[0], 1)), multiples=(1, tf.shape(Xc)[1]))*Xc, axis=0)
        mean_treated = tf.reduce_sum(tf.tile(tf.reshape(Wt, shape=(tf.shape(Xt)[0], 1)), multiples=(1, tf.shape(Xt)[1]))*Xt, axis=0)
    else:
        mean_control = tf.reduce_mean(Xc,reduction_indices=0)
        mean_treated = tf.reduce_mean(Xt,reduction_indices=0)

    mmd = tf.reduce_sum(tf.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def mmd2_rbf(X,t,p,sig,weights=None):
    """ Computes the l2-RBF MMD for X given t """

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    if weights is not None:
        Wc = tf.gather(weights,ic)/(tf.reduce_sum(tf.gather(weights,ic))+tf.constant(1e-7))
        Wt = tf.gather(weights,it)/(tf.reduce_sum(tf.gather(weights,it))+tf.constant(1e-7))

    Kcc = tf.exp(-pdist2sq(Xc,Xc)/tf.square(sig))
    Kct = tf.exp(-pdist2sq(Xc,Xt)/tf.square(sig))
    Ktt = tf.exp(-pdist2sq(Xt,Xt)/tf.square(sig))

    if weights is not None:
	Wcc_mask = tf.tile(tf.reshape(Wc, shape=(tf.shape(Xc)[0], 1)), multiples=(1, tf.shape(Xc)[0]))*tf.tile(tf.reshape(Wc, shape=(1, tf.shape(Xc)[0])), multiples=(tf.shape(Xc)[0], 1))
        Wtt_mask = tf.tile(tf.reshape(Wt, shape=(tf.shape(Xt)[0], 1)), multiples=(1, tf.shape(Xt)[0]))*tf.tile(tf.reshape(Wt, shape=(1, tf.shape(Xt)[0])), multiples=(tf.shape(Xt)[0], 1))
        Wct_mask = tf.tile(tf.reshape(Wc, shape=(tf.shape(Xc)[0], 1)), multiples=(1, tf.shape(Xt)[0]))*tf.tile(tf.reshape(Wt, shape=(1, tf.shape(Xt)[0])), multiples=(tf.shape(Xc)[0], 1))

	mmd = tf.square(1.0-p)*(tf.reduce_sum(Kcc*Wcc_mask)-tf.reduce_sum(tf.diag_part(Kcc*Wcc_mask)))/(tf.reduce_sum(Wcc_mask)-tf.reduce_sum(tf.diag_part(Wcc_mask)))
        mmd = mmd + tf.square(p)*(tf.reduce_sum(Ktt*Wtt_mask)-tf.reduce_sum(tf.diag_part(Ktt*Wtt_mask)))/(tf.reduce_sum(Wtt_mask)-tf.reduce_sum(tf.diag_part(Wtt_mask)))
	mmd = mmd - 2.0*p*(1.0-p)*(tf.reduce_sum(Kct*Wct_mask))/(tf.reduce_sum(Wct_mask))
	mmd = 4.0*mmd
    else:
	m = tf.to_float(tf.shape(Xc)[0])
	n = tf.to_float(tf.shape(Xt)[0])

	mmd = tf.square(1.0-p)/(m*(m-1.0))*(tf.reduce_sum(Kcc)-m)
	mmd = mmd + tf.square(p)/(n*(n-1.0))*(tf.reduce_sum(Ktt)-n)
	mmd = mmd - 2.0*p*(1.0-p)/(m*n)*tf.reduce_sum(Kct)
	mmd = 4.0*mmd

    return mmd

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X),1,keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y),1,keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D

def pdist2(X,Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))
def pdist2_mahalanobis(X,Y):
    return safe_sqrt(pdist2sq_mahalanobis(X,Y))
def pdist2sq_mahalanobis(X,Y,cov):
    icov = tf.linalg.inv(cov)
    num_x = tf.shape(X)[0]
    num_y = tf.shape(Y)[0]
    C = -2*tf.matmul(tf.matmul(X,icov),tf.transpose(Y))
    nx = tf.diag(tf.matmul(tf.matmul(X,icov),tf.transpose(X)))
    ny = tf.diag(tf.matmul(tf.matmul(Y,icov),tf.transpose(Y)))
    nx_repeat = tf.tile(nx,tf.constant([1,num_y],tf.int32))
    ny_repeat = tf.tile(ny,tf.constant([1,num_x],tf.int32))
    D =  tf.transpose(nx_repeat) + ny_repeat + C
    return D

def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx

def pop_dist(X,t):
    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    M = pdist2(Xt,Xc)
    return M

def wasserstein(X,t,p,lam=10,weights=None,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))

    if weights is not None:
        Wc = tf.gather(weights,ic)/(tf.reduce_sum(tf.gather(weights,ic))+tf.constant(1e-7))
        Wt = tf.gather(weights,it)/(tf.reduce_sum(tf.gather(weights,it))+tf.constant(1e-7))
	Wtc_mask = tf.tile(tf.reshape(Wt, shape=(tf.shape(Xt)[0], 1)), multiples=(1, tf.shape(Xc)[0]))*tf.tile(tf.reshape(Wc, shape=(1, tf.shape(Xc)[0])), multiples=(tf.shape(Xt)[0], 1))

    ''' Estimate lambda and delta '''
    if weights is not None:
	M_mean = tf.reduce_sum(M*Wtc_mask)
    else:
	M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M,10/(nc*nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam/M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*tf.ones(tf.shape(M[0:1,:]))
    col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))], 0)
    Mt = tf.concat([M,row], 0)
    Mt = tf.concat([Mt,col], 1)


    ''' Compute marginal vectors '''
    if weights is not None:
	a = tf.concat([p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))*tf.reshape(Wt, shape=(tf.shape(Xt)[0], 1)), (1-p)*tf.ones((1,1))], 0)
	b = tf.concat([(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))*tf.reshape(Wc, shape=(tf.shape(Xc)[0], 1)), p*tf.ones((1,1))], 0)
    else:
	a = tf.concat([p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))], 0)
	b = tf.concat([(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))], 0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    a_zeros = tf.cast(tf.equal(a, tf.zeros_like(a)), dtype=tf.float32)
    a_corrected = (1 - a_zeros)*a + a_zeros*tf.constant(1e-7)*tf.ones_like(a)
    ainvK = K/(a_corrected)

    u = a
    for i in range(0,its):
        u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
    v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

    T = u*(tf.transpose(v)*K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T*Mt
    D = 2*tf.reduce_sum(E)

    return D, Mlam

def simplex_project(x,k):
    """ Projects a vector x onto the k-simplex """
    d = x.shape[0]
    mu = np.sort(x,axis=0)[::-1]
    nu = (np.cumsum(mu)-k)/range(1,d+1)
    I = [i for i in range(0,d) if mu[i]>nu[i]]
    theta = nu[I[-1]]
    w = np.maximum(x-theta,0)
    return w

###########################################3

#Helper functions
def mlp(x, layer_dims, activations):
    N_layers = len(layer_dims)
    with tf.variable_scope("mlp", reuse = tf.AUTO_REUSE):
        h = x
        for i in range(N_layers):
            name_str = 'fc'+str(i)
            h = tf.layers.dense(h, layer_dims[i], activation=activations[i], name=name_str)
    return h

def mlp_last_bias(x, layer_dims, activations):
    N_layers = len(layer_dims)
    with tf.variable_scope("mlp", reuse = tf.AUTO_REUSE):
        h = x
        for i in range(N_layers-1):
            h = tf.layers.dense(h, layer_dims[i], activation=activations[i], name='fc'+str(i))
        name_str = tf.get_variable_scope().name+'/fc'+str(N_layers-1)+'/bias:0'
        h = tf.layers.dense(h, layer_dims[N_layers-1], activation=activations[N_layers-1], name='fc'+str(N_layers-1))
    return h, name_str

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def nan_to_zero(tensor):
    return tf.where(tf.is_nan(tensor), tf.zeros_like(tensor), tensor)
