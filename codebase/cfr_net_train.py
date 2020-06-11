import tensorflow as tf
import numpy as np
import sys, os
import getopt
import random
import datetime
import traceback


import cfr.cfr_net as cfr
from cfr.util import *
from train_propensity_nn import *

''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
# flags
tf.app.flags.DEFINE_string('weight_scheme', 'JW', """JW, IPW, MW, OW, TruncIPW""")
tf.app.flags.DEFINE_integer('n_prop', 1, """Number of propensity arch hidden layers""")
tf.app.flags.DEFINE_integer('dim_prop', 20, """Dim of propensity arch hidden layers""")
tf.app.flags.DEFINE_integer('iter_prop', 1000, """Dim of propensity arch hidden layers""")
tf.app.flags.DEFINE_float('lr_prop', 0.001, """Propensity learning rate""")
tf.app.flags.DEFINE_boolean('reweight_imb', 1, """Whether to reweight samples for calculating the discrepancy. """)
tf.app.flags.DEFINE_float('trunc_alpha', 0.1, """Truncation threshold for TruncIPW weighting """)
tf.app.flags.DEFINE_integer('weight_norm', 0, """ Whether to divide by the sum of the weights """)
tf.app.flags.DEFINE_integer('use_batches', 1, """ Whether to use batches """)
tf.app.flags.DEFINE_integer('balance_batches', 1, """ Whether to balance batches according to overall treat/control ratio""")
tf.app.flags.DEFINE_integer('joh_balance', 0, """ Reweight samples by 1/n1 and 1/n0, respectively """)

tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('n_in', 2, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 2, """Number of regression layers. """)
tf.app.flags.DEFINE_float('p_alpha', 1e-4, """Imbalance regularization param. """)
tf.app.flags.DEFINE_float('p_lambda', 0.0, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 1, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 0.9, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 0.9, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'relu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 0.05, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.5, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('dim_in', 100, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'none', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
tf.app.flags.DEFINE_integer('experiments', 1, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 2000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.01, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('wass_iterations', 20, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_float('wass_lambda', 1, """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_bpt', 0, """Backprop through T matrix? """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_string('outdir', '../results/tfnet_topic/alpha_sweep_22_d100/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', '../data/topic/csv/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'topic_dmean_seed_%d.csv', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', '', """Test data filename form. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_integer('use_p_correction', 1, """Whether to use population size p(t) in mmd/wass.""")
tf.app.flags.DEFINE_string('optimizer', 'RMSProp', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_string('imb_fun', 'mmd_lin', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.app.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', -1, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 0, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)

if FLAGS.sparse:
    import scipy.sparse as sparse

NUM_ITERATIONS_PER_DECAY = 100

__DEBUG__ = False
if FLAGS.debug:
    __DEBUG__ = True

def train(CFR, sess, train_step, D, I_valid, D_test, logfile, i_exp, propensity_model=None):
    """ Trains a CFR model on supplied data """

    ''' Train/validation split '''
    n = D['x'].shape[0]
    I = range(n); I_train = list(set(I)-set(I_valid))
    n_train = len(I_train)
    n_treated = int(np.sum(D['t'][I_train,:]))
    n_control = n_train-n_treated
    I_train_np = np.array(I_train)
    I_treated = I_train_np[np.squeeze(D['t'][I_train])==1]
    I_control = I_train_np[np.squeeze(D['t'][I_train])==0]
    ''' Compute treatment probability'''
    p_treated = np.mean(D['t'][I_train,:])

    ''' Set up loss feed_dicts'''
    dict_factual = {CFR.x: D['x'][I_train,:], CFR.t: D['t'][I_train,:], CFR.y_: D['yf'][I_train,:], \
      CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, \
      CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}

    if FLAGS.val_part > 0:
        dict_valid = {CFR.x: D['x'][I_valid,:], CFR.t: D['t'][I_valid,:], CFR.y_: D['yf'][I_valid,:], \
          CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, \
          CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}

    if D['HAVE_TRUTH']:
        dict_cfactual = {CFR.x: D['x'][I_train,:], CFR.t: 1-D['t'][I_train,:], CFR.y_: D['ycf'][I_train,:], \
          CFR.do_in: 1.0, CFR.do_out: 1.0}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    # propensity network
    if (propensity_model):
        propensity_model.train(sess, D, I_valid) 
        propensity_model.saver.restore(sess, propensity_model.save_path)

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []

    ''' Compute losses '''
    losses = []
    obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],\
      feed_dict=dict_factual)

    cf_error = np.nan
    if D['HAVE_TRUTH']:
        cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

    valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
    if FLAGS.val_part > 0:
        valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],\
          feed_dict=dict_valid)

    losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])

    objnan = False

    reps = []
    reps_test = []
    reps_norm = []
    reps_norm_test = []
    best_weights = None
    best_weights_test = None
    best_rep = None
    best_rep_test = None
    best_rep_norm = None
    best_rep_norm_test = None
    ''' Train for multiple iterations '''


    min_valid_obj = float('inf')
    for i in range(FLAGS.iterations):

        ''' Fetch sample '''
        if(FLAGS.use_batches):
            if(FLAGS.balance_batches):
                # construct batches according to overall treatment/control ratio
                n_treated_batch = int(p_treated*FLAGS.batch_size)
                n_control_batch = FLAGS.batch_size-n_treated_batch
                I_treated_batch = random.sample(range(0, n_treated), n_treated_batch)
                I_control_batch = random.sample(range(0,n_control),n_control_batch)
                
                x_batch_treated = D['x'][I_treated,:][I_treated_batch,:]
                t_batch_treated = D['t'][I_treated,:][I_treated_batch]
                y_batch_treated = D['yf'][I_treated,:][I_treated_batch]
                
                x_batch_control = D['x'][I_control,:][I_control_batch,:]
                t_batch_control = D['t'][I_control,:][I_control_batch]
                y_batch_control = D['yf'][I_control,:][I_control_batch]
                
                x_batch = np.concatenate([x_batch_treated,x_batch_control],axis=0)
                t_batch = np.concatenate([t_batch_treated,t_batch_control])
                y_batch = np.concatenate([y_batch_treated,y_batch_control])
            else:
                I = random.sample(range(0, n_train), FLAGS.batch_size)
                x_batch = D['x'][I_train,:][I,:]
                t_batch = D['t'][I_train,:][I]
                y_batch = D['yf'][I_train,:][I]
        else:
            x_batch = D['x'][I_train,:]
            t_batch = D['t'][I_train,:]
            y_batch = D['yf'][I_train,:]

        if __DEBUG__:
            M = sess.run(cfr.pop_dist(CFR.x, CFR.t), feed_dict={CFR.x: x_batch, CFR.t: t_batch})
            log(logfile, 'Median: %.4g, Mean: %.4f, Max: %.4f' % (np.median(M.tolist()), np.mean(M.tolist()), np.amax(M.tolist())))

        ''' Do one step of gradient descent '''
        if not objnan:
            feeds = {CFR.x: x_batch, CFR.t: t_batch, \
                CFR.y_: y_batch, CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out, \
                CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}
            
            sess.run(train_step, feed_dict=feeds)
        ''' Project variable selection weights '''
        if FLAGS.varsel:
            wip = simplex_project(sess.run(CFR.weights_in[0]), 1)
            sess.run(CFR.projection, feed_dict={CFR.w_proj: wip})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss,f_error,imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                    feed_dict=dict_factual)
            rep = sess.run(CFR.h_rep_norm, feed_dict={CFR.x: D['x'], CFR.do_in: 1.0})
            rep_norm = np.mean(np.sqrt(np.sum(np.square(rep), 1)))

            cf_error = np.nan
            if D['HAVE_TRUTH']:
                cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
            
            if FLAGS.val_part > 0:
                    valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], feed_dict=dict_valid)

            losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])
            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tImb: %.3f, \tVal: %.3f,\tValImb: %.3f, \tValObj: %.2f' \
                        % (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj)

            if FLAGS.loss == 'log':
                y_pred = sess.run(CFR.output, feed_dict={CFR.x: x_batch, \
                    CFR.t: t_batch, CFR.do_in: 1.0, CFR.do_out: 1.0})
                y_pred = 1.0*(y_pred > 0.5)
                acc = 100*(1 - np.mean(np.abs(y_batch - y_pred)))
                loss_str += ',\tAcc: %.2f%%' % acc

            log(logfile, loss_str)

            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True

        ''' Compute predictions every M iterations '''
        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i==FLAGS.iterations-1:

            y_pred_f = sess.run(CFR.output, feed_dict={CFR.x: D['x'], \
                CFR.t: D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0,CFR.p_t:p_treated})
            y_pred_cf = sess.run(CFR.output, feed_dict={CFR.x: D['x'], \
                CFR.t: 1-D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0,CFR.p_t:p_treated})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf),axis=1))

            if D_test is not None:
                y_pred_f_test = sess.run(CFR.output, feed_dict={CFR.x: D_test['x'], \
                    CFR.t: D_test['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
                y_pred_cf_test = sess.run(CFR.output, feed_dict={CFR.x: D_test['x'], \
                    CFR.t: 1-D_test['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test),axis=1))

            if FLAGS.save_rep:
                if valid_obj<min_valid_obj:
                    min_valid_obj = valid_obj
                    best_weights = sess.run([CFR.sample_weight], feed_dict={CFR.x: D['x'], CFR.t: D['t'],CFR.p_t: p_treated})
                    best_rep_norm = sess.run([CFR.h_rep_norm], feed_dict={CFR.x: D['x'], \
                        CFR.do_in: 1.0, CFR.do_out: 0.0})
                    best_rep = sess.run([CFR.h_rep], feed_dict={CFR.x: D['x'], \
                        CFR.do_in: 1.0, CFR.do_out: 0.0})

                    if D_test is not None:
                        best_weights_test = sess.run([CFR.sample_weight], feed_dict={CFR.x: D_test['x'], CFR.t: D_test['t'],CFR.p_t: p_treated})
                        best_rep_norm_test = sess.run([CFR.h_rep_norm], feed_dict={CFR.x: D_test['x'], \
                            CFR.do_in: 1.0, CFR.do_out: 0.0})
                        best_rep_test = sess.run([CFR.h_rep], feed_dict={CFR.x: D_test['x'], \
                            CFR.do_in: 1.0, CFR.do_out: 0.0})

    if(propensity_model):
        e_train = sess.run(propensity_model.e,feed_dict={CFR.x:D['x']})
        e_test =  sess.run(propensity_model.e,feed_dict={CFR.x:D_test['x']})
    else:
        e_train = None
        e_test = None

    return losses, preds_train, preds_test, best_rep, best_rep_test,e_train,e_test,best_weights,best_weights_test,best_rep_norm,best_rep_norm_test

def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = outdir+'result'
    npzfile_test = outdir+'result.test'
    repfile = outdir+'reps'
    repfile_test = outdir+'reps.test'
    repfile_norm = outdir+'reps_norm'
    repfile_norm_test = outdir+'reps_norm.test'
    weightfile = outdir+'weights'
    weightfile_test = outdir + 'weights.test'
    outform = outdir+'y_pred'
    outform_test = outdir+'y_pred.test'
    lossform = outdir+'loss'
    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform

    has_test = False
    if not FLAGS.data_test == '': # if test set supplied
        has_test = True
        dataform_test = FLAGS.datadir + FLAGS.data_test

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir+'config.txt')

    log(logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha,FLAGS.p_lambda))

    ''' Load Data '''
    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform % 1
        if has_test:
            datapath_test = dataform_test % 1

    log(logfile,     'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath)
    D_test = None
    if has_test:
        D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    ''' Start Session '''
    sess = tf.Session()

    ''' Initialize input placeholders '''
    x  = tf.placeholder("float", shape=[None, D['dim']], name='x') # Features
    t  = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    ''' Parameter placeholders '''
    r_alpha = tf.placeholder("float", name='r_alpha')
    r_lambda = tf.placeholder("float", name='r_lambda')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_in, FLAGS.dim_out]

    propensity_model = None
    e = None
    # Define propensity model
    if FLAGS.reweight_sample:
        propensity_model = Propensity_NN(x, t, FLAGS)
        e = propensity_model.e

    CFR = cfr.cfr_net(x, t, y_, p, e, FLAGS, r_alpha, r_lambda, do_in, do_out, dims)

    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(FLAGS.lrate, global_step, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)
    
    opt = None

    if FLAGS.optimizer == 'Adagrad':
        opt = tf.train.AdagradOptimizer(lr)
    elif FLAGS.optimizer == 'GradientDescent':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(lr)
    else:
        opt = tf.train.RMSPropOptimizer(lr, FLAGS.decay)

    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    train_step = opt.minimize(CFR.tot_loss,global_step=global_step, var_list = all_vars)

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []
    if(FLAGS.save_rep):
        all_reps = []
        all_reps_norm = []
        all_reps_test = []
        all_reps_norm_test = []
        all_sample_weights = []
        all_sample_weights_test = []
    if(propensity_model):
        all_e_train= []
        all_e_test = []
    else:
        all_e_train = None
        all_e_test = None

    if FLAGS.varsel:
        all_weights = None
        all_beta = None

    all_preds_test = []

    ''' Handle repetitions '''
    n_experiments = FLAGS.experiments
    if FLAGS.repetitions>1:
        if FLAGS.experiments>1:
            log(logfile, 'ERROR: Use of both repetitions and multiple experiments is currently not supported.')
            sys.exit(1)
        n_experiments = FLAGS.repetitions

    ''' Run for all repeated experiments '''
    for i_exp in range(1,n_experiments+1):

        if FLAGS.repetitions>1:
            log(logfile, 'Training on repeated initialization %d/%d...' % (i_exp, FLAGS.repetitions))
        else:
            log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))

        ''' Load Data (if multiple repetitions, reuse first set)'''

        if i_exp==1 or FLAGS.experiments>1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x']  = D['x'][:,:,i_exp-1]
                D_exp['t']  = D['t'][:,i_exp-1:i_exp]
                D_exp['yf'] = D['yf'][:,i_exp-1:i_exp]
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf'][:,i_exp-1:i_exp]
                else:
                    D_exp['ycf'] = None

                if has_test:
                    D_exp_test = {}
                    D_exp_test['x']  = D_test['x'][:,:,i_exp-1]
                    D_exp_test['t']  = D_test['t'][:,i_exp-1:i_exp]
                    D_exp_test['yf'] = D_test['yf'][:,i_exp-1:i_exp]
                    if D_test['HAVE_TRUTH']:
                        D_exp_test['ycf'] = D_test['ycf'][:,i_exp-1:i_exp]
                    else:
                        D_exp_test['ycf'] = None
            else:
                datapath = dataform % i_exp
                D_exp = load_data(datapath)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    D_exp_test = load_data(datapath_test)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']
            if has_test:
                D_exp_test['HAVE_TRUTH'] = D_test['HAVE_TRUTH']

        ''' Split into training and validation sets '''
        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)
                
        ''' Run training loop '''
        losses, preds_train, preds_test, reps, reps_test,e_train,e_test,sample_weights,sample_weights_test,reps_norm,reps_norm_test = \
            train(CFR, sess, train_step, D_exp, I_valid, \
                D_exp_test, logfile, i_exp, propensity_model)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_losses.append(losses)

        if(FLAGS.save_rep):
            all_reps.append(reps)
            all_reps_norm.append(reps_norm)
            all_reps_test.append(reps_test)
            all_reps_norm_test.append(reps_norm_test)
            all_sample_weights.append(sample_weights)
            all_sample_weights_test.append(sample_weights_test)
        if(propensity_model):
            all_e_train.append(e_train)
            all_e_test.append(e_test)

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train,1,3),0,2)
        if  has_test:
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test,1,3),0,2)
        out_losses = np.swapaxes(np.swapaxes(all_losses,0,2),0,1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)
        if FLAGS.output_csv:
            np.savetxt('%s_%d.csv' % (outform,i_exp), preds_train[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (outform_test,i_exp), preds_test[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (lossform,i_exp), losses, delimiter=',')

        ''' Compute weights if doing variable selection '''
        if FLAGS.varsel:
            if i_exp == 1:
                all_weights = sess.run(CFR.weights_in[0])
                all_beta = sess.run(CFR.weights_pred)
            else:
                all_weights = np.dstack((all_weights, sess.run(CFR.weights_in[0])))
                all_beta = np.dstack((all_beta, sess.run(CFR.weights_pred)))

        ''' Save results and predictions '''
        all_valid.append(I_valid)
        if FLAGS.varsel:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, w=all_weights, beta=all_beta, val=np.array(all_valid))
        else:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid),e=all_e_train)

        if has_test:
            np.savez(npzfile_test, pred=out_preds_test,e=all_e_test)

        ''' Save representations '''
        if FLAGS.save_rep:
            np.savez(repfile, rep=all_reps)
            np.savez(repfile_norm, rep=all_reps_norm)
            np.savez(weightfile,weights=all_sample_weights)
            if has_test:
                np.savez(repfile_test, rep=all_reps_test)
                np.savez(repfile_norm_test,rep=all_reps_norm_test)
                np.savez(weightfile_test,weights=all_sample_weights_test)
                

def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir+'/results_'+timestamp+'/'
    os.mkdir(outdir)

    try:
        run(outdir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise

if __name__ == '__main__':
    tf.app.run()
