import tensorflow as tf
import numpy as np
from cfr.util import mlp
from tqdm import trange
import sys
import time

class Propensity_NN:

    def __init__(self, X_ph, T_ph, FLAGS):
                
        self.t_classifier_dims = [FLAGS.dim_prop] * FLAGS.n_prop
        self.t_classifier_dims.append(1)

        self.t_classifier_acts = [tf.nn.relu] * FLAGS.n_prop
        self.t_classifier_acts.append(None)

        self.X_ph = X_ph
        self.T_ph = T_ph

        self.build_graph()

        self.tvars = tf.trainable_variables(scope='t_classifier')
        self.saver = tf.train.Saver(var_list = self.tvars)
        self.numiter = FLAGS.iter_prop
        np.random.seed()
        rand = np.random.randint(1,100000000)
        np.random.seed(123)

        ucode = str(time.time()).split('.')[0] + '_' + str(rand)
        
	# self.save_path = "./ckpts/t_prop/t_model_{}.ckpt".format(ucode)
        self.save_path = "/home/serge/Documents/causal/t_prop2/t_model_{}.ckpt".format(ucode)

        self.metrics()
        self.define_optimizer()

    def define_optimizer(self):
        t_opt = tf.train.AdamOptimizer()
        self.t_step = t_opt.minimize(self.t_loss, var_list=self.tvars)
        

    def build_graph(self):     
        with tf.variable_scope("t_classifier"):
            self.t_preds = tf.squeeze(mlp(self.X_ph, self.t_classifier_dims, self.t_classifier_acts), axis=1)
            
            # print ("Tpred: " + str(self.t_preds.get_shape()))
            # print ("Tshape: " + str(self.T_ph.get_shape()))
            self.e = tf.nn.sigmoid(self.t_preds)
	    
	    ####added block
            # T_float = tf.squeeze(tf.cast(self.T_ph, tf.float32), axis=1)
            # num_1 = tf.reduce_sum(T_float)
            # num_0 = tf.reduce_sum(1-T_float)
            # num_tot = num_1 + num_0
            # p_1 = tf.cast(tf.squeeze(num_1/num_tot), tf.float32)
            # p_0 = tf.cast(tf.squeeze(num_0/num_tot), tf.float32)
            # self.e = (p_1*self.e)/(p_1*self.e + p_0*(1-self.e)) #corrected for prior probabilities
	    ####added block

            self.e = tf.stop_gradient(self.e)
 
    def construct_feed_dict(self, x, t):        
        feeds = {self.X_ph: x, self.T_ph: t}
        return feeds

    def metrics(self):

        T_float = tf.squeeze(tf.cast(self.T_ph, tf.float32),axis=1)

        num_1 = tf.reduce_sum(T_float)
        num_0 = tf.reduce_sum(1-T_float)
        num_tot = num_1 + num_0 

        t_weights = (num_tot/num_1)*T_float + (num_tot/num_0)*(1-T_float)
        self.t_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(T_float, self.t_preds, pos_weight=t_weights))
        # t_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=t_preds, labels= tf.cast(D,tf.float32)))

        # self.t_acc, _ = tf.metrics.mean_per_class_accuracy(labels=T_float, predictions=self.e>0.5, num_classes=2)

        # t_ps = tf.cast(self.e>0.5, tf.int32)
        squeezed_T_ph = tf.squeeze(self.T_ph,axis=1)
        # T_int = tf.cast(tf.squeeze(self.T_ph,axis=1), tf.int32)
        # self.T_int = T_int
        # corr = tf.cast(tf.equal(t_ps, T_int), tf.int32)
        # t_hat = tf.cast(self.e>0.5, tf.float32)
        t_hat = tf.where(self.t_preds > 0.0, tf.ones((tf.shape(self.t_preds)), dtype=tf.float32), tf.zeros((tf.shape(self.t_preds)), dtype=tf.float32))
        corr = tf.where(tf.equal(t_hat, squeezed_T_ph), tf.ones((tf.shape(self.t_preds)), dtype=tf.float32), tf.zeros((tf.shape(self.t_preds)), dtype=tf.float32))
        self.corr = corr
        # acc1 = tf.reduce_sum(corr*T_int)/tf.reduce_sum(T_int)
        # acc2 = tf.reduce_sum(corr*(1-T_int))/tf.reduce_sum(1-T_int)
        acc1 = tf.reduce_sum(corr*squeezed_T_ph)/tf.reduce_sum(squeezed_T_ph)
        acc2 = tf.reduce_sum(corr*(1-squeezed_T_ph))/tf.reduce_sum(1-squeezed_T_ph)
        self.t_acc = (acc1 + acc2)/2


    def train(self, sess, D_exp, I_valid):

        n = D_exp['x'].shape[0]
        I = range(n); I_train = list(set(I)-set(I_valid))

        X_valid = D_exp['x'][I_valid, :]
        X_train = D_exp['x'][I_train, :]
        t_valid = D_exp['t'][I_valid, :]
        t_train = D_exp['t'][I_train, :]

        print ('Number of samples: {}'.format(n))
        
        feeds_train = self.construct_feed_dict(X_train, t_train)
        feeds_val = self.construct_feed_dict(X_valid, t_valid) 
  
        # TODO:
        # edit this function to include printing t metrics
        # save model with the best val performance
        # save the histories of losses, AUCs etc.
        # save weight distributions for samples over time
        pbar = trange(self.numiter)
        best_val_loss = np.inf
        for i in pbar:
            _, lt, acc = sess.run([self.t_step, self.t_loss, self.t_acc], feed_dict=feeds_train)
            lt_val, acc_val = sess.run([self.t_loss, self.t_acc], feed_dict=feeds_val)
            
            if (lt_val<best_val_loss):
            
                best_acc = acc_val
                best_val_loss = lt_val
                e_train = sess.run(self.e, feed_dict=feeds_train)
                e_val = sess.run(self.e, feed_dict=feeds_val)
                self.saver.save(sess, self.save_path)

        #             t_val = feeds_val[D]
        #             t_train = feeds[D]
        #             PS_best_1_val = PS_best_val[np.where(t_val)]
        #             PS_best_0_val = PS_best_val[np.where(1-t_val)]
        #             PS_best_1_train = PS_best_train[np.where(t_train)]
        #             PS_best_0_train = PS_best_train[np.where(1-t_train)]
            # print (acc, acc_val)
            pbar.set_description("lt=%.3f, acc=%.3f, lt_val=%.3f, acc_val=%.3f"  % (lt, acc, lt_val, acc_val))
            
        print("best val ACC: %.3f" % (best_acc))
        #     print(describe(PS_best_1))
        #     print(describe(PS_best_0))
        return e_train, e_val
