import tensorflow as tf
import numpy as np
from cfr.util import mlp
from tqdm import trange
import sys
import time

class Disc_NN:

    def __init__(self, Phi, T,sample_weight, FLAGS):
                
        self.disc_classifier_dims = [FLAGS.dim_disc] * FLAGS.n_disc
        self.disc_classifier_dims.append(1)

        self.disc_classifier_acts = [tf.nn.relu] * FLAGS.n_disc
        self.disc_classifier_acts.append(None)

        self.Phi = Phi
        self.T = T
        self.sample_weight = sample_weight
        self.build_graph()

        self.disc_vars = tf.trainable_variables(scope='disc')
        self.saver = tf.train.Saver(var_list = self.disc_vars)

        # ucode = str(time.time()).split('.')[0]
        # self.save_path = "/home/serge/Documents/causal/disc/t_model_{}.ckpt".format(ucode)

        self.metrics(FLAGS)
    #     self.define_optimizer()

    # def define_optimizer(self):
    #     disc_opt = tf.train.AdamOptimizer()
    #     self.disc_step = t_opt.minimize(self.disc_loss, var_list=self.disc_vars)
        

    def build_graph(self):
        
        with tf.variable_scope("disc"):
            self.t_preds = tf.squeeze(mlp(self.Phi, self.disc_classifier_dims, self.disc_classifier_acts), axis=1)
            
            # print ("Tpred: " + str(self.t_preds.get_shape()))
            # print ("Tshape: " + str(self.T_ph.get_shape()))
            # self.e = tf.nn.sigmoid(self.t_preds)
            # self.e = tf.stop_gradient(self.e)

    # def construct_feed_dict(self, phi, t):        
    #     feeds = {self.Phi_ph: phi, self.T_ph: t}
    #     return feeds

    def metrics(self,FLAGS):

        T_float = tf.squeeze(tf.cast(self.T, tf.float32),axis=1)

        num_1 = tf.reduce_sum(T_float)
        num_0 = tf.reduce_sum(1-T_float)
        num_tot = num_1 + num_0 



        # disc_weights = (num_tot/num_1)*T_float + (num_tot/num_0)*(1-T_float)
        self.disc = None
        if(FLAGS.imb_fun == 'disc'):
            # self.disc = -tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(T_float, self.t_preds, pos_weight=disc_weights))
            i0 = tf.to_int32(tf.where(self.T < 1)[:,0])
            i1 = tf.to_int32(tf.where(self.T > 0)[:,0])

            t_preds_0 = tf.gather(self.t_preds, i0)
            t_preds_1 = tf.gather(self.t_preds, i1)
            self.disc = tf.reduce_mean(t_preds_0) - tf.reduce_mean(t_preds_1)
        if(FLAGS.imb_fun == 'weighted_disc'):
            i0 = tf.to_int32(tf.where(self.T < 1)[:,0])
            i1 = tf.to_int32(tf.where(self.T > 0)[:,0])

            gw = self.t_preds*self.sample_weight
            gw0 = tf.gather(gw,i0)
            gw1 = tf.gather(gw,i1)

            self.disc = tf.abs(tf.reduce_mean(gw0)-tf.reduce_mean(gw1))
            # self.disc = tf.reduce_mean(t_preds_0/(1-e_0)) - tf.reduce_mean(t_preds_1/e_1)
        # t_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=t_preds, labels= tf.cast(D,tf.float32)))

        # self.t_acc, _ = tf.metrics.mean_per_class_accuracy(labels=T_float, predictions=self.e>0.5, num_classes=2)

        # t_ps = tf.cast(self.e>0.5, tf.int32)
        squeezed_T = tf.squeeze(self.T,axis=1)
        t_hat = tf.where(self.t_preds > 0.0, tf.ones((tf.shape(self.t_preds)), dtype=tf.float32), tf.zeros((tf.shape(self.t_preds)), dtype=tf.float32))
        corr = tf.where(tf.equal(t_hat, squeezed_T), tf.ones((tf.shape(self.t_preds)), dtype=tf.float32), tf.zeros((tf.shape(self.t_preds)), dtype=tf.float32))
        self.corr = corr
        acc1 = tf.reduce_sum(corr*squeezed_T)/tf.reduce_sum(squeezed_T)
        acc2 = tf.reduce_sum(corr*(1-squeezed_T))/tf.reduce_sum(1-squeezed_T)
        self.disc_acc = (acc1 + acc2)/2


    # def train(self, sess, D_exp, I_valid, niter=1000):
    #     n = D_exp['x'].shape[0]
    #     I = range(n); I_train = list(set(I)-set(I_valid))

    #     X_valid = D_exp['x'][I_valid, :]
    #     X_train = D_exp['x'][I_train, :]
    #     t_valid = D_exp['t'][I_valid, :]
    #     t_train = D_exp['t'][I_train, :]

    #     print ('Number of samples: {}'.format(n))
        
    #     feeds_train = self.construct_feed_dict(X_train, t_train)
    #     feeds_val = self.construct_feed_dict(X_valid, t_valid) 
  
    #     # TODO:
    #     # edit this function to include printing t metrics
    #     # save model with the best val performance
    #     # save the histories of losses, AUCs etc.
    #     # save weight distributions for samples over time
    #     pbar = trange(niter)
    #     best_val_loss = np.inf
    #     for i in pbar:
    #         _, lt, acc = sess.run([self.t_step, self.t_loss, self.t_acc], feed_dict=feeds_train)
    #         lt_val, acc_val = sess.run([self.t_loss, self.t_acc], feed_dict=feeds_val)
            
    #         if (lt_val<best_val_loss):
            
    #             best_acc = acc_val
    #             best_val_loss = lt_val
    #             e_train = sess.run(self.e, feed_dict=feeds_train)
    #             e_val = sess.run(self.e, feed_dict=feeds_val)
    #             self.saver.save(sess, self.save_path)

    #     #             t_val = feeds_val[D]
    #     #             t_train = feeds[D]
    #     #             PS_best_1_val = PS_best_val[np.where(t_val)]
    #     #             PS_best_0_val = PS_best_val[np.where(1-t_val)]
    #     #             PS_best_1_train = PS_best_train[np.where(t_train)]
    #     #             PS_best_0_train = PS_best_train[np.where(1-t_train)]
    #         # print (acc, acc_val)
    #         pbar.set_description("lt=%.3f, acc=%.3f, lt_val=%.3f, acc_val=%.3f"  % (lt, acc, lt_val, acc_val))
            
    #     print("best val ACC: %.3f" % (best_acc))
    #     #     print(describe(PS_best_1))
    #     #     print(describe(PS_best_0))
    #     return e_train, e_val