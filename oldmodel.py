from __future__ import division
import os
import time
import scipy
from scipy import misc
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import scipy.io as sio
from ops import *
from utils import *

class DCGAN(object):
  def __init__(self, sess, input_height=64, input_width=64, is_crop=False,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, water_dataset_name='default',air_dataset_name='default',
         depth_dataset_name='default',input_fname_pattern='*.png', checkpoint_dir=None, sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.is_crop = is_crop
    self.is_grayscale = (c_dim == 1)

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.c_dim = c_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.water_dataset_name = water_dataset_name
    self.air_dataset_name = air_dataset_name
    self.depth_datset_name = depth_dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

    image_dims = [self.input_height, self.input_width, self.c_dim]

    self.water_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.air_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='air_images')
    self.depth_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + [self.input_height,self.input_width], name='depth')
    self.water_sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    water_inputs = self.water_inputs
    water_sample_inputs = self.water_sample_inputs
    air_inputs = self.air_inputs
    depth_inputs = self.depth_inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = tf.summary.histogram("z", self.z)

    self.G,eta,beta = self.wc_generator(self.z,air_inputs, depth_inputs)
    self.D, self.D_logits = self.discriminator(water_inputs)

    self.sampler = self.wc_sampler(self.z,air_inputs, depth_inputs)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    self.d_sum = tf.summary.histogram("d", self.D)
    self.d__sum = tf.summary.histogram("d_", self.D_)
    self.G_sum = tf.summary.image("G", self.G,max_outputs=200)

    self.d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits, targets=tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits_, targets=tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits_, targets=tf.ones_like(self.D_)))

    self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
    self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    self.saver = tf.train.Saver()

  def train(self, config):
    """Train DCGAN"""
    water_data = glob(os.path.join("./data", config.water_dataset, self.input_fname_pattern))
    air_data = glob(os.path.join("./data", config.air_dataset, self.input_fname_pattern))
    depth_data = glob(os.path.join("./data", config.depth_dataset, "*.mat"))
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = tf.summary.merge([self.z_sum, self.d__sum,self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    #self.G_img = tf.summary.merge([self.G_sum])
    self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1,1,size=(self.sample_num,self.z_dim))

    # Load data
    sample_files = water_data[0:self.sample_num]
    sample = [
        get_image(sample_file,
                  input_height=self.input_height,
                  input_width=self.input_width,
                  resize_height=self.output_height,
                  resize_width=self.output_width,
                  is_crop=self.is_crop,
                  is_grayscale=self.is_grayscale) for sample_file in sample_files]
    water_sample_inputs = np.array(sample).astype(np.float32)

    # Start training
    counter = 1
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      water_data = sorted(glob(os.path.join(
        "./data", config.water_dataset, self.input_fname_pattern)))
      air_data = sorted(glob(os.path.join(
        "./data", config.air_dataset, self.input_fname_pattern)))
      depth_data = sorted(glob(os.path.join(
        "./data", config.depth_dataset, "*.mat")))

      water_batch_idxs = min(min(len(air_data),len(water_data)), config.train_size) // config.batch_size
      air_batch_idxs = water_batch_idxs

      # Load water images
      for idx in xrange(0, water_batch_idxs):
        water_batch_files = water_data[idx*config.batch_size:(idx+1)*config.batch_size]
        water_batch = [
            get_image(water_batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      is_crop=self.is_crop,
                      is_grayscale=self.is_grayscale) for water_batch_file in water_batch_files]
        water_batch_images = np.array(water_batch).astype(np.float32)

      # Load air images and depth
      for idx in xrange(0, air_batch_idxs):
        air_batch_files = air_data[idx*config.batch_size:(idx+1)*config.batch_size]
        air_batch = [
            get_image(air_batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      is_crop=self.is_crop,
                      is_grayscale=self.is_grayscale) for air_batch_file in air_batch_files]
        depth_batch_files = depth_data[idx*config.batch_size:(idx+1)*config.batch_size]
        depth_batch = [sio.loadmat(depth_batch_file)["im_depth"] for depth_batch_file in depth_batch_files]
        air_batch_images = np.array(air_batch).astype(np.float32)
        #print(depth_batch)
        #depth_batch_images_i = np.expand_dims(depth_batch,axis=3)
        #depth_batch_images = np.broadcast_to(depth_batch_images_i,(config.batch_size,self.output_height,self.output_width,self.c_dim))
        depth_batch_images = depth_batch

        batch_z = np.random.uniform(-1,1,[config.batch_size,self.z_dim]).astype(np.float32)

        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict={ self.z: batch_z,self.water_inputs: water_batch_images,self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images})
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={self.z:batch_z, self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images })
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={self.z:batch_z,self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images})
        self.writer.add_summary(summary_str, counter)
        #print(self.G)
        errD_fake = self.d_loss_fake.eval({self.z:batch_z, self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images })
        errD_real = self.d_loss_real.eval({self.water_inputs: water_batch_images })
        errG = self.g_loss.eval({self.z:batch_z,self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, water_batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))
        if np.mod(counter, 10) == 1:
          print(self.sess.run('wc_generator/g_atten/g_eta:0'))
          print(self.sess.run('wc_generator/g_bscatter/g_beta:0'))

          #if config.water_dataset == 'mnist':
          #  print("oops")
          #else:
            #try:
           # samples, d_loss, g_loss = self.sess.run(
         #     [self.wc_sampler, self.d_loss, self.g_loss],
         #     feed_dict={
         #         self.air_inputs: air_batch_images,
         #         self.depth_inputs: depth_batch_images,
         #         self.water_inputs: water_sample_inputs,
          #    },
          #  )
          #  print("SAVING SAMPLES")
          #  print(samples)
          #  save_images(samples, [8, 8],
          #        './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
          #  print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            #except:
            #print("one pic error!...")

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)
        #print(self.g_vars.eval())

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

      return tf.nn.sigmoid(h4), h4

  def wc_generator(self, z, image, depth, y=None):
    with tf.variable_scope("wc_generator") as scope:

      # define image dims
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_h4, s_h8, s_h16 = \
          int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
      s_w2, s_w4, s_w8, s_w16 = \
          int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

 #depth_batch_images_i = np.expand_dims(depth_batch,axis=3)
        #depth_batch_images = np.broadcast_to(depth_batch_images_i,(config.batch_size,self.output_height,self.$ #depth_batch_images_i = np.expand_dims(depth_batch,axis=3)
        #depth_batch_images = np.broadcast_to(depth_batch_images_i,(config.batch_size,self.output_height,self.$
      # Augment 1: attenuation
      with tf.variable_scope("g_atten"):
          init = tf.random_uniform([1,1,1,3],minval=-1,maxval=0,dtype=tf.float32)
          eta = tf.get_variable("g_eta",initializer=init)
          eta_d = tf.mul(depth3,eta)
      h0 = tf.mul(image,tf.exp(eta_d))

      # Augment 2: backscatter
      with tf.variable_scope("g_bscatter"):
          init = tf.random_uniform([1],minval=0,maxval=1,dtype=tf.float32)
          beta = tf.get_variable("g_beta",initializer=init)
      h1 = h0 + tf.mul(beta,(1.0-tf.exp(eta_d)))

      return h1, eta, beta

  def wc_sampler(self, z, image, depth, y=None):
    with tf.variable_scope("wc_generator") as scope:
      scope.reuse_variables()

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_h4, s_h8, s_h16 = \
          int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
      s_w2, s_w4, s_w8, s_w16 = \
          int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

      # Augment 1: attenuation
      with tf.variable_scope("g_atten"):
          init = tf.random_uniform([1,1,1,3],minval=-1,maxval=0,dtype=tf.float32)
          eta = tf.get_variable("g_eta",initializer=init)
          eta_d = tf.mul(depth,eta)
      h0 = tf.mul(image,tf.exp(eta_d))

      # Augment 2: backscatter
      with tf.variable_scope("g_bscatter"):
          init = tf.random_uniform([1],minval=0,maxval=256,dtype=tf.float32)
          beta = tf.get_variable("g_beta",initializer=init)
      h1 = h0 + tf.mul(beta,(1.0-tf.exp(eta_d)))

      return h1

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.water_dataset_name, self.batch_size,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False
