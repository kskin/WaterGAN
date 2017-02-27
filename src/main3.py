import os
import scipy.misc
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from complexmodel3 import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 64, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 64, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("input_water_height", 64, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_water_width", 64, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 64, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_float("max_depth", 3.0, "Dimension of image color. [3.0]")
flags.DEFINE_string("water_dataset", "water_images", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("air_dataset","air_images","The name of dataset with air images")
flags.DEFINE_string("depth_dataset","air_depth","The name of dataset with depth images")
flags.DEFINE_string("waterdepth_dataset","air_depth","The name of dataset with depth images")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("is_stereo", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("num_samples", 4000, "True for visualizing, False for nothing [4000]")
flags.DEFINE_integer("save_epoch", 100, "The size of the output images to produce. If None, same value as output_height [None]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

#  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
#  run_config.GPUOptions(visible_device_list="2")
#  run_config.allow_soft_placement=True
#  run_config.log_device_placement=True
  #with tf.device('/gpu:2'):
  #  sess = tf.Session(config=run_config)
  with tf.Session(config=run_config) as sess:
  #  with tf.device('/gpu:2'):
    dcgan = DCGAN(
      sess,
      input_width=FLAGS.input_width,
      input_height=FLAGS.input_height,
      input_water_width=FLAGS.input_water_width,
      input_water_height=FLAGS.input_water_height,
      output_width=FLAGS.output_width,
      output_height=FLAGS.output_height,
      batch_size=FLAGS.batch_size,
      c_dim=FLAGS.c_dim,
      max_depth = FLAGS.max_depth,
      save_epoch=FLAGS.save_epoch,
      water_dataset_name=FLAGS.water_dataset,
      air_dataset_name = FLAGS.air_dataset,
      depth_dataset_name = FLAGS.depth_dataset,
      waterdepth_dataset_name = FLAGS.depth_dataset,
      input_fname_pattern=FLAGS.input_fname_pattern,
      is_crop=FLAGS.is_crop,
      is_stereo=FLAGS.is_stereo,
      checkpoint_dir=FLAGS.checkpoint_dir,
      sample_dir=FLAGS.sample_dir,
      num_samples = FLAGS.num_samples)

    if FLAGS.is_train:
   #  with tf.device('/gpu:2'):
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir):
        raise Exception("[!] Train a model first, then run test mode")


    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    #OPTION = 1
    #visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  #with tf.device('/gpu:2'):
  tf.app.run()
