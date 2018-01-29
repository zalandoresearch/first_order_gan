import numpy as np
import os
import pprint
import scipy.misc
import time

from gan_language_JSD import FOGAN

import tensorflow as tf

pp = pprint.PrettyPrinter()

flags = tf.app.flags

# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!
flags.DEFINE_integer("iterations", 200000, "Now many iterations to train for [200000]")
flags.DEFINE_string("gan_divergence", "WGANGP", "divergence to use, can be 'WGANGP', 'PWGAN' or 'FOGAN' ['WGANGP']")
flags.DEFINE_boolean("squared_divergence", False, "Optimized something close to squred Wasserstein distance")
flags.DEFINE_string("data_path", "data", "path to the extracted data files [data]")
flags.DEFINE_integer("seq_len", 32, "Sequence length in characters [32]")
flags.DEFINE_integer("max_n_examples", 10000000, "Max number of data examples to load. Lower for faster loading [10000000]")
flags.DEFINE_integer("n_ngrams", 6, "NGRAM statistics for N_NGRAMS [6]")
flags.DEFINE_integer("batch_size", 64, "Batch size for test and training [64]")
flags.DEFINE_integer("dim", 512, "Model dimensionality. This is fairly slow and overfits [512]")
flags.DEFINE_string("activation_d", "relu", "activation function used in discriminator, can be 'relu' or 'elu' [relu]")
flags.DEFINE_boolean("batch_norm_d", False, "use batch norm for discriminator")
flags.DEFINE_boolean("batch_norm_g", False, "use batch norm for generator")
flags.DEFINE_integer("critic_iters", 10, "How many critic iterations per generator iteration [10]")
flags.DEFINE_float("learning_rate_d", .0001, "Learnining rate of the discriminator [1e-4]")
flags.DEFINE_float("learning_rate_g", .0001, "Learning rate of the generator [1e-4]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_float("lipschitz_penalty", 10.0, "Weight of lipschitz penelty")
flags.DEFINE_float("gradient_penalty", 10.0, "Weight of gradient penelty")
flags.DEFINE_string("log_dir", "logs", "Directory name for summary logs [logs]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("print_interval", 10, "How many iterations before printing new infromation [10]")
flags.DEFINE_integer("num_sample_batches", 10, "How many batches to sample for JSD (important since JSD estimation is biased) [10]")
flags.DEFINE_integer("jsd_test_interval", 100, "Number of training iters between JSD calculations [100]")
flags.DEFINE_boolean("use_fast_lang_model", False, "Use the faster C++ language model (must be compiled first) [False]")

FLAGS = flags.FLAGS


def main(_):
  
  assert FLAGS.gan_divergence in ['WGANGP', 'FOGAN', 'PWGAN'], "FLAGS.gan_divergence must be either 'WGANGP', 'PWGAN' or 'FOGAN'"
  assert FLAGS.activation_d in ['relu', 'elu'], "FLAGS.gan_divergence must be either 'relu' or 'elu'"
  
  timestamp = time.strftime("%m%d_%H%M%S")
  DIR = "/%s_%6f_%.6f" % (timestamp, FLAGS.learning_rate_d, FLAGS.learning_rate_g)
  FLAGS.log_dir += DIR
  FLAGS.sample_dir += DIR
  FLAGS.checkpoint_dir += DIR
  
  pp.pprint(flags.FLAGS.__flags)
  # Create directories if necessary
  if not os.path.exists(FLAGS.log_dir):
    print("*** create log dir %s" % FLAGS.log_dir)
    os.makedirs(FLAGS.log_dir)
  if not os.path.exists(FLAGS.sample_dir):
    print("*** create sample dir %s" % FLAGS.sample_dir)
    os.makedirs(FLAGS.sample_dir)
  if not os.path.exists(FLAGS.checkpoint_dir):
    print("*** create checkpoint dir %s" % FLAGS.checkpoint_dir)
    os.makedirs(FLAGS.checkpoint_dir)

  # Write flags to log dir
  flags_file = open("%s/flags.txt" % FLAGS.log_dir, "w")
  for k, v in flags.FLAGS.__flags.items():
        line = '{}, {}'.format(k, v)
        print(line, file=flags_file)
  flags_file.close()

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    fogan = FOGAN(sess,
                  iterations=FLAGS.iterations,
                  gan_divergence=FLAGS.gan_divergence,
                  squared_divergence=FLAGS.squared_divergence,
                  data_dir=FLAGS.data_path,
                  seq_len=FLAGS.seq_len,
                  max_n_examples=FLAGS.max_n_examples,
                  n_ngrams=FLAGS.n_ngrams,
                  batch_size=FLAGS.batch_size,
                  dim=FLAGS.dim,
                  activation_d=FLAGS.activation_d,
                  batch_norm_d=FLAGS.batch_norm_d,
                  batch_norm_g=FLAGS.batch_norm_g,
                  lipschitz_penalty=FLAGS.lipschitz_penalty,
                  critic_iters=FLAGS.critic_iters,
                  lr_disc=FLAGS.learning_rate_d,
                  lr_gen=FLAGS.learning_rate_g,
                  sample_dir=FLAGS.sample_dir,
                  log_dir=FLAGS.log_dir,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  print_interval=FLAGS.print_interval,
                  num_sample_batches=FLAGS.num_sample_batches,
                  jsd_test_interval=FLAGS.jsd_test_interval,
                  use_fast_lang_model=FLAGS.use_fast_lang_model,
                  gradient_penalty=FLAGS.gradient_penalty)
    
    if FLAGS.is_train:
      fogan.train()
    else:
      if not fogan.load(FLAGS.checkpoint_dir):
        raise Exception("[!] Train a model first, then run test mode")

if __name__ == '__main__':
  tf.app.run()
