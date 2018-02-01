import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import language_helpers
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot

BN_SCALE = False
def make_noise(shape):
    return tf.random_normal(shape)
  
def l2norm(x, axis=[1,2]):
    return tf.squeeze(tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis)))
  
class batch_norm_class(object):
  def __init__(self, scale=True, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name
      self.scale = scale

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=self.scale,
                      is_training=train,
#                      reuse=REUSE_BN,
                      scope=self.name)


  
def text_to_array(lines, charmap):
  answer = np.empty((len(lines), len(lines[0])),dtype=np.int32)
  for i,l in enumerate(lines):
    assert answer.shape[1] == len(l), 'all lines must have same length'
    for j,c in enumerate(l):
      answer[i,j] = charmap[c]
  return answer

def smooth_relu(x, other_args=None):
  x = tf.nn.relu(x)
  return tf.where(tf.less(x, 1 / 2000. * tf.ones_like(x,dtype=tf.float32)), 1000 * tf.square(x), x - 1 / 4000.)

class FOGAN(object):
  def __init__(self, session,
               gan_divergence,
               squared_divergence,
               iterations, data_dir,
               seq_len, max_n_examples,
               n_ngrams, batch_size,
               dim, activation_d, 
               lipschitz_penalty, batch_norm_d,
               batch_norm_g,
               critic_iters, lr_disc,
               lr_gen, sample_dir,
               log_dir, checkpoint_dir,
               print_interval,
               num_sample_batches,
               jsd_test_interval,
               use_fast_lang_model,
               gradient_penalty):
    self.session=session
    self.iterations=iterations
    self.gan_divergence=gan_divergence
    self.squared_divergence=squared_divergence
    self.data_dir=data_dir
    self.seq_len=seq_len
    self.max_n_examples=max_n_examples
    self.n_ngrams=n_ngrams
    self.batch_size=batch_size
    self.dim=dim
    self.lipschitz_penalty=lipschitz_penalty
    self.critic_iters=critic_iters
    self.lr_disc=lr_disc
    self.lr_gen=lr_gen
    self.sample_dir=sample_dir
    self.log_dir=log_dir
    self.checkpoint_dir=checkpoint_dir
    self.print_interval=print_interval
    self.num_sample_batches=num_sample_batches
    self.jsd_test_interval=jsd_test_interval
    self.use_fast_lang_model=use_fast_lang_model
    self.gradient_penalty=gradient_penalty
    if activation_d == "relu":
      self.activation_d=tf.nn.relu
    elif activation_d == "elu":
      self.activation_d=tf.nn.elu
    elif activation_d == "selu":
      self.activation_d=smooth_relu
    else:
      raise ValueError("invalid activation_d")
    def stuff1(x, train=None):
      return x
    
    def stuff2(scale, name): 
        return stuff1
      
    if batch_norm_d:
      self.batch_norm_d = batch_norm_class
    else:
      self.batch_norm_d = stuff2
    
    if batch_norm_g:
      self.batch_norm_g = batch_norm_class
    else:
      self.batch_norm_g = stuff2
    
    
    self.line=None
    self.charmap=None
    self.inv_charmap=None
    
    if self.use_fast_lang_model:
      import ngram_language_model.ngram_language_model as language_model
      self.NgramLanguageModel = language_model.NgramLanguageModel
    else:
      self.NgramLanguageModel = language_helpers.NgramLanguageModel
    self.load_data()
    self.build_model()
    self.generate_lang_model()
  
  def load_data(self):
    if len(self.data_dir) == 0:
      raise Exception('Please specify path to data directory in gan_language.py!')
    
    # Load data
    self.lines, self.charmap, self.inv_charmap = language_helpers.load_dataset(
      max_length=self.seq_len,
      max_n_examples=self.max_n_examples,
      data_dir=self.data_dir
    )
  
  def generate_lang_model(self):
    print("true char ngram lms:", end=" ", flush=True)
    if self.use_fast_lang_model:
      self.true_char_ngram_lms=self.NgramLanguageModel(text_to_array(self.lines, self.charmap), self.n_ngrams, len(self.charmap))
    else:
      self.true_char_ngram_lms = []
      for i in range(self.n_ngrams):
        print(i, end=" ", flush=True)
        self.true_char_ngram_lms.append(self.NgramLanguageModel(i+1, self.lines, tokenize=False))
    print()


  def build_model(self):

    print("build model...")
    
    # Inputs
    self.real_inputs_discrete = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_len])
    real_inputs = tf.one_hot(self.real_inputs_discrete, len(self.charmap))
    with tf.variable_scope("Generator.") as scope:
      self.fake_inputs = self.Generator(self.batch_size, train=True)
    with tf.variable_scope("Generator.") as scope:
      scope.reuse_variables()
      self.fake_samples = self.Generator(self.batch_size, train=False)
    # Disc prop
    with tf.variable_scope("Discriminator.") as scope:
      disc_real = self.Discriminator(real_inputs)
    if self.gan_divergence == 'PWGAN' or self.gan_divergence == 'FOGAN':
      # WGAN lipschitz-penalty
      alpha = tf.random_uniform(
          shape=[self.batch_size,1,1],
          minval=0.99,
          maxval=1.0
      )
      differences = self.fake_inputs - real_inputs
      interpolates = real_inputs + (alpha*differences)
      with tf.variable_scope("Discriminator.") as scope:
        scope.reuse_variables()
        disc_fake = self.Discriminator(interpolates)
    else:
      with tf.variable_scope("Discriminator.") as scope:
        scope.reuse_variables()
        disc_fake = self.Discriminator(self.fake_inputs)


    # Costs & summaries
    self.disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    self.gen_cost  = -tf.reduce_mean(disc_fake)

    disc_cost_sum = tf.summary.scalar("bill disc cost ws", self.disc_cost)
    gen_cost_sum  = tf.summary.scalar("bill gen cost", self.gen_cost)

    # JSD summaries
    self.js_ph = []
    for i in range(self.n_ngrams):
      self.js_ph.append(tf.placeholder(tf.float32, shape=()))

    js_sums = []
    for i in range(self.n_ngrams):
      js_sums.append(tf.summary.scalar("bill js%d" % (i + 1), self.js_ph[i]))

    self.js_sum_op = tf.summary.merge(js_sums)

    if self.gan_divergence == 'WGANGP':
      # WGAN lipschitz-penalty
      alpha = tf.random_uniform(
          shape=[self.batch_size,1,1],
          minval=0.,
          maxval=1.
      )
      differences = self.fake_inputs - real_inputs
      interpolates = real_inputs + (alpha*differences)
      with tf.variable_scope("Discriminator.") as scope:
        scope.reuse_variables()
        disc_int = self.Discriminator(interpolates)
      gradients = tf.gradients(disc_int, [interpolates])[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
      gradient_penalty = tf.reduce_mean((slopes-1.)**2)
      self.disc_cost += self.gradient_penalty*gradient_penalty
    
    if self.gan_divergence in ['PWGAN', 'FOGAN']:
      top = tf.squeeze(disc_real - disc_fake)
      if self.squared_divergence:
        bot = tf.squeeze(tf.reduce_sum(tf.square(real_inputs - interpolates), axis=[1,2]))
      else:
        bot = tf.squeeze(tf.sqrt(tf.reduce_sum(tf.square(real_inputs - interpolates), axis=[1,2])))
      
      diff_penalty = tf.where(tf.less(bot, 10e-9 * tf.ones_like(top,dtype=tf.float32)), tf.zeros_like(top,dtype=tf.float32), tf.square(top) / bot)
      self.disc_cost += self.lipschitz_penalty * tf.reduce_mean(diff_penalty)

      if self.gan_divergence == 'FOGAN':
        # this whole bit calculates the formula
        #\frac{
        #\left\Vert\mathbb E_{\tilde x\sim\mathbb P}[(\tilde x- x')\frac{f(\tilde x)-f(x')}{\Vert x'-\tilde x\Vert^3}]\right\Vert}
        #{\mathbb E_{\tilde x\sim\mathbb P}[\frac{1}{\Vert x'-\tilde x\Vert}]})
        #
        # by doing a mapping that (batch wise) outputs everything that needs to be summed up to calculate the expected
        # value in both the bottom and top part of the formula
        def map_function(map_pack):
          map_G_interpolate, map_g_logits_interpolate = map_pack
          map_input_difference = real_inputs - map_G_interpolate
          if self.squared_divergence:
            map_norm = tf.reduce_sum(tf.square(map_input_difference), axis=[1,2]) + 10e-7
            map_bot = tf.squeeze(tf.pow(map_norm , -2))
          else:
            map_norm = l2norm(map_input_difference) + 10e-7
            map_bot = tf.squeeze(tf.pow(map_norm , -3))
          map_top = tf.squeeze(disc_real - map_g_logits_interpolate)
          first_output = map_input_difference * tf.reshape(map_top * map_bot, [self.batch_size, 1, 1])
          first_output = tf.norm(tf.reduce_mean(first_output, axis=[0]))

          second_output = tf.reduce_mean(tf.pow(map_norm , -1))

          return first_output / second_output
        
        d_slope_target = tf.map_fn(map_function,
                                   (tf.stop_gradient(interpolates), tf.stop_gradient(disc_fake)),
                                   back_prop=False,
                                   dtype=tf.float32)
        
        gradients = tf.gradients(disc_fake, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
        #sd = tf.abs(slopes - d_slope_target)
        #ss = tf.where(tf.less(sd, .5 * tf.ones_like(top,dtype=tf.float32)), tf.square(sd), sd-.25)
        self.disc_cost += self.gradient_penalty * tf.reduce_mean(tf.square(slopes - d_slope_target))
        #self.disc_cost += self.gradient_penalty * tf.reduce_mean(ss)


    disc_cost_opt_sum = tf.summary.scalar("bill disc cost opt", self.disc_cost)

    disc_cost_sum_op = tf.summary.merge([disc_cost_sum, disc_cost_opt_sum])
    gen_cost_sum_op = gen_cost_sum

    # Params
    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')

    self.disc_sums_op = disc_cost_sum_op

    # Merge gen summaries
    self.gen_sums_op = gen_cost_sum_op
    #self.gen_train_op = tf.train.AdagradOptimizer(learning_rate=self.lr_gen).minimize(self.gen_cost, var_list=gen_params)
    #self.gen_train_op = tf.train.MomentumOptimizer(learning_rate=self.lr_gen, momentum=0.9, use_nesterov=True).minimize(self.gen_cost, var_list=gen_params)
    #self.gen_train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr_gen).minimize(self.gen_cost, var_list=gen_params)
    #self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_gen, beta1=0.25, beta2=0.8).minimize(self.gen_cost, var_list=gen_params)
    self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_gen, beta1=0.5, beta2=0.9).minimize(self.gen_cost, var_list=gen_params)
    self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_disc, beta1=0.5, beta2=0.9).minimize(self.disc_cost, var_list=disc_params)

  # Dataset iterator
  def inf_train_gen(self):
    while True:
        np.random.shuffle(self.lines)
        for i in range(0, len(self.lines)-self.batch_size+1, self.batch_size):
            yield np.array(
                [[self.charmap[c] for c in l] for l in self.lines[i:i+self.batch_size]],
                dtype='int32'
            )

  def generate_samples(self):
    samples = self.session.run(self.fake_samples)
    samples = np.argmax(samples, axis=2)
    decoded_samples = []
    for i in range(len(samples)):
        decoded = []
        for j in range(len(samples[i])):
            decoded.append(self.inv_charmap[samples[i][j]])
        decoded_samples.append(tuple(decoded))
    return samples, decoded_samples


  def train(self):

    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    sum_writer = tf.summary.FileWriter(self.log_dir, self.session.graph)

    gen = self.inf_train_gen()

    _disc_cost_sum = 0
    _disc_cost = 0
    _gen_cost_sum = 0
    time_sum = 0
    train_time_sum = 0
    gen_update_sum = 0
    
    iteration = 0
    big_average_disc = np.zeros(200)
    small_average_disc = np.zeros(20)
    # Run
    try:
      while(iteration < self.iterations):

        start_time = time.time()

        # Generate samples and eval JSDs
        if iteration % self.jsd_test_interval == 0:
            encoded_samples = []
            decoded_samples = []
            for i in range(self.num_sample_batches):
                es, ds = self.generate_samples()
                encoded_samples.extend(es)
                decoded_samples.extend(ds)

            js = []
            js_string = ''
            if self.use_fast_lang_model:
              lm = self.NgramLanguageModel(encoded_samples, self.n_ngrams, len(self.charmap))
              for i in range(self.n_ngrams):
                js.append(lm.js_with(self.true_char_ngram_lms, i+1))
                js_string += ' ' + str(js[i])
              del lm
            else:
              for i in range(self.n_ngrams):
                  lm = self.NgramLanguageModel(i+1, decoded_samples, tokenize=False)
                  js.append(lm.js_with(self.true_char_ngram_lms[i]))
                  js_string += ' ' + str(js[i])
            print('JS=' + js_string)
            feed_dict = {k: v for k, v in zip(self.js_ph, js)}
            js_sum = self.session.run(self.js_sum_op, feed_dict=feed_dict)
            sum_writer.add_summary(js_sum, iteration)

            with open('%s/samples_%d.txt' % (self.sample_dir, iteration + 1), 'w', encoding='utf-8') as f:
                for s in decoded_samples:
                    s = "".join(s)
                    f.write(s + "\n")

        
        train_start_time = time.time()
        # Train generator
        if iteration > 0: # and _disc_cost < .95 * np.mean(big_average_disc) and np.mean(small_average_disc) < .95 * np.mean(big_average_disc):
          if self.gan_divergence in ['PWGAN', 'FOGAN']:
            _data = gen.__next__()
            summary_string, _ = self.session.run([self.gen_sums_op, self.gen_train_op],
                                                 feed_dict={self.real_inputs_discrete:_data})
          else:
            summary_string, _ = self.session.run([self.gen_sums_op, self.gen_train_op])
          gen_update_sum += 1
          sum_writer.add_summary(summary_string, iteration)

        # Train critic
        for i in range(self.critic_iters - 1):
            _data = gen.__next__()
            _disc_cost, _ = self.session.run(
                [self.disc_cost, self.disc_train_op],
                feed_dict={self.real_inputs_discrete:_data}
            )
            _disc_cost_sum += _disc_cost
        _data = gen.__next__()
        _disc_cost, summary_string, _ = self.session.run(
            [self.disc_cost, self.disc_sums_op, self.disc_train_op],
            feed_dict={self.real_inputs_discrete:_data}
        )
        big_average_disc[iteration % len(big_average_disc)] = min(_disc_cost,0)
        small_average_disc[iteration % len(small_average_disc)] = _disc_cost
        _disc_cost_sum += _disc_cost
        sum_writer.add_summary(summary_string, iteration)

        time_sum += time.time() - start_time
        train_time_sum += time.time() - train_start_time
        if iteration % self.print_interval == 0:
          print('iteration '+ str(iteration) + \
                ' time ' + str(time_sum) + \
                ' train_time ' + str(train_time_sum) + \
                ' num_gen_updates = ' + str(gen_update_sum))
          print('train disc cost', _disc_cost_sum / (self.print_interval * self.critic_iters), flush=True)
          
          time_sum = 0
          train_time_sum = 0
          gen_update_sum = 0
          _disc_cost_sum=0

        if iteration > 0 and iteration % 10000 == 0:
          saver.save(self.session, '%s/model' % (self.checkpoint_dir, ), global_step=iteration)
        
        iteration += 1
    except KeyboardInterrupt:
      saver.save(self.session, '%s/model' % (self.checkpoint_dir, ), global_step=iteration)


  def softmax(self, logits):
      return tf.reshape(
          tf.nn.softmax(
              tf.reshape(logits, [-1, len(self.charmap)])
          ),
          tf.shape(logits)
      )

  def ResBlock(self, name, activation, inputs, bn1=tf.identity, bn2=tf.identity, train=True):
      output = inputs
      output = activation(output)
      output = bn1(lib.ops.conv1d.Conv1D(name+'.1', self.dim, self.dim, 5, output), train=train)
      output = activation(output)
      output = bn2(lib.ops.conv1d.Conv1D(name+'.2', self.dim, self.dim, 5, output), train=train)
      return inputs + (0.3 * output)

  def Generator(self, n_samples, prev_outputs=None, train=True):
      g_bn0 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn0')
      g_bn1 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn1')
      g_bn2 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn2')
      g_bn3 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn3')
      g_bn4 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn4')
      g_bn5 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn5')
      g_bn6 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn6')
      g_bn7 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn7')
      g_bn8 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn8')
      g_bn9 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn9')
      g_bn10 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn10')
      g_bn11 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn11')
      g_bn12 = self.batch_norm_g(scale=BN_SCALE, name='Generator.g_bn12')
    
      output = g_bn0(make_noise(shape=[n_samples, 128]))
      output = g_bn1(lib.ops.linear.Linear('Generator.Input', 128, self.seq_len*self.dim, output))
      output = tf.reshape(output, [-1, self.dim, self.seq_len])
      output = self.ResBlock('Generator.1', tf.nn.relu, output, g_bn2, g_bn3, train)
      output = self.ResBlock('Generator.2', tf.nn.relu, output, g_bn4, g_bn5, train)
      output = self.ResBlock('Generator.3', tf.nn.relu, output, g_bn6, g_bn7, train)
      output = self.ResBlock('Generator.4', tf.nn.relu, output, g_bn8, g_bn9, train)
      output = self.ResBlock('Generator.5', tf.nn.relu, output, g_bn10, g_bn11, train)
      output = lib.ops.conv1d.Conv1D('Generator.Output', self.dim, len(self.charmap), 1, output)
      output = tf.transpose(output, [0, 2, 1])
      output = self.softmax(output)
      return output

  def Discriminator(self, inputs):
      d_bn1 = self.batch_norm_d(scale=BN_SCALE, name='d_bn1')
      d_bn2 = self.batch_norm_d(scale=BN_SCALE, name='d_bn2')
      d_bn3 = self.batch_norm_d(scale=BN_SCALE, name='d_bn3')
      d_bn4 = self.batch_norm_d(scale=BN_SCALE, name='d_bn4')
      d_bn5 = self.batch_norm_d(scale=BN_SCALE, name='d_bn5')
      d_bn6 = self.batch_norm_d(scale=BN_SCALE, name='d_bn6')
      d_bn7 = self.batch_norm_d(scale=BN_SCALE, name='d_bn7')
      d_bn8 = self.batch_norm_d(scale=BN_SCALE, name='d_bn8')
      d_bn9 = self.batch_norm_d(scale=BN_SCALE, name='d_bn9')
      d_bn10 = self.batch_norm_d(scale=BN_SCALE, name='d_bn10')
      d_bn11 = self.batch_norm_d(scale=BN_SCALE, name='d_bn11')

    
      output = tf.transpose(inputs, [0,2,1])
      output = d_bn1(lib.ops.conv1d.Conv1D('Discriminator.Input', len(self.charmap), self.dim, 1, output))
      output = self.ResBlock('Discriminator.1', self.activation_d, output, d_bn2, d_bn3)
      output = self.ResBlock('Discriminator.2', self.activation_d, output, d_bn4, d_bn5)
      output = self.ResBlock('Discriminator.3', self.activation_d, output, d_bn6, d_bn7)
      output = self.ResBlock('Discriminator.4', self.activation_d, output, d_bn8, d_bn9)
      output = self.ResBlock('Discriminator.5', self.activation_d, output, d_bn10, d_bn11)
      output = tf.reshape(output, [-1, self.seq_len * self.dim])
      output = lib.ops.linear.Linear('Discriminator.Output', self.seq_len * self.dim, 1, output)
      return output
