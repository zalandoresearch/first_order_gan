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


def make_noise(shape):
    return tf.random_normal(shape)


class FOGAN(object):
  def __init__(self, session,
               iterations, data_dir,
               seq_len, max_n_examples,
               n_ngrams, batch_size,
               dim, lambda_,
               critic_iters, lr_disc,
               lr_gen, sample_dir,
               log_dir, checkpoint_dir,
               print_interval):
    self.session=session
    self.iterations=iterations
    self.data_dir=data_dir
    self.seq_len=seq_len
    self.max_n_examples=max_n_examples
    self.n_ngrams=n_ngrams
    self.batch_size=batch_size
    self.dim=dim
    self.lambda_=lambda_
    self.critic_iters=critic_iters
    self.lr_disc=lr_disc
    self.lr_gen=lr_gen
    self.sample_dir=sample_dir
    self.log_dir=log_dir
    self.checkpoint_dir=checkpoint_dir
    self.print_interval=print_interval
    
    self.line=None
    self.charmap=None
    self.inv_charmap=None
    
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
    self.true_char_ngram_lms = []
    for i in range(self.n_ngrams):
      print(i, end=" ", flush=True)
      self.true_char_ngram_lms.append(language_helpers.NgramLanguageModel(i+1, self.lines, tokenize=False))
    print()


  def build_model(self):

    print("build model...")
    
    # Inputs
    self.real_inputs_discrete = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_len])
    real_inputs = tf.one_hot(self.real_inputs_discrete, len(self.charmap))
    self.fake_inputs = self.Generator(self.batch_size)
    
    # Disc prop
    disc_real = self.Discriminator(real_inputs)
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

    # WGAN lipschitz-penalty
    alpha = tf.random_uniform(
        shape=[self.batch_size,1,1],
        minval=0.,
        maxval=1.
    )
    differences = self.fake_inputs - real_inputs
    interpolates = real_inputs + (alpha*differences)
    gradients = tf.gradients(self.Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    self.disc_cost += self.lambda_*gradient_penalty

    disc_cost_opt_sum = tf.summary.scalar("bill disc cost opt", self.disc_cost)

    disc_cost_sum_op = tf.summary.merge([disc_cost_sum, disc_cost_opt_sum])
    gen_cost_sum_op = gen_cost_sum

    # Params
    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')

    self.disc_sums_op = disc_cost_sum_op

    # Merge gen summaries
    self.gen_sums_op = gen_cost_sum_op

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
    samples = self.session.run(self.fake_inputs)
    samples = np.argmax(samples, axis=2)
    decoded_samples = []
    for i in range(len(samples)):
        decoded = []
        for j in range(len(samples[i])):
            decoded.append(self.inv_charmap[samples[i][j]])
        decoded_samples.append(tuple(decoded))
    return decoded_samples


  def train(self):

    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    sum_writer = tf.summary.FileWriter(self.log_dir, self.session.graph)

    gen = self.inf_train_gen()
    
    _disc_cost_sum = 0
    _gen_cost_sum = 0
    time_sum = 0
    train_time_sum = 0
    
    iteration = 0
    # Run
    try:
      while(iteration < self.iterations):

        start_time = time.time()

        # Generate samples and eval JSDs
        if iteration % 100 == 0:
            samples = []
            for i in range(10):
                samples.extend(self.generate_samples())

            js = []
            js_string = ''
            for i in range(self.n_ngrams):
                lm = language_helpers.NgramLanguageModel(i+1, samples, tokenize=False)
                js.append(lm.js_with(self.true_char_ngram_lms[i]))
                js_string += ' ' + str(js[i])
            print('JS=' + js_string)
            feed_dict = {k: v for k, v in zip(self.js_ph, js)}
            js_sum = self.session.run(self.js_sum_op, feed_dict=feed_dict)
            sum_writer.add_summary(js_sum, iteration)

            with open('%s/samples_%d.txt' % (self.sample_dir, iteration + 1), 'w', encoding='utf-8') as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")

        
        train_start_time = time.time()
        # Train generator
        if iteration > 0:
          summary_string, _ = self.session.run([self.gen_sums_op, self.gen_train_op])
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
        _disc_cost_sum += _disc_cost
        sum_writer.add_summary(summary_string, iteration)

        time_sum += time.time() - start_time
        train_time_sum += time.time() - train_start_time
        if iteration % self.print_interval == 0:
          print('iteration '+ str(iteration) + \
                ' time ' + str(time_sum / self.print_interval) + \
                ' train_time ' + str(train_time_sum / self.print_interval))
          print('train disc cost', _disc_cost_sum / (self.print_interval * self.critic_iters), flush=True)
          
          time_sum = 0
          train_time_sum = 0
          _disc_cost_sum=0

        if iteration > 0 and iteration % 1000 == 0:
          saver.save(session, '%s/model' % (self.checkpoint_dir, ), global_step=iteration)
        
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

  def ResBlock(self, name, inputs):
      output = inputs
      output = tf.nn.relu(output)
      output = lib.ops.conv1d.Conv1D(name+'.1', self.dim, self.dim, 5, output)
      output = tf.nn.relu(output)
      output = lib.ops.conv1d.Conv1D(name+'.2', self.dim, self.dim, 5, output)
      return inputs + (0.3 * output)

  def Generator(self, n_samples, prev_outputs=None):
      output = make_noise(shape=[n_samples, 128])
      output = lib.ops.linear.Linear('Generator.Input', 128, self.seq_len*self.dim, output)
      output = tf.reshape(output, [-1, self.dim, self.seq_len])
      output = self.ResBlock('Generator.1', output)
      output = self.ResBlock('Generator.2', output)
      output = self.ResBlock('Generator.3', output)
      output = self.ResBlock('Generator.4', output)
      output = self.ResBlock('Generator.5', output)
      output = lib.ops.conv1d.Conv1D('Generator.Output', self.dim, len(self.charmap), 1, output)
      output = tf.transpose(output, [0, 2, 1])
      output = self.softmax(output)
      return output

  def Discriminator(self, inputs):
      output = tf.transpose(inputs, [0,2,1])
      output = lib.ops.conv1d.Conv1D('Discriminator.Input', len(self.charmap), self.dim, 1, output)
      output = self.ResBlock('Discriminator.1', output)
      output = self.ResBlock('Discriminator.2', output)
      output = self.ResBlock('Discriminator.3', output)
      output = self.ResBlock('Discriminator.4', output)
      output = self.ResBlock('Discriminator.5', output)
      output = tf.reshape(output, [-1, self.seq_len * self.dim])
      output = lib.ops.linear.Linear('Discriminator.Output', self.seq_len * self.dim, 1, output)
      return output
