import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def smooth(inp):
  assert len(inp.shape) == 2
  output = np.empty(inp.shape)
  for i in range(inp.shape[0]):
    output[i] = np.mean(inp[max(i-3,0):i+1,:], axis=0)
  return output

def extract_one_file(filename, starting_key, iter_key):
  if not os.path.isfile(filename):
    raise IOError("file '" + str(filename) + "' does not exist")
  output = []
  wallclock = [0.]
  divergence = []
  with open(filename, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    while True:
      try:
        line = reader.next()
        if line[0]=='train' and line[1] == 'disc':
          divergence[-1].append(-1 * float(line[3]))
        if line[0]==iter_key:
          wallclock[-1] += float(line[5])
        if line[0]==starting_key:
          output.append([])
          if len(divergence) > 0:
            divergence[-1] = np.mean(np.array(divergence[-1]))
          divergence.append([])
          wallclock.append(wallclock[-1])
          for i in range(len(line)-1):
            try:
              output[-1].append(float(line[i+1]))
            except ValueError:
              print('strange line = ', line[i+1], type(line[i+1]))
      except csv.Error:
        pass
      except StopIteration:
        break
  return smooth(np.array(output)), wallclock[0:len(output)], divergence[0:-1]

def wall(js, wallclock, divergence, js_index, mode):
  if mode == 'walltime':
    x = wallclock
    y = js[:,js_index-1]
  elif mode == 'iterations':
    x = np.arange(len(js[:,js_index-1]))
    y = js[:,js_index-1]
  elif mode == 'divergence':
    x = np.arange(len(divergence))
    y = divergence
  else:
    raise ValueError('unknow mode = ', mode)
  return x,y

def walltime_interpolate(all_x, all_y, mode):
  if mode=='iterations':
    out_x = all_x[0,:]
    for i in range(all_x.shape[0]):
      assert np.max(np.abs(out_x - all_x[i,:]))==0
    return out_x, all_y

  shortest_index = np.argmin(all_x[:,all_x.shape[1]-1])
  shortest_time  = all_x[shortest_index, all_x.shape[1]-1]
  out_x = np.linspace(0, shortest_time, int(shortest_time / 20))
  out_y = np.empty((all_y.shape[0], len(out_x)), dtype=np.float32)
  for i in range(out_y.shape[0]):
    out_y[i,:] = np.interp(out_x, all_x[i,:], all_y[i,:])
  
  return out_x / 3600., out_y

def plot(js_index, mode, xlim=None, ylim=None):
  legends = []
  colors = ['b','r','g','k','m','c','y']

  key = 'JS='
  iter_key = 'iteration'
  '''
  js, wallclock = extract_one_file('log_ttur_1.txt', key, iter_key)
  x = wall(wallclock, walltime)
  l, = plt.plot(x, js[:,js_index-1], linewidth=0.5, label='ttur', color=colors[0])
  legends.append(l)

  js, wallclock = extract_one_file('log_fogan_3.txt', key, iter_key)
  x = wall(wallclock, walltime)
  l, = plt.plot(x, js[:,js_index-1], linewidth=0.5, label='fogan_1', color=colors[1])
  legends.append(l)

  js, wallclock = extract_one_file('log_fogan_2.txt', key, iter_key)
  x = wall(wallclock, walltime)
  l, = plt.plot(x, js[:,js_index-1], linewidth=0.5, label='fogan_2', color=colors[2])
  legends.append(l)
  '''
  
  #js, wallclock, divergence = extract_one_file('log_ttur_batch_norm_1.txt', key, iter_key)
  #x,y = wall(js, wallclock, divergence, js_index, mode)
  #l, = plt.plot(x, y, linewidth=0.5, label='ttur_bn', color=colors[3])
 
  js, wallclock, divergence = extract_one_file('log_ttur_batch_norm_3.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  all_x = np.empty((5, len(x)), dtype=np.float32)
  all_y = np.empty((5, len(y)), dtype=np.float32)
  all_y[0,:] = y
  all_x[0,:] = x

  js, wallclock, divergence = extract_one_file('log_ttur_batch_norm_4.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  all_y[1,:] = y
  all_x[1,:] = x

  js, wallclock, divergence = extract_one_file('log_ttur_batch_norm_5.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  all_y[2,:] = y
  all_x[2,:] = x

  js, wallclock, divergence = extract_one_file('log_ttur_batch_norm_6.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  all_y[3,:] = y
  all_x[3,:] = x

  js, wallclock, divergence = extract_one_file('log_ttur_batch_norm_7.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  all_y[4,:] = y
  all_x[4,:] = x

  x, all_y = walltime_interpolate(all_x, all_y, mode)

  for i in range(all_x.shape[0]):
    l, = plt.plot(x, all_y[i,:], linewidth=0.25, label='ttur_bn', color=colors[1])

  y_mean = np.mean(all_y, axis=0)
  y_std = np.std(all_y, axis=0)
  
  plt.fill_between(x, y_mean-2*y_std, y_mean+2*y_std, linewidth=0, color=colors[1], alpha=0.25)
  l, = plt.plot(x, y_mean, linewidth=1, label='WGAN-GP', color=colors[1])
  legends.append(l)

  '''
  js, wallclock, divergence = extract_one_file('log_fogan_equal_lr_1.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  l, = plt.plot(x, y, linewidth=0.5, label='fogan_equal', color=colors[0])
  legends.append(l)

  js, wallclock, divergence = extract_one_file('log_wgan-gp_equal_lr_1.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  l, = plt.plot(x, y, linewidth=0.5, label='wgan_equal', color=colors[1])
  legends.append(l)
  '''

#  js, wallclock, divergence = extract_one_file('log_fogan_2.txt', key, iter_key)
#  x,y = wall(js, wallclock, divergence, js_index, mode)
#  l, = plt.plot(x, y, linewidth=0.5, label='fogan', color=colors[2])

  js, wallclock, divergence = extract_one_file('log_fogan_5.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  all_x = np.empty((5, len(x)), dtype=np.float32)
  all_y = np.empty((5, len(y)), dtype=np.float32)
  all_x[0,:] = x
  all_y[0,:] = y

  js, wallclock, divergence = extract_one_file('log_fogan_6.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  all_x[1,:] = x
  all_y[1,:] = y

  js, wallclock, divergence = extract_one_file('log_fogan_7.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  all_x[2,:] = x
  all_y[2,:] = y

  js, wallclock, divergence = extract_one_file('log_fogan_8.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  all_x[3,:] = x
  all_y[3,:] = y

  js, wallclock, divergence = extract_one_file('log_fogan_9.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  all_x[4,:] = x
  all_y[4,:] = y

  #js, wallclock, divergence = extract_one_file('log_fogan_10.txt', key, iter_key)
  #x,y = wall(js, wallclock, divergence, js_index, mode)
  #all_y[5,:] = y
  #l, = plt.plot(x, y, linewidth=0.25, label='fogan', color=colors[2])

  x, all_y = walltime_interpolate(all_x, all_y, mode)

  for i in range(all_x.shape[0]):
    l, = plt.plot(x, all_y[i,:], linewidth=0.25, label='FOGAN', color=colors[2])

  y_mean = np.mean(all_y, axis=0)
  y_std = np.std(all_y, axis=0)
  
  plt.fill_between(x, y_mean-2*y_std, y_mean+2*y_std, linewidth=0, color=colors[2], alpha=0.25)
  l, = plt.plot(x, y_mean, linewidth=1, label='FOGAN', color=colors[2])
  legends.append(l)


  '''
  js, wallclock, divergence = extract_one_file('log_fogan_dynamic_1.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  l, = plt.plot(x, y, linewidth=0.5, label='fogan_dynamic', color=colors[4])
  legends.append(l)

  js, wallclock, divergence = extract_one_file('log_fogan_adam_25_08__1.txt', key, iter_key)
  x,y = wall(js, wallclock, divergence, js_index, mode)
  l, = plt.plot(x, y, linewidth=0.5, label='fogan_adam', color=colors[5])
  legends.append(l)

  #js, wallclock, divergence = extract_one_file('log_fogan_dynamic_squared_divergence_1', key, iter_key)
  #x,y = wall(js, wallclock, divergence, js_index, mode)
  #l, = plt.plot(x, y, linewidth=0.5, label='fogan_sq_d', color=colors[6])
  #legends.append(l)
  '''
  if xlim:
    plt.xlim(xlim)
  if ylim:
    plt.ylim(ylim)
  if xlim and ylim:
    plt.axes().set_aspect(.4 * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))

  plt.legend(handles=legends)
  #plt.xlim([0,50])
  #plt.ylim([10,40])
  plt.grid()
  if mode == 'walltime':
    plt.xlabel('wallclock time in hours')
    plt.ylabel('jsd' + str(js_index) + ' on 100x64 samples')
    plt.savefig('new_walltime_js' + str(js_index) + '_zoom.pdf', bbox_inches='tight')
  if mode == 'iterations':
    plt.xlabel('mini-batch x 2K')
    plt.ylabel('jsd' + str(js_index) + ' on 100x64 samples')
    plt.savefig('new_iter_js' + str(js_index) + '_zoom.pdf', bbox_inches='tight')
  if mode == 'divergence':
    plt.xlabel('mini-batch x 2K')
    plt.ylabel('divergence')
    plt.savefig('new_divergence_stuff.pdf', bbox_inches='tight')
  plt.clf()
  #plt.savefig('big_start_plot.pdf', dpi=400)

if __name__=='__main__':
  #plot(4, 'walltime', [0,30.5], [.525, 1.])
  #plot(6, 'walltime', [0,30.5], [.525, 1.])
  #plot(4, 'iterations', [0,250], [.525, 1.])
  #plot(6, 'iterations', [0,250], [.525, 1.])
  plot(4, 'walltime', [6,30.5], [.525, .675])
  plot(6, 'walltime', [6,30.5], [.525, .675])
  plot(4, 'iterations', [50,250], [.525, .675])
  plot(6, 'iterations', [50,250], [.525, .675])
  #plot(6, 'divergence')

