from __future__ import print_function
import sys
sys.path.append("..")

import pygmo
import random
import pickle
import math
import traceback
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--pop_size")
parser.add_argument("--generations")
parser.add_argument("--number_of_runs", )
args = parser.parse_args()


DATASET = args.dataset
if(args.pop_size):
  POP_SIZE = int(args.pop_size)
else:
  POP_SIZE = 20

if(args.pop_size):
  GENERATIONS = int(args.generations)
else:
  GENERATIONS = 20
  
if(args.number_of_runs):
  NUMBER_OF_RUNS = int(args.number_of_runs)
else:
  NUMBER_OF_RUNS = 10


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
# Verbosity is now 0

physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  print("Invalid device or cannot modify virtual devices once initialized.")
  pass
from tensorflow.keras.datasets import mnist,cifar10,fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras import backend as K
from ne import Devol_based_NE, GenomeHandler

# **Prepare dataset**

K.set_image_data_format("channels_last")
if(DATASET=='mnist'):
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
elif(DATASET=='fmnist'):
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
elif(DATASET=='cifar10'):
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train = x_train.reshape(x_train.shape[0], 32, 32, 3).astype('float32') / 255
  x_test = x_test.reshape(x_test.shape[0], 32, 32, 3).astype('float32') / 255
else:
  raise Exception('Only mnist,fmnist and cifar10 are currently supported!')
# SPECIFY THE DATASET HERE! The Reshape dimensions have to be adjusted also!
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
dataset = ((x_train, x_train), (x_test, x_test))



def calculate_contrib_hvi(objectives):
    hv = pygmo.hypervolume(objectives) 
    reference_points = [4,8]
    hv.compute(reference_points)
    return hv.contributions(reference_points)


seeds_used = []
result = None
for run_num in range(NUMBER_OF_RUNS):
  seed = random.randint(1,100)
  seeds_used.append(seed)
  np.random.seed(0)
  np.random.seed(seed)
  tf.random.set_seed(seed)

  for exp_setting in ['hvi']:
    for eval_method in ['OG']:
      genome_handler = GenomeHandler(max_conv_layers=6, 
                                  max_filters=256,
                                  input_shape=x_train.shape[1:],
                                  n_classes=10,
                                  dropout=False)
      neuroevolution = Devol_based_NE(genome_handler, 
                        experiment_run = run_num,
                        experiment_setting = exp_setting,
                        evaluation_method=eval_method
                      )
      try:
        result = neuroevolution.run(dataset=dataset,
                          num_generations=GENERATIONS,
                          pop_size=POP_SIZE,
                          epochs=5,
                          fitness = calculate_contrib_hvi,
                          metric='hvi'
                          )
      except Exception as e:
        print(e)
        traceback.print_exc()
      finally:
        report = result + (seed,)
        with open(exp_setting+'-'+eval_method+'-'+str(run_num)+'.pkl', 'wb') as f:
          pickle.dump(report, f)
        del result
        del genome_handler
        del neuroevolution
with open('seeds_used.pkl', 'wb') as f:
    pickle.dump(seeds_used, f)