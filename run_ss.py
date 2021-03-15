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

TRAIN_WITH_GEN = False

input_shape = None
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
elif(DATASET=='oxford_pets'):
  input_dir = "../data/ss/images/"
  target_dir = "../data/ss/annotations/trimaps/"
  img_size = (160, 160)
  num_classes = 3
  batch_size = 64
  from oxford_pets import OxfordPets,get_img_paths
  input_img_paths,target_img_paths = get_img_paths(input_dir,target_dir)
  print("Number of samples:", len(input_img_paths))   
  # Split our img paths into a training and a validation set
  val_samples = 1000
  random.Random(1337).shuffle(input_img_paths)
  random.Random(1337).shuffle(target_img_paths)
  train_input_img_paths = input_img_paths[:-val_samples]
  train_target_img_paths = target_img_paths[:-val_samples]
  val_input_img_paths = input_img_paths[-val_samples:]
  val_target_img_paths = target_img_paths[-val_samples:]

  # Instantiate data Sequences for each split
  train_gen = OxfordPets(
      batch_size, img_size, train_input_img_paths, train_target_img_paths
  )
  val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
  dataset = (train_gen,val_gen)
  input_shape = (160,160,3)
  TRAIN_WITH_GEN = True
else:
  raise Exception('Only mnist,fmnist and cifar10 are currently supported!')

if(not TRAIN_WITH_GEN):
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  dataset = ((x_train, x_train), (x_test, x_test))
  input_shape = x_train.shape[1:]
  

def calculate_contrib_hvi(objectives):
    hv = pygmo.hypervolume(objectives) 
    reference_points = [10,10,10]
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
                                  input_shape=input_shape,
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
                          train_with_gen=TRAIN_WITH_GEN,
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