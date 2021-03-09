"""
Run a genetic algorithm to find an appropriate architecture for some MONCAE task 
with Keras+TF.

To use, define a `GenomeHandler` defined in genomehandler.py. Then pass it, with
training data, to a DEvol instance to run the genetic algorithm. See the readme
for more detailed instructions.
"""

from __future__ import print_function
import tensorflow as tf
import random as rand
import csv
import operator
import math
import gc
import os
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.python.framework import ops
import sys 
sys.path.append("..")
from tqdm import tqdm
import pygmo

__all__ = ['Devol_based_NE']

METRIC_OPS = [operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [min, max]
HVIPOPULATION_EXPERIMENT_SETTINGS = ['HVI','HVI_SS']
SS_EVALUATION_EXPERIMENT_SETTINGS = ['KFold','HVI_SS']




class Devol_based_NE:
    """
    Object which carries out genetic search and returns top performing model
    upon completion.
    """

    def __init__(self, genome_handler, data_path="",experiment_run=0,experiment_setting='OG',evaluation_method='OG'):
        """
        Initialize a DEvol object which carries out the training and evaluation
        of a genetic search.

        Args:
            genome_handler (GenomeHandler): the genome handler object defining
                    the restrictions for the architecture search space
            data_path (str): the file which the genome encodings and metric data
                    will be stored in
        """
        
        self.genome_handler = genome_handler
        self.evaluation_method = evaluation_method
        self.experiment_run = experiment_run
        self.experiment_setting = experiment_setting
        self.batch_size = genome_handler.batch_size
        self.datafile = data_path or (datetime.now().isoformat()[:-10] + '.csv')
        self._indivual_id = 0
        self._bssf = -1

    def set_objective(self, metric):
        """
        Set the metric for optimization. Can also be done by passing to
        `run`.

        Args:
            metric (str): either 'acc' to maximize classification accuracy, or
                    else 'loss' to minimize the loss function
        """
        if metric not in ['loss', 'hvi']:
            raise ValueError(('Invalid metric name {} provided - should be'
                              '"accuracy" or "loss"').format(metric))
        self._metric = metric
        self._objective = "max" if self._metric == "hvi" else "min"
        #TODO currently loss and accuracy
        self._metric_index = 0 
        self._metric_op = METRIC_OPS[self._objective == 'max']
        self._metric_objective = METRIC_OBJECTIVES[self._objective == 'max']

    def run(self, dataset, num_generations, pop_size, epochs, fitness=None,
            metric='loss'):
        """
        Run genetic search on dataset given number of generations and
        population size

        Args:
            dataset : tuple or list of numpy arrays in form ((train_data,
                    train_labels), (validation_data, validation_labels))
            num_generations (int): number of generations to search
            pop_size (int): initial population size
            epochs (int): epochs for each model eval, passed to keras model.fit
            fitness (None, optional): scoring function to be applied to
                    population scores, will be called on a numpy array which is
                    a min/max scaled version of evaluated model metrics, so It
                    should accept a real number including 0. If left as default
                    just the min/max scaled values will be used.
            metric (str, optional): must be "accuracy" or "loss" , defines what
                    to optimize during search

        Returns:
            keras model: best model found with weights
        """
        import time
        generation_times = []
        generation_performances = []
        generation_members = []
        time_global_start = time.time()
        self.set_objective(metric)
        # If no validation data is given set it to None
        if len(dataset) == 2:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
            self.x_val = None
            self.y_val = None
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_val, self.y_val) = dataset
        self.x_train_full = self.x_train.copy()
        self.y_train_full = self.y_train.copy()
        time_start = time.time()
        # generate and evaluate initial population
        members = self._generate_random_population(pop_size)
        pop,objectives = self._evaluate_population(members,
                                        epochs,
                                        fitness,
                                        0,
                                        num_generations)
        time_finish = time.time()
        # evolve
        generation_performances.append(objectives)
        generation_members.append(pop.members)
        generation_times.append(time_finish-time_start)
        
        for gen in tqdm(range(1, num_generations)):
            time_start = time.time()
            members = self._reproduce(pop, gen)
            pop,objectives = self._evaluate_population(members,
                                            epochs,
                                            fitness,
                                            gen,
                                            num_generations)
            time_finish = time.time()
            generation_times.append(time_finish-time_start)
            generation_performances.append(objectives)
            generation_members.append(pop.members)
        time_global_finish = time.time()
        time_global = time_global_finish-time_global_start
        return (time_global,generation_times,generation_performances,generation_members)

    def _reproduce(self, pop, gen):
        members = []

        # 95% of population from crossover
        for _ in range(int(len(pop) * 0.95)):
            members.append(self._crossover(pop.select(), pop.select()))

        # best models survive automatically
        members += pop.get_best(len(pop) - int(len(pop) * 0.95))

        # randomly mutate
        for imem, mem in enumerate(members):
            members[imem] = self._mutate(mem, gen)
        return members
    
    def _evaluate(self, genome, epochs, igen, ngen):
        try:
            model,level_of_compression = self.genome_handler.decode(genome)
        except Exception as e:
            print(e)
        model,level_of_compression = self.genome_handler.decode(genome)
        loss, accuracy = None, None
        performance = []
        fit_params = {
            'x': self.x_train_full,
            'y': self.y_train_full,
            'validation_split': 0.1,
            'batch_size':self.batch_size,
            'shuffle':True,
            'steps_per_epoch': int(len(self.x_train_full)/self.batch_size),
            'epochs': epochs,
            'verbose': 0,
            'callbacks': [
                EarlyStopping(monitor='val_loss', patience=1, verbose=0)
            ]
        }

        if self.x_val is not None:
            fit_params['validation_data'] = (self.x_val, self.y_val)
        try:
            history = model.fit(**fit_params)
            performance = model.evaluate(self.x_test, self.y_test, verbose=0)
        except Exception as e:
            performance = self._handle_broken_model(model, e)

        if(igen+1==ngen):
            self._record_model(model,genome,performance)
        return model, performance , level_of_compression

    def _record_model(self,model,genome,performance):
        try:
            os.remove('best-model-'+str(self._indivual_id)+'-'+str(self.experiment_setting)+'-'+str(self.evaluation_method)+'-'+str(self.experiment_run)+'.h5')
        except OSError:
            pass
        model.save('best-model-'+str(self._indivual_id)+'-'+str(self.experiment_setting)+'-'+str(self.evaluation_method)+'-'+str(self.experiment_run)+'.h5')
        self._indivual_id+=1

    #TODO update!!!
    def _handle_broken_model(self, model, error):
        del model

        n = self.genome_handler.n_classes
        performance = [log_loss(np.concatenate(([1], np.zeros(n - 1))), np.ones(n) / n), math.log((self.genome_handler.input_shape[1]*self.genome_handler.input_shape[1]),10)]
        gc.collect()

        if K.backend() == 'tensorflow':
            K.clear_session()
            #Changed from tensorflow
            ops.reset_default_graph()

        print('An error occurred and the model could not train!')
        print(('Model assigned poor score. Please ensure that your model'
               'constraints live within your computational resources.'))
        return performance

    def _evaluate_population(self, members, epochs, fitness, igen, ngen):
        fit = []
        objectives = []
        #Look for res
        self._indivual_id=0
        for imem, mem in enumerate(members):
            self._print_evaluation(imem, len(members), igen, ngen)
            if(self.evaluation_method == "OG"):
                res = self._evaluate(mem, epochs,igen,ngen)
            else:
                raise Exception('No valid evaluation method specified')
            #KEY loss capped at 4
            v = min(res[1][self._metric_index],4)
            objectives.append((v,res[2]))
            del res
            fit.append(v)

        fit = np.array(fit)
        objs = np.array(objectives)
        self._print_result(fit, igen)
        return _Population(members, objs, fitness, obj=self._objective), objectives

    def _print_evaluation(self, imod, nmod, igen, ngen):
        fstr = '\nmodel {0}/{1} - generation {2}/{3}:\n'
        #TODO Currently prints disabled
        # print(fstr.format(imod + 1, nmod, igen + 1, ngen))

    def _generate_random_population(self, size):
        return [self.genome_handler.generate() for _ in range(size)]

    def _print_result(self, fitness, generation):
        result_str = ('Generation {3}:\t\tbest {4}: {0:0.4f}\t\taverage:'
                      '{1:0.4f}\t\tstd: {2:0.4f}')
        #TODO print currently disabled!
        # print(result_str.format(self._metric_objective(fitness),
        #                         np.mean(fitness),
        #                         np.std(fitness),
        #                         generation + 1, self._metric))

    def _crossover(self, genome1, genome2):
        cross_ind = rand.randint(0, len(genome1))
        child = genome1[:cross_ind] + genome2[cross_ind:]
        return child

    def _mutate(self, genome, generation):
        # increase mutations as program continues
        num_mutations = max(3, generation // 4)
        return self.genome_handler.mutate(genome, num_mutations)

    def _mutate_params(self,model,mut_type='random_mask'):
        #Variate population
        if(mut_type=='random'):
            for layer in model.layers:
                weights = layer.get_weights()
                if(len(weights)==2):
                    layer.set_weights([np.random.rand(*weights[0].shape)*2 +0.0001,np.random.rand(*weights[1].shape)*2 +0.0001])
                else:
                    continue
        return model
        if(mut_type=='random_mask'):
            for num_mutation in range(num_mutations):
                for layer in model.layers:
                    w = [0,0]
                    weights = layer.get_weights()
                    if(len(weights)==2):
                        #Mask for which weight values will be changed
                        mask_weights = np.random.randint(0,2,size=weights[0].shape).astype(np.bool)
                        mask_biases = np.random.randint(0,2,size=weights[1].shape).astype(np.bool)
                        #Random values in range [0.01,2.001) 
                        #Used to be *2 + 0.001
                        w[0] = np.random.rand(*weights[0].shape)*4 -0.99 
                        weights[0][mask_weights] = weights[0][mask_weights] * w[0][mask_weights]
                        w[1] = np.random.rand(*weights[1].shape)*4 -0.99 
                        weights[1][mask_biases] = weights[1][mask_biases] * w[1][mask_biases]
                        layer.set_weights(weights)
                    else:
                        continue
        return model

    def get_KFold_split(self,x_train,y_train,n_splits):
        #CAUTION CURRENTLY WORKING ONLY FOR MULTI-CLASS, but notmulti-label!!!
        X = []
        Y = []
        skf = KFold(n_splits=n_splits)
        for train_index, test_index in skf.split(x_train, y_train.argmax(1)):
            X.append(x_train[np.array(test_index)])
            Y.append(y_train[np.array(test_index)])
        return X,Y


class _Population(object):

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, score, obj='max'):
        self.members = members
        scores = fitnesses
        if score:
            self.scores = score(scores)
        else:
            self.scores = scores
        self.s_fit = sum(self.scores)

    def get_best(self, n):
        combined = [(self.members[i], self.scores[i])
                    for i in range(len(self.members))]
        sorted(combined, key=(lambda x: x[1]), reverse=True)
        return [x[0] for x in combined[:n]]

    def select(self):
        dart = rand.uniform(0, self.s_fit)
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.scores[i]
            if sum_fits >= dart:
                return self.members[i]
