import numpy as np
import random as rand
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf


class GenomeHandler:
    """
    *Adapter from DEvol
    Defines the configuration and handles the conversion and mutation of
    individual genomes. Should be created and passed to a `Devol_based_NE` instance.

    ---
    Genomes are represented as fixed-width lists of integers corresponding
    to sequential layers and properties. 
    """

    def __init__(self, max_conv_layers, max_filters,
                input_shape, n_classes, batch_size=256,
                 batch_normalization=True, dropout=True, max_pooling=True,
                 optimizers=None, activations=None,skip_ops=None):
        """
        Creates a GenomeHandler according

        Args:
            max_conv_layers: The maximum number of convolutional layers
            max_filters: The maximum number of conv filters (feature maps) in a
                    convolutional layer
            input_shape: The shape of the input
            n_classes: The number of classes
            batch_normalization (bool): whether the GP should include batch norm
            dropout (bool): whether the GP should include dropout
            max_pooling (bool): whether the GP should include max pooling layers
            optimizers (list): list of optimizers to be tried by the GP. By
                    default, the network uses Keras's built-in adam, rmsprop,
                    adagrad, and adadelta
            activations (list): list of activation functions to be tried by the
                    GP. By default, relu and sigmoid.
        """
        if max_conv_layers < 1:
            raise ValueError(
                "At least one conv layer is required for AE to work"
            )
        if max_filters > 0:
            filter_range_max = int(math.log(max_filters, 2)) + 1
        else:
            filter_range_max = 0
        self.optimizer = optimizers or [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta'
        ]
        self.activation = activations or [
            'relu',
            'sigmoid',
        ]
        self.skip_op = skip_ops or[
            'none',
            'add',
            'concatenate'
        ]
        self.convolutional_layer_shape = [
            "active",
            "num filters",
            "kernel_size",
            "batch normalization",
            "activation",
            "dropout",
            "max pooling",
            "skip_op"
        ]
        self.layer_params = {
            "active": [0, 1],
            "num filters": [2**i for i in range(2, filter_range_max)],
            #Added after paper release
            "kernel_size": [3,5,7],
            "batch normalization": [0, (1 if batch_normalization else 0)],
            "activation": list(range(len(self.activation))),
            "dropout": [(i if dropout else 0) for i in range(11)],
            "max pooling": list(range(3)) if max_pooling else 0,
            #In development
            "skip_op": list(range(len(self.skip_op)))
        }

        self.convolution_layers = max_conv_layers
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.batch_size = batch_size

    def convParam(self, i):
        key = self.convolutional_layer_shape[i]
        return self.layer_params[key]

    def mutate(self, genome, num_mutations):
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            index = np.random.choice(list(range(1, len(genome))))
            if index < self.convolution_layer_size * self.convolution_layers:
                if genome[index - index % self.convolution_layer_size]:
                    range_index = index % self.convolution_layer_size
                    choice_range = self.convParam(range_index)
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01:  # randomly flip deactivated layers
                    genome[index - index % self.convolution_layer_size] = 1
            elif index == len(genome) -1:
                genome[index] = np.random.choice(list(range(len(self.optimizer))))
        return genome

    def decode(self, genome):
        if not self.is_compatible_genome(genome):
            raise ValueError("Invalid genome for specified configs")
        model = Sequential()
        dim = 0
        offset = 0
        optim_offset = 0
        if self.convolution_layers > 0:
            dim = min(self.input_shape[:-1])  # keep track of smallest dimension
        input_layer = True
        dims = []
        lays = []
        temp_features = 0
        features = dict()
        for i in range(self.convolution_layers):
            if genome[offset]:
                convolution = None
                if input_layer:
                    temp_features = genome[offset + 1]
                    temp_kernel = genome[offset + 2]
                    convolution = Convolution2D(
                        temp_features, (temp_kernel, temp_kernel),
                        padding='same',
                        input_shape=self.input_shape
                    )
                    input_layer = False
                else:
                    temp_features = int(min(features[list(features.keys())[-1]],genome[offset + 1]))
                    temp_kernel = genome[offset + 2]
                    convolution = Convolution2D(
                        temp_features, (temp_kernel, temp_kernel),
                        padding='same'
                    )
                model.add(convolution)
                if genome[offset + 3]:
                    model.add(BatchNormalization())
                model.add(Activation(self.activation[genome[offset + 4]]))
                max_pooling_type = genome[offset + 6]
                if max_pooling_type == 1 and dim >= 3:
                    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
                    dim /= 2
            dims.append(dim)
            features[i] = temp_features
            dim = int(math.ceil(dim))
            if(i<self.convolution_layers-1):
                offset += self.convolution_layer_size
            else:
                optim_offset = offset + self.convolution_layer_size
        level_of_compression = np.prod(model.layers[-1].output_shape[1:])
        level_of_compression = min(math.log(level_of_compression,10),5)
        model.add(Convolution2D(temp_features,(3,3),padding='same'))
        needed_reductions = [i-2 for i,temp_dim in enumerate(dims) if(math.ceil(temp_dim)!=math.floor(temp_dim))]
        #Reset the offset
        for i in reversed(range(self.convolution_layers)):
            #Done to fix shape when 14->7-> 4 => 4->8->16->14
            if(not(dim in dims) and ((dim-2)*2 in dims or (not(dim*2 in dims) and (dim-2)*2==min(self.input_shape[:-1])))):
                model.add(Convolution2D(features[i],(3,3)))
                dim-=2
            if genome[offset]:
                max_pooling_type = genome[offset + 6]
                convolution = Convolution2D(
                    features[i], (genome[offset+2], genome[offset+2]),
                    padding='same'
                )
                model.add(convolution)
                model.add(Activation(self.activation[genome[offset + 4]]))
                if ((dim*2 in dims or (dim*2)-2 in dims or (dim*4)-2 or dim==(int(min(self.input_shape[:-1]))/2)) and dim<min(self.input_shape[:-1])):
                    model.add(UpSampling2D((2, 2)))
                    dim*=2
            if(dim>max(self.input_shape)):
                import pdb
                pdb.set_trace()
            offset -= self.convolution_layer_size
        model.add(Convolution2D(self.input_shape[-1], (genome[2],genome[2]), activation=self.activation[genome[4]], padding='same'))
        model.compile(loss='binary_crossentropy',
                      optimizer=self.optimizer[genome[optim_offset]],
                      metrics=["accuracy"])
        return model,level_of_compression

    def genome_representation(self):
        encoding = []
        for i in range(self.convolution_layers):
            for key in self.convolutional_layer_shape:
                encoding.append("Conv" + str(i) + " " + key)
        encoding.append("Optimizer")
        return encoding

    def generate(self):
        genome = []
        for i in range(self.convolution_layers):
            for key in self.convolutional_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))
        genome.append(np.random.choice(list(range(len(self.optimizer)))))
        genome[0] = 1
        return genome

    def is_compatible_genome(self, genome):
        expected_len = self.convolution_layers * self.convolution_layer_size + 1
        if len(genome) != expected_len:
            return False
        ind = 0
        for i in range(self.convolution_layers):
            for j in range(self.convolution_layer_size):
                if genome[ind + j] not in self.convParam(j):
                    return False
            ind += self.convolution_layer_size
        if genome[ind] not in range(len(self.optimizer)):
            return False
        return True

    def best_genome(self, csv_path, metric="loss", include_metrics=False):
        best = max if metric == "accuracy" else min
        col = -1 if metric == "accuracy" else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        return genome

    def decode_best(self, csv_path, metric="loss"):
        return self.decode(self.best_genome(csv_path, metric, False))
