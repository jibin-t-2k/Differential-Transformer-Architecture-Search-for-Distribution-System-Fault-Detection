from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd 
import os, re
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras import backend as K

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Input, Reshape, Rescaling, 
                                    TimeDistributed, MaxPool1D, BatchNormalization, 
                                    Embedding, Dense, Dropout,
                                    Flatten, Softmax)

from search_layers import (regularizer, kernel_init, 
                            Zero, MultiHeadEncoderAttention, MultiHeadDecoderAttention,
                            Densely, Conv, SepConv, DilConv, Identity)


from tensorflow.keras.utils import plot_model
import imageio

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

import gc
from collections import namedtuple

import sys
from graphviz import Digraph

from plot_utils import train_curves, plot

from dl_eval_plot_fns import plot_confusion_matrix, plot_roc, train_curves


try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    #strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU

print("Number of accelerators: ", strategy.num_replicas_in_sync)
print(tf.__version__)

signals = np.load("/content/drive/MyDrive/DS_Fault_Detection/Data/signals.npy", mmap_mode="r")
signals_gts = np.load("/content/drive/MyDrive/DS_Fault_Detection/Data/signals_gts.npy", mmap_mode="r")

X = []
y = []

for signal, signal_gt in tqdm(zip(signals.astype(np.float32), signals_gts), position=0, leave=True):
    if any(signal_gt[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20]]): # LG, LL, LLG, LLL, LLLG, HIF, Non_Linear_Load_Switch
        noise_count = 20
    elif any(signal_gt[[15, 21]]):  # Capacitor_Switch, Insulator_Leakage
        noise_count = 10
    elif signal_gt[16] == 1: # Load_Switch
        noise_count = 5
    elif signal_gt[22] == 1: # Transformer_Inrush
        noise_count = 30
    elif signal_gt[0] == 1: # No Fault
        noise_count = 100

    for n in range(noise_count):
        X.append(signal)
        y.append(signal_gt)
        
X = np.array(X)
np.random.seed(7)
for i in tqdm(range(X.shape[0])):
    noise = np.random.uniform(-5.0, 5.0, (12800, 15)).astype(np.float32)
    X[i] = X[i] + noise
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)
del X, y, signals, signals_gts
gc.collect()

X = np.load("X.npy", mmap_mode="r")
y = np.load("y.npy", mmap_mode="r")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, shuffle = True, random_state = 77, stratify = y)
print(X_tr.shape, y_tr.shape)
print(X_te.shape, y_te.shape)


cfg = {
    # general setting
    "batch_size": 256,
    "init_channels": 64,
    "layers": 4,
    "num_typ_classes": 23,
    "num_loc_classes": 15,
    "sub_name": 'darts_search',

    # training setting
    "epoch": 500,
    "start_search_epoch": 15,
    "init_lr": 0.001,
    "momentum": 0.9,
    "weights_decay": 3e-4,
    "grad_clip": 10.0,

    "arch_learning_rate": 0.001,
    "arch_weight_decay": 0.001,
    }


tr_shape = X_tr.shape[0]
X_train1 = tf.convert_to_tensor(X_tr[:(tr_shape//8)*1])
X_train2 = tf.convert_to_tensor(X_tr[(tr_shape//8)*1: (tr_shape//8)*2])
X_train3 = tf.convert_to_tensor(X_tr[(tr_shape//8)*2: (tr_shape//8)*3])
X_train4 = tf.convert_to_tensor(X_tr[(tr_shape//8)*3: (tr_shape//8)*4])
X_train5 = tf.convert_to_tensor(X_tr[(tr_shape//8)*4: (tr_shape//8)*5])
X_train6 = tf.convert_to_tensor(X_tr[(tr_shape//8)*5: (tr_shape//8)*6])
X_train7 = tf.convert_to_tensor(X_tr[(tr_shape//8)*6: (tr_shape//8)*7])
X_train8 = tf.convert_to_tensor(X_tr[(tr_shape//8)*7:])
X_train = tf.concat([X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8], axis=0)

te_shape = X_te.shape[0]
X_test1 = tf.convert_to_tensor(X_te[: (te_shape//4)*1])
X_test2 = tf.convert_to_tensor(X_te[(te_shape//4)*1:(te_shape//4)*2])
X_test3 = tf.convert_to_tensor(X_te[(te_shape//4)*2:(te_shape//4)*3])
X_test4 = tf.convert_to_tensor(X_te[(te_shape//4)*3:])
X_test = tf.concat([X_test1, X_test2, X_test3, X_test4], axis=0)

y_train = tf.convert_to_tensor(y_tr)
y_test = tf.convert_to_tensor(y_te)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train[:,:23], y_train[:,23:]))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test[:,:23], y_test[:,23:]))

train_dataset = train_dataset.shuffle(4000).batch(cfg["batch_size"], drop_remainder=True)#.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(cfg["batch_size"], drop_remainder=True)#.prefetch(tf.data.experimental.AUTOTUNE)


def count_parameters_in_MB(model):
    """count parameters in MB"""
    return np.sum(
        [tf.keras.backend.count_params(w) for w in model.trainable_weights
         if 'Auxiliary' not in w.name]) / 1e6


OPS = {'none': lambda units, wd: 
           Zero(),
       'max_pool_2': lambda units, wd:
           MaxPool1D(2, strides = 1, padding='same'),
       'skip_connect': lambda units, wd:
           Identity(),
       'encoder_att': lambda units, wd:
           MultiHeadEncoderAttention(units, wd),
       'decoder_att': lambda units, wd:
           MultiHeadDecoderAttention(units, wd),
       'dense': lambda units, wd:
           Densely(units, wd),
       'sep_conv_1': lambda units, wd:
           SepConv(units, 1, wd),
       'sep_conv_3': lambda units, wd:
           SepConv(units, 3, wd),
       'sep_conv_5': lambda units, wd:
           SepConv(units, 5, wd),
       'conv_1': lambda units, wd:
           Conv(units, 1, wd),
       'conv_3': lambda units, wd:
           Conv(units, 3, wd),
       'conv_5': lambda units, wd:
           Conv(units, 5, wd),
       'dil_conv_1': lambda units, wd:
           DilConv(units, 1, 2, wd),
       'dil_conv_3': lambda units, wd:
           DilConv(units, 3, 2, wd),
       'dil_conv_5': lambda units, wd:
           DilConv(units, 5, 2, wd),
           }

Genotype = namedtuple('Genotype', 'normal normal_concat') #reduce reduce_concat')

PRIMITIVES = [
    'none',
    # 'max_pool_2',
    'skip_connect',
    'encoder_att',
    'decoder_att',
    'dense',
    'sep_conv_1',
    'sep_conv_3',
    # 'sep_conv_5',
    'conv_1',
    'conv_3',
    # 'conv_5',
    'dil_conv_1',
    'dil_conv_3',
    # 'dil_conv_5'
    ]


# def channel_shuffle(x, groups):
#     _, steps, num_channels = x.shape

#     assert num_channels % groups == 0
#     channels_per_group = num_channels // groups

#     x = tf.reshape(x, [-1, steps, groups, channels_per_group])
#     x = tf.transpose(x, [0, 1, 3, 2])
#     x = tf.reshape(x, [-1, steps, num_channels])

#     return x


class MixedOP(tf.keras.layers.Layer):
    """Mixed OP"""
    def __init__(self, ch, wd, name='MixedOP', **kwargs):
        super(MixedOP, self).__init__(name=name, **kwargs)

        self._ops = []

        for primitive in PRIMITIVES:
            op = OPS[primitive](ch // 1, wd)

            if 'pool' in primitive:
                op = Sequential([op, BatchNormalization()])

            self._ops.append(op)

    def call(self, x, weights):
        # channel proportion k = 4
        x_1 = x[:, :, :x.shape[2] // 1]
        # x_2 = x[:, :, x.shape[2] // 4:]

        x_1 = tf.add_n([w * op(x_1) for w, op in
                        zip(tf.split(weights, len(PRIMITIVES)), self._ops)])

        # ans = tf.concat([x_1, x_2], axis=2)

        # return channel_shuffle(ans, 4)
        return x_1


class Cell(tf.keras.layers.Layer):
    """Cell Layer"""
    def __init__(self, steps, multiplier, ch, reduction, reduction_prev, wd,
                 name='Cell', **kwargs):
        super(Cell, self).__init__(name=name, **kwargs)

        self.wd = wd
        self.steps = steps
        self.multiplier = multiplier

        self.preprocess0 = Densely(ch, wd=wd)
        self.preprocess1 = Densely(ch, wd=wd)

        self._ops = []
        for i in range(self.steps):
            for j in range(2 + i):
                op = MixedOP(ch, wd=wd)
                self._ops.append(op)

    def call(self, s0, s1, weights): #, edge_weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for _ in range(self.steps):
            s = 0
            for j, h in enumerate(states):
                branch = self._ops[offset + j](h, weights[offset + j])
                s += branch #* edge_weights[offset + j]
            offset += len(states)
            states.append(s)

        return tf.concat(states[-self.multiplier:], axis=-1)


class SplitSoftmax(tf.keras.layers.Layer):
    """Split Softmax Layer"""
    def __init__(self, size_splits, name='SplitSoftmax', **kwargs):
        super(SplitSoftmax, self).__init__(name=name, **kwargs)
        self.size_splits = size_splits
        self.soft_max_func = Softmax(axis=-1)

    def call(self, value):
        return tf.concat(
            [self.soft_max_func(t) for t in tf.split(value, self.size_splits)],
            axis=0)


class SearchNetArch(object):
    """Search Network Architecture"""
    def __init__(self, cfg, steps=4, multiplier=2,
                 name='SearchModel'):
        self.cfg = cfg
        self.steps = steps
        self.multiplier = multiplier
        self.name = name

        self.arch_parameters = self._initialize_alphas()
        self.model = self._build_model()

    def _initialize_alphas(self):
        k = sum(range(2, 2 + self.steps))
        num_ops = len(PRIMITIVES)
        w_init = tf.random_normal_initializer()
        self.alphas_normal = tf.Variable(
            initial_value=1e-3 * w_init(shape=[k, num_ops], dtype='float32'),
            trainable=True, name='alphas_normal')
        # self.alphas_reduce = tf.Variable(
        #     initial_value=1e-3 * w_init(shape=[k, num_ops], dtype='float32'),
        #     trainable=True, name='alphas_reduce')
        # self.betas_normal = tf.Variable(
        #     initial_value=1e-3 * w_init(shape=[k], dtype='float32'),
        #     trainable=True, name='betas_normal')
        # self.betas_reduce = tf.Variable(
        #     initial_value=1e-3 * w_init(shape=[k], dtype='float32'),
        #     trainable=True, name='betas_reduce')

        return [self.alphas_normal, 
                # self.alphas_reduce, 
                # self.betas_normal
                # self.betas_reduce
                ]

    def _build_model(self):
        """Model"""

        ch_init = self.cfg['init_channels']
        wd = self.cfg['weights_decay']

        # define model

        inputs = Input(shape=(12800, 15))
        sig = Rescaling(scale=1.0/6065.3467)(inputs)
        sig = Reshape((50, 256, 15))(sig)
        sig = TimeDistributed(Flatten())(sig)

        sig = Dense(512, activation="relu")(sig)
        sig = Dropout(0.2)(sig)
        sig = Dense(64, activation="relu")(sig)
        sig = Dropout(0.2)(sig)

        alphas_normal = Input([None], name='alphas_normal')
        # alphas_reduce = Input([None], name='alphas_reduce')
        # betas_normal = Input([], name='betas_normal')        
        # betas_reduce = Input([], name='betas_reduce')

        # alphas_reduce_weights = Softmax(
        #     name='AlphasReduceSoftmax')(alphas_reduce)
        alphas_normal_weights = Softmax(
            name='AlphasNormalSoftmax')(alphas_normal)
        # betas_reduce_weights = SplitSoftmax(
        #     range(2, 2 + self.steps), name='BetasReduceSoftmax')(betas_reduce)
        # betas_normal_weights = SplitSoftmax(
        #     range(2, 2 + self.steps), name='BetasNormalSoftmax')(betas_normal)

        embeddings = Embedding(input_dim=50, output_dim=64)
        position_embed = embeddings(tf.range(start=0, limit=50, delta=1))
        sig = sig + position_embed

        s0 = s1 = sig

        ch_curr = ch_init
        reduction_prev = False
        for layer_index in range(4):
            # if layer_index in [-1]:
            #     ch_curr = ch_curr*2
            #     reduction = True
            #     weights = alphas_reduce_weights
            #     edge_weights = betas_reduce_weights
            # else:
            reduction = False
            weights = alphas_normal_weights
            # edge_weights = betas_normal_weights

            cell = Cell(self.steps, self.multiplier, ch_curr, reduction,
                        reduction_prev, wd, name=f'Cell_{layer_index}')
            
            s0, s1 = s1, cell(s0, s1, weights) #, edge_weights)

            reduction_prev = reduction


        sig = Dense(64, activation="relu")(s1)
        sig = Dropout(0.1)(sig)
        sig = Flatten()(sig)

        typ = Dense(512, activation="relu")(sig)
        typ = Dropout(0.2)(typ)
        typ = Dense(128, activation="relu")(typ)
        typ = Dense(32, activation="relu")(typ)
        typ = Dropout(0.2)(typ)
        typ_output = Dense(23, activation="softmax", name="type", kernel_initializer=kernel_init(),
                       kernel_regularizer=regularizer(wd))(typ)

        loc = Dense(512, activation="relu")(sig)
        loc = Dropout(0.2)(loc)
        loc = Dense(128, activation="relu")(loc)
        loc = Dense(32, activation="relu")(loc)
        loc = Dropout(0.2)(loc)
        loc_output = Dense(15, activation="softmax", name="loc", kernel_initializer=kernel_init(),
                       kernel_regularizer=regularizer(wd))(loc)

        return Model(
            (inputs, 
             alphas_normal 
            #  alphas_reduce, 
            #  betas_normal
            #  betas_reduce
             ),
            outputs=[typ_output, loc_output], name=self.name)

    def get_genotype(self):
        """get genotype"""
        def _parse(weights): #, edge_weights):
            n = 2
            start = 0
            gene = []
            for i in range(self.steps):
                end = start + n
                w = weights[start:end].copy()
                # ew = edge_weights[start:end].copy()

                # fused weights
                # for j in range(n):
                #     w[j, :] = w[j, :] * ew[j]

                # pick the top 2 edges (k = 2).
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(w[x][k] for k in range(len(w[x]))
                                       if k != PRIMITIVES.index('none'))
                    )[:2]

                # pick the top best op, and append into genotype.
                for j in edges:
                    k_best = None
                    for k in range(len(w[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or w[j][k] > w[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))

                start = end
                n += 1

            return gene

        # gene_reduce = _parse(
        #     Softmax()(self.alphas_reduce).numpy(),
        #     SplitSoftmax(range(2, 2 + self.steps))(self.betas_reduce).numpy())
        gene_normal = _parse(
            Softmax()(self.alphas_normal).numpy())
            # SplitSoftmax(range(2, 2 + self.steps))(self.betas_normal).numpy())

        concat = range(2 + self.steps - self.multiplier, self.steps + 2)
        genotype = Genotype(normal=gene_normal, normal_concat=concat
                            # reduce=gene_reduce, reduce_concat=concat
                            )

        return genotype   


with strategy.scope():
    sna = SearchNetArch(cfg)

sna.model.summary()
print("param size = {:f}MB".format(count_parameters_in_MB(sna.model)))
plot_model(sna.model, expand_nested=True, show_shapes=True)


with strategy.scope(): 

    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                          reduction=tf.keras.losses.Reduction.NONE)

    def CrossEntropyLoss():
        """"cross entropy loss"""
        def cross_entropy_loss(y_true, y_pred):
            per_example_loss = loss_object(y_true, y_pred)
            cross_entropy_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=cfg["batch_size"])
            return cross_entropy_loss
        return cross_entropy_loss

    # define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['init_lr'])
    optimizer_arch = tf.keras.optimizers.Adam(learning_rate=cfg['arch_learning_rate'])

    # define losses function
    criterion = CrossEntropyLoss()

    typ_train_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='typ_train_accuracy')
    loc_train_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='loc_train_accuracy')
    typ_val_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='typ_val_accuracy')
    loc_val_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='loc_val_accuracy')



with strategy.scope():   
    @tf.function
    def distributed_train_step(inputs, typ_labels, loc_labels):
        per_replica_total_loss = strategy.run(train_step,args=(inputs, typ_labels, loc_labels))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_total_loss, axis=None)

    @tf.function
    def distributed_train_step_arch(inputs, typ_labels, loc_labels):
        per_replica_losses = strategy.run(train_step_arch, args=(inputs, typ_labels, loc_labels))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    # define training step function for model
    def train_step(inputs, typ_labels, loc_labels):
        with tf.GradientTape() as tape:
            typ_output, loc_output = sna.model((inputs, *sna.arch_parameters), training=True)

            losses = {}
            losses['reg'] = tf.reduce_sum(sna.model.losses)
            losses['tce'] = criterion(typ_labels, typ_output)
            losses['lce'] = criterion(loc_labels, loc_output)
            loss_values = [l for l in losses.values()]
            total_loss = tf.add_n(loss_values)

        grads = tape.gradient(loss_values, sna.model.trainable_variables)
        # grads = [(tf.clip_by_norm(grad, cfg['grad_clip'])) for grad in grads]
        optimizer.apply_gradients(zip(grads, sna.model.trainable_variables))
        typ_train_accuracy.update_state(typ_labels, typ_output)
        loc_train_accuracy.update_state(loc_labels, loc_output)

        return total_loss

    # define training step function for arch_parameters
    def train_step_arch(inputs, typ_labels, loc_labels):
        with tf.GradientTape() as tape:
            typ_output, loc_output = sna.model((inputs, *sna.arch_parameters), training=True)

            losses = {}
            losses['reg'] = cfg['arch_weight_decay'] * tf.add_n(
                [tf.reduce_sum(p**2) for p in sna.arch_parameters])
            losses['tce'] = criterion(typ_labels, typ_output)
            losses['lce'] = criterion(loc_labels, loc_output)
            loss_values = [l for l in losses.values()]
            total_loss = tf.add_n(loss_values)

        grads = tape.gradient(loss_values, sna.arch_parameters)
        optimizer_arch.apply_gradients(zip(grads, sna.arch_parameters))
        typ_val_accuracy.update_state(typ_labels, typ_output)
        loc_val_accuracy.update_state(loc_labels, loc_output)

        return total_loss


with strategy.scope(): 
    train_losses = [] 
    val_losses = [] 
    train_typ_accs = [] 
    val_typ_accs = [] 
    train_loc_accs = [] 
    val_loc_accs = [] 

    train_loss = 0
    val_loss = 10
    for epoch in range(500):

        if epoch >= 5:
            val_total_loss = 0.0
            val_num_batches = 0
            for inputs_val, typ_labels_val, loc_labels_val in tqdm(test_dataset):
                val_total_loss  += distributed_train_step_arch(inputs_val, typ_labels_val, loc_labels_val)
                val_num_batches += 1
            val_loss = val_total_loss / val_num_batches

        total_loss = 0.0
        num_batches = 0
        for inputs, typ_labels, loc_labels in tqdm(train_dataset):
            total_loss += distributed_train_step(inputs, typ_labels, loc_labels)
            num_batches += 1
        train_loss = total_loss / num_batches

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        template = ("Epoch {}, Loss: {:.4f}, Type Accuracy: {:.2f}, Location Accuracy: {:.2f}, Val Loss: {:.4f}, Val Type Accuracy: {:.2f}, Val Location Accuracy: {:.2f}\n")
        print(template.format(epoch+1, train_loss,
                            typ_train_accuracy.result()*100,
                            loc_train_accuracy.result()*100,
                            val_loss,
                            typ_val_accuracy.result()*100,
                            loc_val_accuracy.result()*100))

        train_typ_accs.append(typ_train_accuracy.result())
        val_typ_accs.append(typ_val_accuracy.result())
        train_loc_accs.append(loc_train_accuracy.result())
        val_loc_accs.append(loc_val_accuracy.result())

        typ_train_accuracy.reset_states()
        loc_train_accuracy.reset_states()
        typ_val_accuracy.reset_states()
        loc_val_accuracy.reset_states()

        if epoch >= 5:
            genotype = sna.get_genotype()
            print(f"search arch: {genotype}\n")
            f = open('darts_search_arch_genotype_v2.py', 'a')
            f.write(f"\n{cfg['sub_name']}_{epoch} = {genotype}\n")
            f.close()



search_history = {
    "train_loss" : np.array(train_losses),
    "val_loss" : np.array(val_losses),
    "train_typ_accs" : np.array(train_typ_accs),
    "val_typ_accs" : np.array(val_typ_accs),
    "train_loc_accs" : np.array(train_loc_accs),
    "val_loc_accs" : np.array(val_loc_accs)
}

np.save("pc_darts_search_history_v2.npy", search_history)

plt.rcParams.update({'legend.fontsize': 12,
                    'axes.labelsize': 16, 
                    'axes.titlesize': 16,
                    'xtick.labelsize': 16,
                    'ytick.labelsize': 16})

train_curves(search_history, "DARTS Train Search:")


with open("search_arch_genotype_v2.py") as graph_file:
    graphs = graph_file.readlines()
    epoch = 0
    for i, g in enumerate(graphs):
        if i%2 != 0:
            genotype = eval(g.split(" = ")[1])
            plot(genotype.normal, "/content/pc_darts_genotypes/normal", str(epoch+1))
            epoch+=1


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

filenames = ["/content/pc_darts_genotypes/" + f for f in sorted_alphanumeric(os.listdir("/content/pc_darts_genotypes")) if f.endswith(".png")]


images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('genotypes.gif', images)