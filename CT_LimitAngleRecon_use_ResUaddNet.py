import numpy as np
import tflearn
from tflearn import conv_2d, avg_pool_2d, \
    input_data, batch_normalization, leaky_relu, \
    regression
import tensorflow as tf
import time

noisetype = 'nonoise'
fir_list = '1'
sec_list = '05'

I_traindata_npyfile = '../DATASET/CT_dataset/fanbeam_I_{}_{}_{}_' \
                      'train.npy'.format(noisetype, fir_list, sec_list)
I_valdata_npyfile = '../DATASET/CT_dataset/fanbeam_I_{}_{}_{}_' \
                    'val.npy'.format(noisetype, fir_list, sec_list)

R_traindata_npyfile = '../DATASET/CT_dataset/fanbeam_R_{}_{}_{}_' \
                      'train.npy'.format(noisetype, fir_list, sec_list)
R_valdata_npyfile = '../DATASET/CT_dataset/fanbeam_R_{}_{}_{}_' \
                    'val.npy'.format(noisetype, fir_list, sec_list)

parth = 64
partw = 64
n_channels = 1

I_train = np.load(I_traindata_npyfile)
I_train = np.array([i_train for i_train in I_train]).reshape(-1, parth, partw, n_channels)
I_val = np.load(I_valdata_npyfile)
I_val = np.array([i_val for i_val in I_val]).reshape(-1, parth, partw, n_channels)

R_train = np.load(R_traindata_npyfile)
R_train = np.array([r_train for r_train in R_train]).reshape(-1, parth, partw, n_channels)
R_val = np.load(R_valdata_npyfile)
R_val = np.array([r_val for r_val in R_val]).reshape(-1, parth, partw, n_channels)


def resn_unit(incoming, nb, growth, weight_init='variance_scaling',
              weight_decay=0.0001, name='dens_unit'):
    rens = incoming
    with tf.variable_scope(name):
        for i in range(nb):
            conn = rens
            bn1 = batch_normalization(rens, name='bn1')
            relu1 = tf.nn.relu(bn1, name='relu1')
            conv1 = conv_2d(relu1, growth, 3, weights_init=weight_init,
                            weight_decay=weight_decay, name='conv1')
            bn2 = batch_normalization(conv1, name='bn2')
            relu2 = tf.nn.relu(bn2, name='relu2')
            conv2 = conv_2d(relu2, growth, 3, weights_init=weight_init,
                            weight_decay=weight_decay, name='conv2')
            conn_bn = batch_normalization(conn, name='conn_bn')
            conn_relu = tf.nn.relu(conn_bn, name='conn_relu')
            conn_conv = conv_2d(conn_relu, growth, 1, weights_init=weight_init,
                                weight_decay=weight_decay, name='conn_conv')
            rens = tf.add(conv2, conn_conv, name='rens')
        return rens


def resn_ups_concat(incoming, coct, nb, growth, ih, iw,
                    name='resn_ups_concat'):
    with tf.variable_scope(name):
        resn_ud = resn_unit(incoming, nb, growth, name='resn_ud')
        addt = tf.add(resn_ud, coct, name='addt')
        ups = tf.image.resize_nearest_neighbor(addt, (ih, iw), name='ups')
        return ups


lr = 0.0001

tf.reset_default_graph()

net_input = input_data(shape=[None, parth, partw, n_channels], name='net_input')
net_input_h = net_input.get_shape()[1]
net_input_w = net_input.get_shape()[2]

resn1 = resn_unit(net_input, 4, 64, name='resn1')
resn1_ds = avg_pool_2d(resn1, 2, name='resn1_ds')
resn1_ds_h = resn1_ds.get_shape()[1]
resn1_ds_w = resn1_ds.get_shape()[2]
resn2 = resn_unit(resn1_ds, 4, 128, name='resn2')
resn2_ds = avg_pool_2d(resn2, 2, name='resn2_ds')
resn2_ds_h = resn2_ds.get_shape()[1]
resn2_ds_w = resn2_ds.get_shape()[2]
resn3 = resn_unit(resn2_ds, 4, 256, name='resn3')
resn3_ds = avg_pool_2d(resn3, 2, name='resn3_ds')

resn_mid = resn_unit(resn3_ds, 4, 512, name='dens_mid')

resn_upadd1 = resn_ups_concat(resn_mid, resn3_ds, 4, 256, resn2_ds_h,
                              resn2_ds_w, name='resn_upadd1')
resn_upadd2 = resn_ups_concat(resn_upadd1, resn2_ds, 4, 128, resn1_ds_h,
                              resn1_ds_w, name='resn_upadd2')
resn_upadd3 = resn_ups_concat(resn_upadd2, resn1_ds, 4, 64, net_input_h,
                              net_input_w, name='resn_upadd3')

y_add = tf.add(resn_upadd3, net_input, name='y_add')
y_conv = conv_2d(y_add, 1, 3, weights_init='variance_scaling',
                 weight_decay=0.0001, name='y_conv')
y_output = tf.tanh(y_conv, name='y_output')

net_loss = regression(y_output, optimizer='adam', learning_rate=lr,
                      loss='mean_square', name='target')
model = tflearn.DNN(net_loss, clip_gradients=0., tensorboard_dir='log')

start_time = time.time()
netname = 'CNN012_06_9_01'
idx = '7'
epochs = 51
for e in range(1, epochs):
    MODEL_NAME = 'CT_fanbeam-{}-{}-{}.model'.format(netname, idx, e)
    model.fit({'net_input': R_train}, {'target': I_train}, n_epoch=1,
              batch_size=16, validation_set=({'net_input': R_val},
                                             {'target': I_val}),
              run_id=MODEL_NAME)
    model.save(MODEL_NAME)
duration = time.time() - start_time
print(duration/3600)
