#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:19:18 2018

@author: bertinetti
"""

import numpy as np
from keras.models import Model, Sequential
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Reshape, Dropout, concatenate, UpSampling3D, Activation, BatchNormalization,LeakyReLU, PReLU, Deconvolution3D, SpatialDropout3D, Add
from keras.optimizers import Adam, Adadelta
from metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient, weighted_dice_coefficient_loss,tversky_loss
from data import open_hdf5_file, print_hdf5_item_structure

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def MksNet3D(nClasses,shape,W,lr=1e-5):
    print('Input shape:',shape)
    model=Sequential()          #kernel initziaLIZER
    model.add(Conv3D(16,(5,5,5),input_shape=(shape[0],shape[1],shape[2],1), padding='same',activation="relu"))
    model.add(Conv3D(16,(5,5,5),padding='same',activation="relu"))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Dropout(0.25))
    model.add(Conv3D(32,(5,5,5),padding='same',activation="relu"))
    model.add(Conv3D(32,(5,5,5),padding='same',activation="relu"))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Dropout(0.25))
    model.add(Conv3D(64,(5,5,5),padding='same',activation="relu"))
    model.add(Conv3D(64,(5,5,5),padding='same',activation="relu"))
    model.add(UpSampling3D(size=(2,2,2)))
    model.add(Dropout(0.25))
    model.add(Conv3D(32,(5,5,5),padding='same',activation="relu"))
    model.add(Conv3D(32,(5,5,5),padding='same',activation="relu"))
    model.add(UpSampling3D(size=(2,2,2)))
    model.add(Dropout(0.25))
    model.add(Conv3D(16,(5,5,5),padding='same',activation="relu"))
    model.add(Conv3D(16,(5,5,5),padding='same',activation="relu"))
    model.add(Conv3D(nClasses,(5,5,5),padding='same'))
    if nClasses==1:
        model.add(Activation('sigmoid'))
    if nClasses>1:
        model.add(Activation('relu'))
        model.add(Reshape((shape[0] * shape[1]* shape[2],nClasses), input_shape=(shape[0], shape[1],shape[2],nClasses)))
#        model.add(Permute((2, 1)))
        model.add(Activation('softmax'))
    if W !='':
#        print('hdf weight structure:')
#        print_hdf5_item_structure(W)
        model.load_weights(W)
    if nClasses == 1:
        if lr:
            model.compile(loss='binary_crossentropy',optimizer=Adam(lr=lr),metrics=['accuracy'])
        else:
            model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0008),metrics=['accuracy'])
    if nClasses > 1:
        if lr:
            model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr),metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-5),metrics=['accuracy'])
#            model.compile(loss=tversky_loss,optimizer=Adam(lr=1e-4),metrics=['accuracy'])
    model.summary()
    return model

#def MksNet3D(n_labels,shape,W,lr=5e-4, n_base_filters=16, depth=3, dropout_rate=0.25,pool_size=(2,2,2,),kernel=(4, 4, 4),batch_normalization=False,deconvolution=False,
#                      data_format="channels_last",droput_rate=0.25,optimizer=Adam,activation_name="sigmoid"):
#    inputs = Input(shape)
#    print('Input shape:',shape)
#    current_layer = inputs
#    levels = list()
#    for layer_depth in range(depth):
#        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
#                                          batch_normalization=batch_normalization,kernel=kernel)
#        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth),
#                                          batch_normalization=batch_normalization,kernel=kernel)
#        if layer_depth < depth - 1:
#            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
#            current_layer = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(current_layer)
#            levels.append([layer1, layer2, current_layer])
#        else:
#            current_layer = layer2
#            levels.append([layer1, layer2])
#    for layer_depth in range(depth-2, -1, -1):
#        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
#                                            n_filters=levels[layer_depth][1]._keras_shape[4])(current_layer)
#        dout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(up_convolution)
#        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[4],
#                                                 input_layer=dout, batch_normalization=batch_normalization,kernel=kernel)
#        concat = concatenate([current_layer, levels[layer_depth][1]], axis=4)
#        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[4],
#                                                 input_layer=concat, batch_normalization=batch_normalization,kernel=kernel)
#        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[4],
#                                                 input_layer=current_layer,
#                                                 batch_normalization=batch_normalization,kernel=kernel)
#    if n_labels>1:
#        final_convolution = Conv3D(n_labels, 1)(current_layer)
#        o = Reshape((shape[0] * shape[1]* shape[2],n_labels), input_shape=(shape[0], shape[1], shape[2],n_labels))(final_convolution)
#        activation_name="softmax"
##        o = (Permute((2, 1)))(o)
#    if n_labels==1:
#        o = Conv3D(n_labels, (1, 1, 1))(current_layer)
#        activation_name="sigmoid"
#    act = Activation(activation_name)(o)
#    model = Model(inputs=inputs, outputs=act)            
#    if W !='':
#        model.load_weights(W)
#    if n_labels>1:
##        model.compile(loss=weighted_dice_coefficient_loss, optimizer = Adam(lr = initial_learning_rate) , metrics=metrics )
#        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr =  lr) , metrics=['categorical_accuracy'] )
#    if n_labels==1:
#        model.compile(loss="binary_crossentropy", optimizer = Adam(lr = lr) , metrics=['accuracy'] )
##    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
#    model.summary()
#    return model    
    

def unet_model_3d(n_labels,shape,W,lr=1e-5, pool_size=(2, 2, 2), initial_learning_rate=0.00001, deconvolution=False,
                  depth=3, n_base_filters=16, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(shape)
    print('Input shape:',shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[4])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=4)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)


    if n_labels>1:
        final_convolution = Conv3D(n_labels, 1)(current_layer)
        o = Reshape((shape[0] * shape[1]* shape[2],n_labels), input_shape=(shape[0], shape[1], shape[2],n_labels))(final_convolution)
        activation_name="softmax"
#        o = (Permute((2, 1)))(o)
    if n_labels==1:
        o = Conv3D(n_labels, (1, 1, 1))(current_layer)
        activation_name="sigmoid"
    act = Activation(activation_name)(o)
    model = Model(inputs=inputs, outputs=act)
    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics
    if W !='':
#        print('hdf weight structure:')
#        print_hdf5_item_structure(W)
        print('Loading weight:',W)
        model.load_weights(W)
    if n_labels>1:
#        model.compile(loss=weighted_dice_coefficient_loss, optimizer = Adam(lr = initial_learning_rate) , metrics=metrics )
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr =  lr) , metrics=['accuracy'] )
    if n_labels==1:
        model.compile(loss="binary_crossentropy", optimizer = Adam(lr = lr) , metrics=['accuracy'] )
#    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    model.summary()
    return model


def isensee2017_3D(n_labels,shape,W,lr=1e-5, n_base_filters=16, depth=4, dropout_rate=0.3,
                      n_segmentation_levels=3, optimizer=Adam, initial_learning_rate=9e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(shape)
    print('Input shape:',shape)
    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters,activation=LeakyReLU, instance_normalization=False)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2),activation=LeakyReLU, instance_normalization=False)

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=4)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, create_convolution_block(current_layer, n_filters=n_labels, kernel=(1, 1, 1)))
#    output_layer = current_layer
    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    if n_labels==1:
        activation_block = Activation('sigmoid')(output_layer)
    if n_labels>1:
        final_convolution = Conv3D(n_labels, 1)(output_layer)
        o = Reshape((shape[0] * shape[1]* shape[2],n_labels), input_shape=(shape[0], shape[1], shape[2],n_labels))(final_convolution)
        activation_block = Activation('softmax')(o)
    model = Model(inputs=inputs, outputs=activation_block)
    if W !='':
        model.load_weights(W)
    if n_labels == 1:
        model.compile(loss='binary_crossentropy',optimizer=Adam(lr=lr),metrics=['accuracy'])
    if n_labels > 1:
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr),metrics=['categorical_accuracy'])
    model.summary()
    return model


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=4)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=4)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)