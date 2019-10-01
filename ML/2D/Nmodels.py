#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:19:18 2018

@author: bertinetti
"""

from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, merge, Dropout, Flatten, Dense, Activation, Layer, Reshape, Permute, Lambda
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta
from keras import backend as K


K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def crop( o1 , o2 , i  ):
    o_shape2 = Model( i  , o2 ).output_shape
    outputHeight2 = o_shape2[1]
    outputWidth2 = o_shape2[2]

    o_shape1 = Model( i  , o1 ).output_shape
    outputHeight1 = o_shape1[1]
    outputWidth1 = o_shape1[2]

    cx = abs( outputWidth1 - outputWidth2 )
    cy = abs( outputHeight2 - outputHeight1 )

    if outputWidth1 > outputWidth2:
        o1= Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o2)
	
    if outputHeight1 > outputHeight2 :
        o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o2)

    return o1 , o2 

def depth_softmax(matrix):
    sigmoid = lambda x: 1 / (1 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
    return softmax_matrix

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def MksNet(nClasses,shape,W):
    model=Sequential()          #kernel initziaLIZER
    model.add(Conv2D(16,(5,5),input_shape=(shape[0],shape[1],1), padding='same',activation="relu"))
    model.add(Conv2D(16,(5,5),padding='same',activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32,(5,5),padding='same',activation="relu"))
    model.add(Conv2D(32,(5,5),padding='same',activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64,(5,5),padding='same',activation="relu"))
    model.add(Conv2D(64,(5,5),padding='same',activation="relu"))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32,(5,5),padding='same',activation="relu"))
    model.add(Conv2D(32,(5,5),padding='same',activation="relu"))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(16,(5,5),padding='same',activation="relu"))
    model.add(Conv2D(16,(5,5),padding='same',activation="relu"))
    model.add(Conv2D(nClasses,(5,5),padding='same'))
    if nClasses==1:
        model.add(Activation('sigmoid'))
    if nClasses>1:
        model.add(Activation('relu'))
        model.add(Reshape((shape[0] * shape[1],nClasses), input_shape=(shape[0], shape[1],nClasses)))
#        model.add(Permute((2, 1)))
        model.add(Activation('softmax'))
    if W !='':
        model.load_weights(W)
    if nClasses == 1:
        model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.00008),metrics=['accuracy'])
    if nClasses > 1:
        model.compile(loss='categorical_crossentropy',optimizer=Adadelta(lr=1.0),metrics=['accuracy'])
    model.summary()
    return model

def UNet(nClasses,shape,W):

    inputs = Input((shape[0], shape[1],1))

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    print ("conv1 shape:",conv1.shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print ("conv1 shape:",conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print ("pool1 shape:",pool1.shape)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    print ("conv2 shape:",conv2.shape)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print ("conv2 shape:",conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print ("pool2 shape:",pool2.shape)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print ("conv3 shape:",conv3.shape)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    print ("conv3 shape:",conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print ("pool3 shape:",pool3.shape)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    
    o_shape = Model(inputs , conv9 ).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    
    if nClasses>1:
        conv10 = Conv2D(nClasses, 1)(conv9)
        Reshape1 = Reshape((shape[0] * shape[1],nClasses), input_shape=(shape[0], shape[1],nClasses))(conv10)
#        o = (Permute((2, 1)))(o)
        o = (Activation('softmax'))(Reshape1)
    if nClasses==1:
        o = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model = Model( inputs , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    if W !='':
        model.load_weights(W)
    if nClasses>1:
        model.compile(loss="categorical_crossentropy", optimizer = Adam(lr = 1e-5) , metrics=['accuracy'] )
    if nClasses==1:
        model.compile(loss="binary_crossentropy", optimizer = Adam(lr = 1e-5) , metrics=['accuracy'] )
    model.summary()  
    return model

def get_unet1(img_rows, img_cols):
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def SegNet(nClasses,shape,W):
    input_height=shape[0]
    input_width=shape[1]
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    model = Sequential()
    model.add(Layer(input_shape=(input_height , input_width , 1)))
    
        # encoder
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    
    	# decoder
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Convolution2D(512, kernel, kernel, border_mode='valid'))
    model.add( BatchNormalization())
    
    model.add( UpSampling2D(size=(pool_size,pool_size)))
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Convolution2D(256, kernel, kernel, border_mode='valid'))
    model.add( BatchNormalization())
    
    model.add( UpSampling2D(size=(pool_size,pool_size)))
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Convolution2D(128, kernel, kernel, border_mode='valid'))
    model.add( BatchNormalization())
    
    model.add( UpSampling2D(size=(pool_size,pool_size)))
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
    model.add( BatchNormalization())


    model.add(Convolution2D( nClasses , 1, 1, border_mode='valid'))

    model.outputHeight = model.output_shape[-2]
    model.outputWidth = model.output_shape[-1]

    if nClasses==1:
        model.add(Activation('sigmoid'))
    if nClasses>1:
        model.add(Activation('relu'))
        model.add(Reshape((shape[0] * shape[1],nClasses), input_shape=(shape[0], shape[1],nClasses)))
#        model.add(Permute((2, 1)))
        model.add(Activation('softmax'))
    if W !='':
        model.load_weights(W)
    if nClasses == 1:
        model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0001),metrics=['accuracy'])
    if nClasses > 1:
        model.compile(loss='categorical_crossentropy',optimizer=Adadelta(lr=0.0001),metrics=['accuracy'])
    model.summary()
    return model

def VGGSegNet(nClasses,shape,W):
    vgg_level=3
    input_height=shape[0]
    input_width=shape[1]
    img_input = Input(shape=(input_height,input_width,1))
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x
    
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x
    
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x
    
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense( 1000 , activation='softmax', name='predictions')(x)
    
#    vgg  = Model(  img_input , x  )
    
    levels = [f1 , f2 , f3 , f4 , f5 ]
    
    o = levels[ vgg_level ]
    	
    o = ( ZeroPadding2D( (1,1) ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid'))(o)
    o = ( BatchNormalization())(o)
    
    o = ( UpSampling2D( (2,2)))(o)
    o = ( ZeroPadding2D( (1,1)))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid'))(o)
    o = ( BatchNormalization())(o)
    
    o = ( UpSampling2D((2,2)  ) )(o)
    o = ( ZeroPadding2D((1,1) ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' ))(o)
    o = ( BatchNormalization())(o)
    
    o = ( UpSampling2D((2,2)  ))(o)
    o = ( ZeroPadding2D((1,1)  ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  ))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  ))(o)
    o = ( ZeroPadding2D((1,1)  ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  ))(o)
    o = ( BatchNormalization())(o)
    
    o =  Conv2D( nClasses , (3, 3) , padding='same')( o )
    o_shape = Model(img_input , o ).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    
    if nClasses>1:    
        o = (Reshape((  -1  , outputHeight*outputWidth  )))(o)
        o = (Permute((2, 1)))(o)
        o = (Activation('softmax'))(o)
    if nClasses==1:
        o = (Activation('sigmoid'))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    if W !='':
        model.load_weights(W)
    if nClasses>1:
        model.compile(loss="categorical_crossentropy", optimizer = Adam(lr = 1e-4) , metrics=['accuracy'] )
    if nClasses==1:
        model.compile(loss="binary_crossentropy", optimizer = Adam(lr = 1e-4) , metrics=['accuracy'] )
    model.summary()  


#    img_w = shape[1]
#    img_h = shape[0]
#    n_labels = nClasses
#    
#    kernel = 3
#    pad = 1
#    pool_size = 2
#    
#    encoding_layers = [
#        Convolution2D(64, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(64, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        MaxPooling2D(pool_size=(pool_size, pool_size)),
#    
#        Convolution2D(128, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(128, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        MaxPooling2D(pool_size=(pool_size, pool_size)),
#    
#        Convolution2D(256, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(256, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(256, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        MaxPooling2D(pool_size=(pool_size, pool_size)),
#    
#        Convolution2D(512, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(512, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(512, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        MaxPooling2D(pool_size=(pool_size, pool_size)),
#    
#        Convolution2D(512, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(512, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(512, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        MaxPooling2D(pool_size=(pool_size, pool_size)),
#    ]
#    
#    decoding_layers = [
#        UpSampling2D(size=(pool_size,pool_size)),
#        Convolution2D(512, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(512, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(512, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#    
#        UpSampling2D(size=(pool_size,pool_size)),
#        Convolution2D(512, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(512, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(256, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#    
#        UpSampling2D(size=(pool_size,pool_size)),
#        Convolution2D(256, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(256, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(128, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#    
#        UpSampling2D(size=(pool_size,pool_size)),
#        Convolution2D(128, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(64, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#    
#        UpSampling2D(size=(pool_size,pool_size)),
#        Convolution2D(64, kernel, kernel, border_mode='same'),
#        BatchNormalization(),
#        Activation('relu'),
#        Convolution2D(n_labels, 1, 1, border_mode='valid'),
#        BatchNormalization(),
#    ]
#    
#    
#    segnet_basic = Sequential()
#    
#    segnet_basic.add(Layer(input_shape=(img_h, img_w,1)))
#    
#    
#    segnet_basic.encoding_layers = encoding_layers
#    for l in segnet_basic.encoding_layers:
#        segnet_basic.add(l)
#    
#    
#    segnet_basic.decoding_layers = decoding_layers
#    for l in segnet_basic.decoding_layers:
#        segnet_basic.add(l)
#    
#    
#    segnet_basic.add(Reshape((nClasses, img_h * img_w), input_shape=(nClasses,img_h, img_w)))
#    segnet_basic.add(Permute((2, 1)))
#    if nClasses>1:
#        segnet_basic.add(Activation('softmax'))
#    if nClasses==1:
#        segnet_basic.add(Activation('sigmoid'))
#    if nClasses>1:
#        segnet_basic.compile(loss="categorical_crossentropy", optimizer = Adam(lr = 1e-4) , metrics=['accuracy'] )
#    if nClasses==1:
#        segnet_basic.compile(loss="binary_crossentropy", optimizer = Adam(lr = 1e-4) , metrics=['accuracy'] )
#    segnet_basic.summary()    
    return model

def FCN32Net( nClasses ,  shape, W):
    vgg_level=3
    input_height=shape[0]
    input_width=shape[1]
#    assert input_height%32 == 0
#    assert input_width%32 == 0
    img_input = Input(shape=(input_height,input_width,1))



	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1' )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool' )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1' )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool' )(x)
    f2 = x
    
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool' )(x)
    f3 = x
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool' )(x)
    f4 = x
    
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool' )(x)
    f5 = x
    
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense( 1000 , activation='softmax', name='predictions')(x)
    
    o = f5
    
    o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same'))(o)
    o = Dropout(0.5)(o)
    o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same'))(o)
    o = Dropout(0.5)(o)
    o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' ))(o)
    o = Conv2DTranspose( nClasses , kernel_size=(64,64) ,  strides=(32,32), padding='same' , use_bias=False )(o)
    o_shape = Model(img_input , o ).output_shape
    
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    
    if nClasses>1:    
        o = (Reshape((  -1  , outputHeight*outputWidth  )))(o)
        o = (Permute((2, 1)))(o)
        o = (Activation('softmax'))(o)
    if nClasses==1:
        o = (Activation('sigmoid'))(o)

    model = Model( img_input , o )
    
    if W !='':
        model.load_weights(W)
    if nClasses>1:
        model.compile(loss="categorical_crossentropy", optimizer = Adam(lr = 1e-4) , metrics=['accuracy'] )
    if nClasses==1:
        model.compile(loss="binary_crossentropy", optimizer = Adam(lr = 1e-4) , metrics=['accuracy'] )
    model.summary()
    return model

def FCN8Net( nClasses ,  shape, W):
    vgg_level=3
    input_height=shape[0]
    input_width=shape[1]

	# assert input_height%32 == 0
	# assert input_width%32 == 0

	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
    img_input = Input(shape=(input_height,input_width,1))
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense( 1000 , activation='softmax', name='predictions')(x)
    o = f5
    o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same'))(o)
    o = Dropout(0.5)(o)
    o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same'))(o)
    o = Dropout(0.5)(o)
    o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' ))(o)
    o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False)(o)
    o2 = f4
    o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' ))(o2)
    o , o2 = crop( o , o2 , img_input )
    
    o = Add()([ o , o2 ])
    o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False )(o)
    o2 = f3 
    o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' ))(o2)
    o2 , o = crop( o2 , o , img_input )
    o  = Add()([ o2 , o ])
    o = Conv2DTranspose( nClasses , kernel_size=(16,16) ,  strides=(8,8) , use_bias=False )(o)
    
    o_shape = Model(img_input , o ).output_shape
    print(o_shape)

    outputHeight = o_shape[2]
    outputWidth = o_shape[3]
    
    if nClasses>1:    
        o = (Reshape((  -1  , outputHeight*outputWidth  )))(o)
        o = (Permute((2, 1)))(o)
        o = (Activation('softmax'))(o)
    if nClasses==1:
        o = (Activation('sigmoid'))(o)

    model = Model( img_input , o )
    
    if W !='':
        model.load_weights(W)
    if nClasses>1:
        model.compile(loss="categorical_crossentropy", optimizer = Adam(lr = 1e-4) , metrics=['accuracy'] )
    if nClasses==1:
        model.compile(loss="binary_crossentropy", optimizer = Adam(lr = 1e-4) , metrics=['accuracy'] )
    model.summary()
	
#	outputHeight = o_shape[2]
#	outputWidth = o_shape[3]
#
#	o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
#	o = (Permute((2, 1)))(o)
#	o = (Activation('softmax'))(o)
#	model = Model( img_input , o )
#	model.outputWidth = outputWidth
#	model.outputHeight = outputHeight

    return model