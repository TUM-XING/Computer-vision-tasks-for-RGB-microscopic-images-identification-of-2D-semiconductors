from keras.models import Model, load_model
from keras.layers import *
from keras import backend as K

K.set_image_data_format('channels_last')

def expand(x):
    x = K.expand_dims(x, axis=-1)
    return x

def squeeze(x):
    x = K.squeeze(x, axis=-1)
    return x

#2D卷积模块
def BN_block(filter_num, input):
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


#3D卷积模块
#def BN_block3d(filter_num, input):
#    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
#    x = BatchNormalization()(x)
#    x1 = Activation('relu')(x)
#    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
#    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
#    return x

#融合策略1 3D特征降成2D后，直接与2D特征相加
#def D_Add(filter_num, input3d, input2d):
#    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
#    x = Lambda(squeeze)(x)
#    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
#    x = Add()([x, input2d])
#    return x

#融合策略2 3D特征降成2D后，直接与2D特征合并
#def D_concat(filter_num, input3d, input2d):
#    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
#    x = Lambda(squeeze)(x)
#    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
#    x = Concatenate()([x, input2d])
#    x = Conv2D(filter_num, 1, padding='same', activation='relu', kernel_initializer='he_normal')(x)
#    return x

#融合策略3 降维后的3D特征与2D特征各自经过sqeeze-excite-block后再合并
#def D_SE_concat(filter_num, input3d, input2d):
#    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
#    x = Lambda(squeeze)(x)
 #   x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
 #   x = squeeze_excite_block(x)
 #   input2d = squeeze_excite_block(input2d)
 #   x = Concatenate()([x, input2d])
 #   x = Conv2D(filter_num, 1, padding='same', activation='relu', kernel_initializer='he_normal')(x)
 #   return x

#融合策略4 降维后的3D特征先和2D特征相加，再经过sqeeze-excite-block输出
#def D_Add_SE(filter_num, input3d, input2d):
 #   x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
 #   x = Lambda(squeeze)(x)
 #   x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
  #  x = Add()([x, input2d])
  #  x = squeeze_excite_block(x)
  #  return x

#融合策略5 降维后的3D特征与2D特征各自经过sqeeze-excite-block后再相加
#def D_SE_Add(filter_num, input3d, input2d):
 #   x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
 #   x = Lambda(squeeze)(x)
  #  x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
  #  x = squeeze_excite_block(x)
  #  input2d = squeeze_excite_block(input2d)
  #  x = Add()([x, input2d])

   # return x

#融合策略6 降维后的3D特征先和2D特征合并，再经过sqeeze-excite-block输出
#def D_concat_SE(filter_num, input3d, input2d):
 #   x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
 #   x = Lambda(squeeze)(x)
  #  x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
  #  x = Concatenate()([x, input2d])
  #  x = squeeze_excite_block(x)
  #  x = Conv2D(filter_num, 1, padding='same', activation='relu', kernel_initializer='he_normal')(x)
  #  return x

#定义 Squeeze-excite-block
def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

#定义双支路U-net
#def D_Unet(numClass):
#    inputs = Input(shape=(96, 96, 254))
#    inputRGB=Lambda(lambda x:x[:,:,:,:3])(inputs)
#    inputHSI=Lambda(lambda x:x[:,:,:,3:])(inputs)

#    input3d = Lambda(expand)(inputHSI)
#    conv3d1 = BN_block3d(32, input3d)

#    pool3d1 = MaxPooling3D(pool_size=2)(conv3d1)

#    conv3d2 = BN_block3d(64, pool3d1)

 #   pool3d2 = MaxPooling3D(pool_size=2)(conv3d2)

 #   conv3d3 = BN_block3d(128, pool3d2)


  #  conv1 = BN_block(32, inputRGB)
    #conv1 = D_Add(32, conv3d1, conv1)
  #  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

   # conv2 = BN_block(64, pool1)

  #  conv2 = D_SE_concat(64, conv3d2, conv2)
    #conv2 = D_SE_Add(64, conv3d2, conv2)
  #  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

   # conv3 = BN_block(128, pool2)

  #  conv3 = D_SE_concat(128, conv3d3, conv3)
  #  #conv3 = D_SE_Add(128, conv3d3, conv3)
  #  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

''' conv4 = BN_block(256, pool3)
    conv4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = BN_block(512, pool4)
    conv5 = Dropout(0.3)(conv5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate()([conv4, up6])
    conv6 = BN_block(256, merge6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block(128, merge7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block(64, merge8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block(32, merge9)
    outputs = Conv2D(numClass, 1, activation='softmax')(conv9)
    model = Model(inputs=[inputs], outputs=[outputs])

    return model '''

#定义单支路U-net
def Unet(numClass):
    inputs = Input(shape=(256, 256, 3))
    conv1 = BN_block(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BN_block(64, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BN_block(128, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = BN_block(256, pool3)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = BN_block(512, pool4)
    drop5 = Dropout(0.3)(conv5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate()([drop4, up6])
    conv6 = BN_block(256, merge6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block(128, merge7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block(64, merge8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block(32, merge9)
    conv10 = Conv2D(numClass, 1, activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

