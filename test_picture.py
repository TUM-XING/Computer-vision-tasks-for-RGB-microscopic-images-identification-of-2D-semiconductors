import numpy as np
import os
from PIL import Image
from dataProc import mask_to_rgba
from model import *
import cv2
import io
import PIL
import matplotlib.pyplot as plt

#line 62-172 load different models

def test_picture(size = 256, stride = 256, PATH = None, sample = None, substrates = None, objective = None):
    #cut***********************
    TEST_PATH = PATH
    img = np.asarray(Image.open(os.path.join(TEST_PATH)))

    if len(img.shape) > 2 and img.shape[2] == 4:
        # convert the image from RGBA2RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    overlapping = 0
    test_pic = []
    #********************
    i = j = 0
    init_width = img.shape[1]
    init_height = img.shape[0]
    if img.shape[0] % size != 0 and img.shape[1] % size == 0:
        i = img.shape[0] // size + 1
    elif img.shape[0] % size == 0 and img.shape[1] % size != 0:
        j = img.shape[1] // size + 1
    elif img.shape[0] % size != 0 and img.shape[1] % size != 0:
        i = img.shape[0] // size + 1
        j = img.shape[1] // size + 1
    if i != 0 or j != 0:
        resize = np.zeros((i * size, j * size, 3))
        for layer in range(3):
            resize[0:img.shape[0], 0:img.shape[1], layer] = img[:, :, layer]
        img = resize
    #***********************

    if size % stride != 0:
        raise ValueError("size % stride must be equal 0")
    if stride != size:
        overlapping = (size // stride) - 1
    if img.ndim == 3:
        i_max = img.shape[0] // stride - overlapping
        j_max = img.shape[1] // stride - overlapping

        for ii in range(i_max):
            for jj in range(j_max):
                patch = img[ii * stride: ii * stride + size, jj * stride: jj * stride + size, ]
                test_pic.append(patch)

        #test + reconstruct***********************

        #select model********
        Sample = sample
        Substrates = substrates
        Objective = objective
        global model
        if Sample == 'MoS2' and Substrates == '70nm' and Objective == 10:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'MoS2' and Substrates == '70nm' and Objective == 20:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'MoS2' and Substrates == '70nm' and Objective == 50:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'MoS2' and Substrates == '70nm' and Objective == 100:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'MoS2' and Substrates == '270nm' and Objective == 10:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'MoS2' and Substrates == '270nm' and Objective == 20:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'MoS2' and Substrates == '270nm' and Objective == 50:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'MoS2' and Substrates == '270nm' and Objective == 100:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'MoS2' and Substrates == 'Sapphire' and Objective == 10:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'MoS2' and Substrates == 'Sapphire' and Objective == 20:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'MoS2' and Substrates == 'Sapphire' and Objective == 50:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'MoS2' and Substrates == 'Sapphire' and Objective == 100:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')

        elif Sample == 'WS2' and Substrates == '70nm' and Objective == 10:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WS2' and Substrates == '70nm' and Objective == 20:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WS2' and Substrates == '70nm' and Objective == 50:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WS2' and Substrates == '70nm' and Objective == 100:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WS2' and Substrates == '270nm' and Objective == 10:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WS2' and Substrates == '270nm' and Objective == 20:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WS2' and Substrates == '270nm' and Objective == 50:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WS2' and Substrates == '270nm' and Objective == 100:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WS2' and Substrates == 'Sapphire' and Objective == 10:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WS2' and Substrates == 'Sapphire' and Objective == 20:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WS2' and Substrates == 'Sapphire' and Objective == 50:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WS2' and Substrates == 'Sapphire' and Objective == 100:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')

        elif Sample == 'WeS2' and Substrates == '70nm' and Objective == 10:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WeS2' and Substrates == '70nm' and Objective == 20:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WeS2' and Substrates == '70nm' and Objective == 50:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WeS2' and Substrates == '70nm' and Objective == 100:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WeS2' and Substrates == '270nm' and Objective == 10:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WeS2' and Substrates == '270nm' and Objective == 20:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WeS2' and Substrates == '270nm' and Objective == 50:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WeS2' and Substrates == '270nm' and Objective == 100:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WeS2' and Substrates == 'Sapphire' and Objective == 10:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WeS2' and Substrates == 'Sapphire' and Objective == 20:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WeS2' and Substrates == 'Sapphire' and Objective == 50:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        elif Sample == 'WeS2' and Substrates == 'Sapphire' and Objective == 100:
            model = Unet(6)
            model.load_weights('models/cvd_singleU_soft_contrast.h5')
        #********************

        NUM_class = 6

        n_pred = [0] * NUM_class
        nm_layers = 3
        img_reconstruct = np.zeros((size * i_max, size * j_max, nm_layers))

        for n in range(i_max * j_max):
            patch = np.expand_dims(test_pic[n], 0)
            #print(patch.shape)

            pred_test = model.predict(patch, verbose=1)
            pred_test_t = pred_test.argmax(axis=-1)

            pred_0 = np.zeros(np.shape(pred_test_t))
            pred_0[pred_test_t == 0] = 1
            #pred_0 = CCA_postprocessing(np.uint8(np.squeeze(pred_0)))
            pred_0 = np.squeeze(pred_0, axis = 0)

            pred_1 = np.zeros(np.shape(pred_test_t))
            pred_1[pred_test_t == 1] = 1
            #pred_1 = CCA_postprocessing(np.uint8(np.squeeze(pred_1)))
            pred_1 = np.squeeze(pred_1, axis=0)

            pred_2 = np.zeros(np.shape(pred_test_t))
            pred_2[pred_test_t == 2] = 1
            #pred_2 = CCA_postprocessing(np.uint8(np.squeeze(pred_2)))
            pred_2 = np.squeeze(pred_2, axis=0)

            pred_3 = np.zeros(np.shape(pred_test_t))
            pred_3[pred_test_t == 3] = 1
            #pred_3 = CCA_postprocessing(np.uint8(np.squeeze(pred_3)))
            pred_3 = np.squeeze(pred_3, axis=0)

            pred_4 = np.zeros(np.shape(pred_test_t))
            pred_4[pred_test_t == 4] = 1
            #pred_4 = CCA_postprocessing(np.uint8(np.squeeze(pred_4)))
            pred_4 = np.squeeze(pred_4, axis=0)

            pred_5 = np.zeros(np.shape(pred_test_t))
            pred_5[pred_test_t == 5] = 1
            #pred_5 = CCA_postprocessing(np.uint8(np.squeeze(pred_5)))
            pred_5 = np.squeeze(pred_5, axis=0)

            pred_onehot = np.zeros((1, 256, 256, NUM_class), dtype=np.int8)
            pred_onehot[:, :, :, 0] = pred_0
            pred_onehot[:, :, :, 1] = pred_1
            pred_onehot[:, :, :, 2] = pred_2
            pred_onehot[:, :, :, 3] = pred_3
            pred_onehot[:, :, :, 4] = pred_4
            pred_onehot[:, :, :, 5] = pred_5

            pre_class_0 = pred_onehot[0, :, :, 0].astype(int)
            pre_class_1 = pred_onehot[0, :, :, 1].astype(int)
            pre_class_2 = pred_onehot[0, :, :, 2].astype(int)
            pre_class_3 = pred_onehot[0, :, :, 3].astype(int)
            pre_class_4 = pred_onehot[0, :, :, 4].astype(int)
            pre_class_5 = pred_onehot[0, :, :, 5].astype(int)

            n_pred[0] += (pre_class_0 == 1).sum()
            n_pred[1] += (pre_class_1 == 1).sum()
            n_pred[2] += (pre_class_2 == 1).sum()
            n_pred[3] += (pre_class_3 == 1).sum()
            n_pred[4] += (pre_class_4 == 1).sum()
            n_pred[5] += (pre_class_5 == 1).sum()

            plt.figure()
            plt.imshow(mask_to_rgba(pred_0, "gray"))
            plt.imshow(mask_to_rgba(pred_1, "royalblue"))
            plt.imshow(mask_to_rgba(pred_2, "grassgreen"))
            plt.imshow(mask_to_rgba(pred_3, "darkblue"))
            plt.imshow(mask_to_rgba(pred_4, "gold"))
            plt.imshow(mask_to_rgba(pred_5, "orange"))
            plt.axis('off')
            fig = plt.gcf()
            fig.set_size_inches(2.56, 2.56)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.xticks([])
            plt.yticks([])

            buffer_ = io.BytesIO()
            fig.savefig(buffer_, format="png")
            buffer_.seek(0)
            image = PIL.Image.open(buffer_)

            # 转换为numpy array
            ar = np.asarray(image)
            i_1 = n // j_max
            j_1 = n % j_max
            for layer in range(nm_layers):
                img_reconstruct[i_1 * stride: i_1 * stride + size,
                        j_1 * stride: j_1 * stride + size,
                        layer, ] = ar[:, :, layer]
            # 释放缓存
            buffer_.close()

        sum = n_pred[0] + n_pred[1] + n_pred[2] + n_pred[3] + n_pred[4] + n_pred[5]

        #*********************
        if i != 0 or j != 0:
            recons = np.zeros((init_height, init_width, 3))
            for layer in range(3):
                recons[:, :, layer] = img_reconstruct[0:init_height, 0:init_width, layer]
            img_reconstruct = recons
            n_pred[0] = n_pred[0] - (i * j * size * size - init_width * init_height)
            sum = sum - (i * j * size * size - init_width * init_height)
        #*********************

        img_reconstruct = Image.fromarray(img_reconstruct.astype('uint8')).convert('RGB')

        return img_reconstruct, n_pred, sum




