import numpy as np
from scipy import ndimage, misc
from imgaug import augmenters as iaa

# crop
def get_patches(img_arr, size=256, stride=256):

    patches_list = []
    overlapping = 0

    if size % stride != 0:
        raise ValueError("size % stride must be equal 0")
    if stride != size:
        overlapping = (size // stride) - 1
    if img_arr.ndim == 4:
        i_max = img_arr.shape[1] // stride - overlapping
        for im in img_arr:
            for i in range(i_max):
                for j in range(i_max):
                    # print(i*stride, i*stride+size)
                    # print(j*stride, j*stride+size)
                    patches_list.append(
                        im[
                        i * stride: i * stride + size,
                        j * stride: j * stride + size,
                        ]
                    )

    else:
        raise ValueError("img_arr.ndim must be equal 4")

    return np.stack(patches_list)

# reconstruct
def reconstruct_from_patches(img_arr, org_img_size, stride=None, size=None):
    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    nm_layers = img_arr.shape[3]

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    total_nm_images = img_arr.shape[0] // (i_max ** 2)
    nm_images = img_arr.shape[0]

    averaging_value = size // stride
    images_list = []
    kk = 0
    for img_count in range(total_nm_images):
        img_bg = np.zeros(
            (org_img_size[0], org_img_size[1], nm_layers), dtype=img_arr[0].dtype
        )

        for i in range(i_max):
            for j in range(j_max):
                for layer in range(nm_layers):
                    img_bg[
                    i * stride: i * stride + size,
                    j * stride: j * stride + size,
                    layer,
                    ] = img_arr[kk, :, :, layer]

                kk += 1


        images_list.append(img_bg)

    return np.stack(images_list)



# 把普通的label segmentation转换成onehot-label
# onehot-label 的channel数对应class的数量
def convert_to_onehot(label,numClass):
    one_hot=np.zeros((1,label.shape[0],label.shape[1],numClass),dtype=np.float32)
    for i in range (numClass):
        one_hot[0,:, :, i][label == i] = 1

    return one_hot



# 中值滤波模块
# 对分割patch的预测结果使用中值滤波可消除patch边缘的干扰
def median_f(img,size):
    for i in range(img.shape[0]):
        for j in range(img.shape[3]):
            img[i,:,:,j]=ndimage.median_filter(img[i,:,:,j], size)

    return img

# 对数据做augmentation
# 所有数据随机旋转aug=3次，旋转角度[0,90]之间
# 所有数据水平翻转一次

def load_data_aug(x_train, y_train,aug=3,channels=8,num_class=3,size=48):
    imgs=[]
    labels=[]
    num=x_train.shape[0]
    for i in range(aug):
        for j in range(num):
            t = np.random.rand() * 90
            x_train_tmp1=x_train[j,:,:,:]
            y_train_tmp1=y_train[j,:,:,:]
            rotate = iaa.Affine(rotate=(t, t))
            x_train_rotate = rotate.augment_image(x_train_tmp1)
            y_train_rotate = rotate.augment_image(y_train_tmp1)
            imgs.append(x_train_rotate)
            labels.append(y_train_rotate)

    for k in range(num):
        x_train_tmp2 = x_train[k, :, :, :]
        y_train_tmp2 = y_train[k, :, :, :]
        imgs.append(x_train_tmp2)
        labels.append(y_train_tmp2)
        flip = iaa.Fliplr(1)
        x_train_flip = flip.augment_image(x_train_tmp2)
        y_train_flip = flip.augment_image(y_train_tmp2)
        imgs.append(x_train_flip)
        labels.append(y_train_flip)

    return np.stack(imgs),np.stack(labels)




# 将binary的mask 转换成rgba的彩图
def mask_to_rgba(mask, color="red"):

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones, ones), axis=-1)
    elif color == "gray": #background
        return np.stack((ones/2, ones/2, ones/2, ones), axis=-1)
    elif color == "royalblue": #1.layer
        return np.stack((ones/256*61, ones/256*165, ones/256*217, ones), axis=-1)
    elif color == "grassgreen": #2.layer
        return np.stack((ones/256*115, ones/256*191, ones/256*184, ones), axis=-1)
    elif color == "darkblue": #3.layer
        return np.stack((ones/256*35, ones/256*100, ones/256*170, ones), axis=-1)
    elif color == "gold": #4.layer
        return np.stack((ones/256*254, ones/256*198, ones/256*1, ones), axis=-1)
    elif color == "orange": #5.layer
        return np.stack((ones/256*234, ones/256*115, ones/256*23, ones), axis=-1)







