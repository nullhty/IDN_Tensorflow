# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 11:38:32 2018
@author: hty
"""
from utility import *
import os
import scipy.io as sio
import numpy as np
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)


def file_name(file_dir, f):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == f:
                L.append(os.path.join(root, file))
    return L


def compute_psnr(target, ref):
    # assume RGB image
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))
    p = 20 * math.log10(255. / rmse)
    return p


def main():
    database = "Set5"
    scale = 2
    root_path = './IDN_checkpoint5/'
    with tf.Session(config=config) as sess:

        LR_path = file_name("./" + database + "/LRX" + str(scale), ".mat")
        Gnd_path = file_name("./" + database + "/Gnd", ".mat")

        start_model = 1
        step = 1
        model_num = 94
        images = tf.placeholder(tf.float32, [1, None, None, 1], name='images')

        pred = IDN(images, scale)
        # with open('result.txt','w') as f:#w这个参数会清空result文件，再写入；若要让内容不清空，则使用参数a
        #    f.write("******************************")
        #    f.write("\n write now!\n")
        for model in range(1, 1 + model_num):
            p2 = 0.0
            check_point_path = root_path + str(start_model + step * (model - 1)) + '/'
            print('*' * 30)
            print('epoch : ' + str(start_model + step * (model - 1)))
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

            # with open('result.txt','a') as f:#设置文件对象
            #    f.write("\nepoch: %d\n" %(start_model + step * (model-1)))

            for i in range(len(LR_path)):
                low = sio.loadmat(LR_path[i])['im_b']
                original = sio.loadmat(Gnd_path[i])['im_gnd']
                shape = low.shape
                image_low = np.zeros((1, shape[0], shape[1], 1), dtype=float)
                image_low[0, :, :, 0] = low

                pred0 = sess.run([pred], feed_dict={images: image_low})
                output = pred0[0]
                pre1 = output * 255
                pre1[pre1[:] > 255] = 255
                pre1[pre1[:] < 0] = 0
                image_high = pre1[0, :, :, 0]
                image_high = np.round(image_high)
                image_high2 = image_high[scale:shape[0] - scale, scale:shape[1] - scale]

                original2 = original * 255.
                original2[original2[:] > 255] = 255
                original2[original2[:] < 0] = 0
                original2 = np.round(original2)
                original2 = original2[scale:shape[0] - scale, scale:shape[1] - scale]

                pp2 = compute_psnr(image_high2, original2)
                print(str(i + 1) + ". " + str(pp2))
                # with open('result.txt','a') as f:
                #    f.write("%d. bicubic: %s, srcnn: %s\n" %(i+1, str(pp1), str(pp1)))
                p2 = p2 + pp2
            print "IDN average psnr : ", p2 / len(LR_path)
            # with open('result.txt','a') as f:
            #    f.write("bicubic psnr =%s\n" %(str(p1/len(LR_path))))
            #    f.write("srcnn psnr = %s\n" %(str(p2/len(LR_path))))


if __name__ == '__main__':
    main()