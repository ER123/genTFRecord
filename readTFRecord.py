"""#coding:utf-8
import tensorflow as tf
import numpy as np
import cv2
import os
import base64

#just for RNet and ONet, since I change the method of making tfrecord
#as for PNet
def read_single_tfrecord(tfrecord_file):
    # generate a input queue
    # each epoch shuffle
    filename_queue = tf.train.string_input_producer([tfrecord_file],shuffle=True)
    # read tfrecord
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),#one image  one record
            'label': tf.FixedLenFeature([], tf.int64),
            #'image/roi': tf.FixedLenFeature([4], tf.float32),
            #'image/landmark': tf.FixedLenFeature([10],tf.float32)
        }
    )
    image = tf.decode_raw(image_features['image'], tf.uint8)
    #print("image:",image)
    image = tf.reshape(image, [96, 96, 3])
    image = (tf.cast(image, tf.float32)-127.5) / 128
    
    # image = tf.image.per_image_standardization(image)
    label = tf.cast(image_features['label'], tf.float32)
    return image, label

def show_record(path):
    image, label = read_single_tfrecord(path)
    print("image:",image)
    print("label:",label)
    filename_queue = tf.train.string_input_producer([path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                        features={
                                              'image':tf.FixedLenFeature([],tf.string),
                                              'label':tf.FixedLenFeature([],tf.int64),
                                        })
    image = tf.decode_raw(features['image'],tf.uint8)
    image = tf.reshape(image, [96,96,3])
    label = tf.cast(features['label'], tf.int64)
    with tf.Session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      coord = tf.train.Coordinator()
      threads= tf.train.start_queue_runners(coord=coord)
      for i in range(100):
        example, label = sess.run([image, label])
        image = Image.fromarray(example, 'RGB')
        image.save(str(i)+'.jpg')
        print(example, l)
      coord.request_stop()
      coord.join(threads)

if __name__ == '__main__':
  path  = "G:/train.tfrecord_shuffle"
  show_record(path)"""

# -*- coding: utf-8 -*-
# 从 TFRecord 中读取并保存图片
import tensorflow as tf
import numpy as np


SAVE_PATH = 'G:/train.tfrecord_shuffle'


def load_data(width, high):
    reader = tf.TFRecordReader()
    print("SAVE_PATH:",SAVE_PATH)
    filename_queue = tf.train.string_input_producer([SAVE_PATH])

    # 从 TFRecord 读取内容并保存到 serialized_example 中
    _, serialized_example = reader.read(filename_queue)
    # 读取 serialized_example 的格式
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # 解析从 serialized_example 读取到的内容
    images = tf.decode_raw(features['image'], tf.uint8)
    labels = tf.cast(features['label'], tf.int64)

    with tf.Session() as sess:
        # 启动多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 因为我这里只有 2 张图片，所以下面循环 2 次
        for i in range(20):
            # 获取一张图片和其对应的类型
            label, image = sess.run([labels, images])
            # 这里特别说明下：
            #   因为要想把图片保存成 TFRecord，那就必须先将图片矩阵转换成 string，即：
            #       pic2tfrecords.py 中 image_raw = image.tostring() 这行
            #   所以这里需要执行下面这行将 string 转换回来，否则会无法 reshape 成图片矩阵，请看下面的小例子：
            #       a = np.array([[1, 2], [3, 4]], dtype=np.int64) # 2*2 的矩阵
            #       b = a.tostring()
            #       # 下面这行的输出是 32，即： 2*2 之后还要再乘 8
            #       # 如果 tostring 之后的长度是 2*2=4 的话，那可以将 b 直接 reshape([2, 2])，但现在的长度是 2*2*8 = 32，所以无法直接 reshape
            #       # 同理如果你的图片是 500*500*3 的话，那 tostring() 之后的长度是 500*500*3 后再乘上一个数
            #       print len(b)
            #
            #   但在网上有很多提供的代码里都没有下面这一行，你们那真的能 reshape ?
            image = np.fromstring(image, dtype=np.float32)
            # reshape 成图片矩阵
            image = tf.reshape(image, [96, 96, 3])
            # 因为要保存图片，所以将其转换成 uint8
            image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
            # 按照 jpeg 格式编码
            image = tf.image.encode_jpeg(image)
            # 保存图片
            with tf.gfile.GFile('pic_%d.jpg' % label, 'wb') as f:
                f.write(sess.run(image))


load_data(96, 96)
