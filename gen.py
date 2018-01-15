import tensorflow as tf 
import numpy as np 
import random
import cv2
import os
import sys
import argparse
import struct

from tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple, _convert_to_example

def _get_output_filename(output_dir):
    #st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #return '%s/%s_%s_%s.tfrecord' % (output_dir, name, net, st)
    return '%s/train.tfrecord' % (output_dir)

def get_dataset(dataset_dir, label_path):
    print (label_path)
    path = os.path.join(dataset_dir, label_path)
    print("path:",path)
    imagelist = open(path, 'r')

    dataset = []
    label_list = []
    idx = 0
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        file_Path = os.path.join(dataset_dir,info[0])
        data_example['filename'] = file_Path
        data_example['label'] = int(info[1])
        #print("label_path:",file_Path)
        #label_name = os.path.splitext(file_Path)
        #print("label_name:",label_name)
        #label_name = label_name[0] + '.dat1'
        #f = open(label_name, 'rb')
        #all_data = f.read()

        #for i in range(2,3):
        #    labels_elem, = struct.unpack('f', all_data[i*4:i*4+4])
        #    label_list.append(labels_elem)

        #data_example['label'] = label_list

        #label, = struct.unpack('f', all_data[8:12])
        print("label:",int(info[1]))
        #data_example['label'] = int(info[1])

        dataset.append(data_example)
    imagelist.close()
    #print(dataset)
    return dataset

def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    print('---', filename)
    #imaga_data:array to string
    #height:original image's height
    #width:original image's width
    #image_example dict contains image's info
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    #example = _convert_to_example(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())

def run(dataset_dir, output_dir, label, name='MTCNN', shuffling=True):
    """Runs the conversion operation.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """    
    #tfrecord name 
    tf_filename = _get_output_filename(output_dir)
    print("tf_filename::",tf_filename)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # GET Dataset, and shuffling.
    dataset = get_dataset(dataset_dir, label)
    #print("dataset:",dataset)
    # filenames = dataset['filename']
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        #andom.seed(12345454)
        random.shuffle(dataset)
    # Process dataset files.
    # write the data to tfrecord
    print ('<<<<<<<<<<<<<<<  START CONVERT  >>>>>>>>>>>>>>>>>>')
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(dataset)))
            sys.stdout.flush()
            filename = image_example['filename']
            _add_to_tfrecord(filename, image_example, tfrecord_writer)
    tfrecord_writer.close()
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting dataset!')

def parse_args():
    parser = argparse.ArgumentParser(description='Generate tfrecord')
    parser.add_argument('--dir', dest='dir', help='pictures path', type=str)
    parser.add_argument('--output', dest='output', help='save path', type=str)
    parser.add_argument('--label', dest='label', help = 'label path', type=str)
    parser.add_argument('--shuffling', dest='shuffling', help='shuffling', default=True, type=bool)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    run(args.dir, args.output, args.label, shuffling=True)