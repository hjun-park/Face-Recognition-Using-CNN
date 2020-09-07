
# coding: utf-8

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
#from scipy import misc
import scipy
import os
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import matplotlib.pyplot as plt

import sys

import math
import pickle
from sklearn.svm import SVC


## pre_process.py start
class preprocesses:
    def __init__(self, input_datadir, output_datadir):
        self.input_datadir = input_datadir
        self.output_datadir = output_datadir

    def collect_data(self):
        output_dir = os.path.expanduser(self.output_datadir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = facenet.get_dataset(self.input_datadir)
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, './npy')

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        image_size = 182

        # Add a random key to the filename to allow alignment using multiple processes
        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            for cls in dataset:
                output_class_dir = os.path.join(output_dir, cls.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + '.png')
                    print("Image: %s" % image_path)
                    if not os.path.exists(output_filename):
                        try:
                            #img = misc.imread(image_path)
                            img = cv2.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim < 2:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                                print('to_rgb data dimension: ', img.ndim)
                            img = img[:, :, 0:3]

                            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                                        factor)
                            nrof_faces = bounding_boxes.shape[0]
                            print('No of Detected Face: %d' % nrof_faces)
                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces > 1:
                                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                    index = np.argmax(
                                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                    det = det[index, :]
                                det = np.squeeze(det)
                                bb_temp = np.zeros(4, dtype=np.int32)

                                bb_temp[0] = det[0]
                                bb_temp[1] = det[1]
                                bb_temp[2] = det[2]
                                bb_temp[3] = det[3]

                                cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                                #scaled_temp = cv2.resize(cropped_temp, (image_size, image_size), 'bilinear')
                                scaled_temp = cv2.resize(cropped_temp, (image_size, image_size), interpolation = cv2.INTER_AREA)
                              

                                nrof_successfully_aligned += 1
                                plt.imsave(output_filename, scaled_temp)
                                text_file.write('%s %d %d %d %d\n' % (
                                output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                            else:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))

        return (nrof_images_total,nrof_successfully_aligned)
 ## preprocess.py end
    

## data_preprocess.py 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


input_datadir = './train_img'
output_datadir = './pre_img'

obj=preprocesses(input_datadir,output_datadir)
nrof_images_total,nrof_successfully_aligned=obj.collect_data()

print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

## data_preprocess.py end


## classfier.py
class training:
    def __init__(self, datadir, modeldir,classifier_filename):
        self.datadir = datadir
        self.modeldir = modeldir
        self.classifier_filename = classifier_filename

    def main_train(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                img_data = facenet.get_dataset(self.datadir)
                path, label = facenet.get_image_paths_and_labels(img_data)
                print('Classes: %d' % len(img_data))
                print('Images: %d' % len(path))

                facenet.load_model(self.modeldir)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                print('Extracting features of images for model')
                batch_size = 1000
                image_size = 160
                nrof_images = len(path)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, nrof_images)
                    paths_batch = path[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_file_name = os.path.expanduser(self.classifier_filename)

                # Training Started
                print('Training Started')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, label)

                class_names = [cls.name.replace('_', ' ') for cls in img_data]

                # Saving model
                with open(classifier_file_name, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                return classifier_file_name            
## classfier end



datadir = './pre_img'
modeldir = './model/20170511-185253.pb'
                    # CNN Model 
classifier_filename = './class/classifier.pkl'
print ("Training Start")
obj=training(datadir,modeldir,classifier_filename)
                     # training thing
get_file=obj.main_train()
print('Saved classifier model to file "%s"' % get_file)
sys.exit("All Done")


## train_main done 





    

