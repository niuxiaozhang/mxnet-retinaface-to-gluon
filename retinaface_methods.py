from __future__ import division

import os
import numpy as np
import cv2
import mxnet.ndarray as nd
import mxnet as mx
from retinaface import RetinaFace


class RetinaFaceDetect(object):
    def __init__(self, gpu=0):
        self._gpu = gpu
        self._thresh = 0.8
        self._model = self.load_model()
        self.pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.pixel_scale = float(1.0)
        self._model = self.load_model()

    def load_model(self):
        '''
        load net and weights params
        :return: net and weights parms
        '''
        sep = os.sep
        file_abs = os.path.abspath(__file__)
        file_path = file_abs[0:file_abs.rfind(sep)]
        model_path = file_path + sep + '../../models/RetinaFace/R50/R50'
        detector = RetinaFace('./model/test', 0, self._gpu, 'net3')
        return detector

    def img_process(self, img, is_resize=True, target_size=512):
        if is_resize:
            im_shape = img.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > target_size:
                im_scale = float(target_size) / float(im_size_max)

            im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            im = im.astype(np.float32)
            im_pad = np.zeros((target_size, target_size, im.shape[2]), dtype=np.float32)
            im_pad[0:im.shape[0], 0:im.shape[1], :] = im
        else:
            im_pad = img
            im_scale = 1

        im_tensor = np.zeros((1, 3, im_pad.shape[0], im_pad.shape[1]), dtype=np.float32)
        for i in range(3):
            im_tensor[0, i, :, :] = (im_pad[:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i]) / \
                                    self.pixel_stds[2 - i]
        return im_tensor, im_scale

    def detect(self, img, is_resize=True, target_size=512):
        results = []
        im_tensor, im_scale = self.img_process(img, is_resize, target_size)

        faces, landmarks = self._model.detect(im_tensor, self._thresh, im_scale)
        if faces is None or landmarks is None:
            return results
        for face, landmark in zip(faces, landmarks):
            face = face.astype(np.int)
            landmark = landmark.astype(np.int)
            results.append([face, landmark])
        return results



if __name__ == '__main__':
    import glob
    save_path = 'result/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    detector = RetinaFaceDetect(gpu=0)

    sep = os.sep
    file_abs = os.path.abspath(__file__)
    file_path = file_abs[0:file_abs.rfind(sep)]
    imgs = glob.glob('./sample-images/*.jpg')

    for img in imgs:
        name = img.split('/')[-1]
        img = cv2.imread(img)
        results = detector.detect(img, is_resize=True, target_size=512)
        print('find', len(results), 'faces')
        if len(results) == 0:
            continue
        for result in results:
            face = result[0]
            landmark = result[1]

            color = (0, 0, 255)
            cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), color, 2)

            for l in range(landmark.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(img, (landmark[l][0], landmark[l][1]), 1, color, 2)

        cv2.imwrite(save_path + name, img)

