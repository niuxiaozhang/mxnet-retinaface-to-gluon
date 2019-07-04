from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper

class RetinaFace:
  def __init__(self, ctx_id=0, network='net3', nms=0.4, nocrop=False, decay4 = 0.5, vote=False):
    self.ctx_id = ctx_id
    self.network = network
    self.decay4 = decay4
    self.nms_threshold = nms
    self.vote = vote
    self.nocrop = nocrop
    self.debug = False
    self.fpn_keys = []
    self.anchor_cfg = None
    self.preprocess = False
    _ratio = (1.,)
    fmc = 3
    if network=='ssh' or network=='vgg':
      self.preprocess = True
    elif network=='net3':
      _ratio = (1.,)
    elif network=='net3a':
      _ratio = (1.,1.5)
    elif network=='net6': #like pyramidbox or s3fd
      fmc = 6
    elif network=='net5': #retinaface
      fmc = 5
    elif network=='net5a':
      fmc = 5
      _ratio = (1.,1.5)
    elif network=='net4':
      fmc = 4
    elif network=='net4a':
      fmc = 4
      _ratio = (1.,1.5)
    else:
      assert False, 'network setting error %s'%network

    if fmc==3:
      self._feat_stride_fpn = [32, 16, 8]
      self.anchor_cfg = {
          '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
      }
    elif fmc==4:
      self._feat_stride_fpn = [32, 16, 8, 4]
      self.anchor_cfg = {
          '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '4': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
      }
    elif fmc==6:
      self._feat_stride_fpn = [128, 64, 32, 16, 8, 4]
      self.anchor_cfg = {
          '128': {'SCALES': (32,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '64': {'SCALES': (16,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '32': {'SCALES': (8,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '16': {'SCALES': (4,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '8': {'SCALES': (2,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '4': {'SCALES': (1,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
      }
    elif fmc==5:
      self._feat_stride_fpn = [64, 32, 16, 8, 4]
      self.anchor_cfg = {}
      _ass = 2.0**(1.0/3)
      _basescale = 1.0
      for _stride in [4, 8, 16, 32, 64]:
        key = str(_stride)
        value = {'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999}
        scales = []
        for _ in range(3):
          scales.append(_basescale)
          _basescale *= _ass
        value['SCALES'] = tuple(scales)
        self.anchor_cfg[key] = value

    print(self._feat_stride_fpn, self.anchor_cfg)

    for s in self._feat_stride_fpn:
        self.fpn_keys.append('stride%s'%s)

    dense_anchor = False
    self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(dense_anchor=dense_anchor, cfg=self.anchor_cfg)))
    for k in self._anchors_fpn:
      v = self._anchors_fpn[k].astype(np.float32)
      self._anchors_fpn[k] = v

    self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
    if self.ctx_id>=0:
      self.ctx = mx.gpu(self.ctx_id)
      self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)
    else:
      self.ctx = mx.cpu()
      self.nms = cpu_nms_wrapper(self.nms_threshold)
    self.use_landmarks = True


  def detect(self, img, net_out, threshold=0.5, im_scale=1.0):
    proposals_list = []
    scores_list = []
    landmarks_list = []
    for _idx,s in enumerate(self._feat_stride_fpn):
        _key = 'stride%s'%s
        stride = int(s)
        if self.use_landmarks:
          idx = _idx*3
        else:
          idx = _idx*2
        scores = net_out[idx].asnumpy()
        scores = scores[:, self._num_anchors['stride%s'%s]:, :, :]

        idx+=1
        bbox_deltas = net_out[idx].asnumpy()

        height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

        A = self._num_anchors['stride%s'%s]
        K = height * width
        anchors_fpn = self._anchors_fpn['stride%s'%s]
        anchors = anchors_plane(height, width, stride, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = self._clip_pad(scores, (height, width))
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
        bbox_pred_len = bbox_deltas.shape[3]//A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))

        proposals = self.bbox_pred(anchors, bbox_deltas)
        proposals = clip_boxes(proposals, (img.shape[2], img.shape[3]))

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel>=threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]
        if stride==4 and self.decay4<1.0:
          scores *= self.decay4

        proposals[:,0:4] /= im_scale

        proposals_list.append(proposals)
        scores_list.append(scores)

        if not self.vote and self.use_landmarks:
          idx+=1
          landmark_deltas = net_out[idx].asnumpy()
          landmark_deltas = self._clip_pad(landmark_deltas, (height, width))
          landmark_pred_len = landmark_deltas.shape[1]//A
          landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len//5))
          landmarks = self.landmark_pred(anchors, landmark_deltas)
          landmarks = landmarks[order, :]
          landmarks[:,:,0:2] /= im_scale
          landmarks_list.append(landmarks)

    proposals = np.vstack(proposals_list)
    landmarks = None
    if proposals.shape[0]==0:
      if self.use_landmarks:
        landmarks = np.zeros( (0,5,2) )
      return np.zeros( (0,5) ), landmarks
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    proposals = proposals[order, :]
    scores = scores[order]
    if not self.vote and self.use_landmarks:
      landmarks = np.vstack(landmarks_list)
      landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)
    if not self.vote:
      keep = self.nms(pre_det)
      det = np.hstack( (pre_det, proposals[:,4:]) )
      det = det[keep, :]
      if self.use_landmarks:
        landmarks = landmarks[keep]
    else:
      det = np.hstack( (pre_det, proposals[:,4:]) )
      det = self.bbox_vote(det)
    return det, landmarks


  @staticmethod
  def _clip_pad(tensor, pad_shape):
      """
      Clip boxes of the pad area.
      :param tensor: [n, c, H, W]
      :param pad_shape: [h, w]
      :return: [n, c, h, w]
      """
      H, W = tensor.shape[2:]
      h, w = pad_shape

      if h < H or w < W:
        tensor = tensor[:, :, :h, :w].copy()

      return tensor

  @staticmethod
  def bbox_pred(boxes, box_deltas):
      """
      Transform the set of class-agnostic boxes into class-specific boxes
      by applying the predicted offsets (box_deltas)
      :param boxes: !important [N 4]
      :param box_deltas: [N, 4 * num_classes]
      :return: [N 4 * num_classes]
      """
      if boxes.shape[0] == 0:
          return np.zeros((0, box_deltas.shape[1]))

      boxes = boxes.astype(np.float, copy=False)
      widths = boxes[:, 2] - boxes[:, 0] + 1.0
      heights = boxes[:, 3] - boxes[:, 1] + 1.0
      ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
      ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

      dx = box_deltas[:, 0:1]
      dy = box_deltas[:, 1:2]
      dw = box_deltas[:, 2:3]
      dh = box_deltas[:, 3:4]

      pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
      pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
      pred_w = np.exp(dw) * widths[:, np.newaxis]
      pred_h = np.exp(dh) * heights[:, np.newaxis]

      pred_boxes = np.zeros(box_deltas.shape)
      # x1
      pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
      # y1
      pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
      # x2
      pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
      # y2
      pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

      if box_deltas.shape[1]>4:
        pred_boxes[:,4:] = box_deltas[:,4:]

      return pred_boxes

  @staticmethod
  def landmark_pred(boxes, landmark_deltas):
      if boxes.shape[0] == 0:
          return np.zeros((0, landmark_deltas.shape[1]))
      boxes = boxes.astype(np.float, copy=False)
      widths = boxes[:, 2] - boxes[:, 0] + 1.0
      heights = boxes[:, 3] - boxes[:, 1] + 1.0
      ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
      ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
      pred = landmark_deltas.copy()
      for i in range(5):
        pred[:,i,0] = landmark_deltas[:,i,0]*widths + ctr_x
        pred[:,i,1] = landmark_deltas[:,i,1]*heights + ctr_y
      return pred
      #preds = []
      #for i in range(landmark_deltas.shape[1]):
      #  if i%2==0:
      #    pred = (landmark_deltas[:,i]*widths + ctr_x)
      #  else:
      #    pred = (landmark_deltas[:,i]*heights + ctr_y)
      #  preds.append(pred)
      #preds = np.vstack(preds).transpose()
      #return preds

  def bbox_vote(self, det):
      #order = det[:, 4].ravel().argsort()[::-1]
      #det = det[order, :]
      if det.shape[0] == 0:
          dets = np.array([[10, 10, 20, 20, 0.002]])
          det = np.empty(shape=[0, 5])
      while det.shape[0] > 0:
          # IOU
          area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
          xx1 = np.maximum(det[0, 0], det[:, 0])
          yy1 = np.maximum(det[0, 1], det[:, 1])
          xx2 = np.minimum(det[0, 2], det[:, 2])
          yy2 = np.minimum(det[0, 3], det[:, 3])
          w = np.maximum(0.0, xx2 - xx1 + 1)
          h = np.maximum(0.0, yy2 - yy1 + 1)
          inter = w * h
          o = inter / (area[0] + area[:] - inter)

          # nms
          merge_index = np.where(o >= self.nms_threshold)[0]
          det_accu = det[merge_index, :]
          det = np.delete(det, merge_index, 0)
          if merge_index.shape[0] <= 1:
              if det.shape[0] == 0:
                  try:
                      dets = np.row_stack((dets, det_accu))
                  except:
                      dets = det_accu
              continue
          det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
          max_score = np.max(det_accu[:, 4])
          det_accu_sum = np.zeros((1, 5))
          det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
                                        axis=0) / np.sum(det_accu[:, -1:])
          det_accu_sum[:, 4] = max_score
          try:
              dets = np.row_stack((dets, det_accu_sum))
          except:
              dets = det_accu_sum
      dets = dets[0:750, :]
      return dets

