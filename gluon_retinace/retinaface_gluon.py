from mxnet.gluon import nn
from mxnet import gluon


class RetinfaceGluon(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(RetinfaceGluon, self).__init__(**kwargs)
        with self.name_scope():
            self.bn_data = nn.BatchNorm(epsilon=2e-05)

            self.conv0 = nn.Conv2D(channels=64, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3), use_bias=False)
            self.bn0 = nn.BatchNorm(epsilon=2e-05)
            self.relu0 = nn.Activation(activation="relu")
            self.pooling0 = nn.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding=(1, 1))

            self.stage1_unit1_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage1_unit1_relu1 = nn.Activation(activation="relu")
            self.stage1_unit1_conv1 = nn.Conv2D(channels=64, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage1_unit1_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage1_unit1_relu2 = nn.Activation(activation="relu")
            self.stage1_unit1_conv2 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage1_unit1_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage1_unit1_relu3 = nn.Activation(activation="relu")
            self.stage1_unit1_conv3 = nn.Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)
            self.stage1_unit1_sc = nn.Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage1_unit2_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage1_unit2_relu1 = nn.Activation(activation="relu")
            self.stage1_unit2_conv1 = nn.Conv2D(channels=64, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage1_unit2_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage1_unit2_relu2 = nn.Activation(activation="relu")
            self.stage1_unit2_conv2 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage1_unit2_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage1_unit2_relu3 = nn.Activation(activation="relu")
            self.stage1_unit2_conv3 = nn.Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage1_unit3_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage1_unit3_relu1 = nn.Activation(activation="relu")
            self.stage1_unit3_conv1 = nn.Conv2D(channels=64, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage1_unit3_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage1_unit3_relu2 = nn.Activation(activation="relu")
            self.stage1_unit3_conv2 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage1_unit3_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage1_unit3_relu3 = nn.Activation(activation="relu")
            self.stage1_unit3_conv3 = nn.Conv2D(channels=256, kernel_size=(1, 1),strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage2_unit1_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage2_unit1_relu1 = nn.Activation(activation="relu")
            self.stage2_unit1_conv1 = nn.Conv2D(channels=128, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage2_unit1_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage2_unit1_relu2 = nn.Activation(activation="relu")
            self.stage2_unit1_conv2 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), use_bias=False)

            self.stage2_unit1_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage2_unit1_relu3 = nn.Activation(activation="relu")
            self.stage2_unit1_conv3 = nn.Conv2D(channels=512, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage2_unit1_sc = nn.Conv2D(channels=512, kernel_size=(1, 1), strides=(2, 2), padding=(0, 0),  use_bias=False)

            self.stage2_unit2_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage2_unit2_relu1 = nn.Activation(activation="relu")
            self.stage2_unit2_conv1 = nn.Conv2D(channels=128, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage2_unit2_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage2_unit2_relu2 = nn.Activation(activation="relu")
            self.stage2_unit2_conv2 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage2_unit2_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage2_unit2_relu3 = nn.Activation(activation="relu")
            self.stage2_unit2_conv3 = nn.Conv2D(channels=512, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage2_unit3_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage2_unit3_relu1 = nn.Activation(activation="relu")
            self.stage2_unit3_conv1 = nn.Conv2D(channels=128, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage2_unit3_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage2_unit3_relu2 = nn.Activation(activation="relu")
            self.stage2_unit3_conv2 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage2_unit3_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage2_unit3_relu3 = nn.Activation(activation="relu")
            self.stage2_unit3_conv3 = nn.Conv2D(channels=512, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage2_unit4_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage2_unit4_relu1 = nn.Activation(activation="relu")
            self.stage2_unit4_conv1 = nn.Conv2D(channels=128, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage2_unit4_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage2_unit4_relu2 = nn.Activation(activation="relu")
            self.stage2_unit4_conv2 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage2_unit4_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage2_unit4_relu3 = nn.Activation(activation="relu")
            self.stage2_unit4_conv3 = nn.Conv2D(channels=512, kernel_size=(1, 1),strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage3_unit1_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit1_relu1 = nn.Activation(activation="relu")
            self.stage3_unit1_conv1 = nn.Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage3_unit1_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit1_relu2 = nn.Activation(activation="relu")
            self.stage3_unit1_conv2 = nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), use_bias=False)

            self.stage3_unit1_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit1_relu3 = nn.Activation(activation="relu")
            self.stage3_unit1_conv3 = nn.Conv2D(channels=1024, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)
            self.stage3_unit1_sc = nn.Conv2D(channels=1024, kernel_size=(1, 1), strides=(2, 2), padding=(0, 0), use_bias=False)

            self.stage3_unit2_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit2_relu1 = nn.Activation(activation="relu")
            self.stage3_unit2_conv1 = nn.Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage3_unit2_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit2_relu2 = nn.Activation(activation="relu")
            self.stage3_unit2_conv2 = nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage3_unit2_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit2_relu3 = nn.Activation(activation="relu")
            self.stage3_unit2_conv3 = nn.Conv2D(channels=1024, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage3_unit3_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit3_relu1 = nn.Activation(activation="relu")
            self.stage3_unit3_conv1 = nn.Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage3_unit3_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit3_relu2 = nn.Activation(activation="relu")
            self.stage3_unit3_conv2 = nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage3_unit3_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit3_relu3 = nn.Activation(activation="relu")
            self.stage3_unit3_conv3 = nn.Conv2D(channels=1024, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage3_unit4_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit4_relu1 = nn.Activation(activation="relu")
            self.stage3_unit4_conv1 = nn.Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage3_unit4_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit4_relu2 = nn.Activation(activation="relu")
            self.stage3_unit4_conv2 = nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage3_unit4_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit4_relu3 = nn.Activation(activation="relu")
            self.stage3_unit4_conv3 = nn.Conv2D(channels=1024, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage3_unit5_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit5_relu1 = nn.Activation(activation="relu")
            self.stage3_unit5_conv1 = nn.Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage3_unit5_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit5_relu2 = nn.Activation(activation="relu")
            self.stage3_unit5_conv2 = nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage3_unit5_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit5_relu3 = nn.Activation(activation="relu")
            self.stage3_unit5_conv3 = nn.Conv2D(channels=1024, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage3_unit6_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit6_relu1 = nn.Activation(activation="relu")
            self.stage3_unit6_conv1 = nn.Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage3_unit6_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit6_relu2 = nn.Activation(activation="relu")
            self.stage3_unit6_conv2 = nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage3_unit6_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage3_unit6_relu3 = nn.Activation(activation="relu")
            self.stage3_unit6_conv3 = nn.Conv2D(channels=1024, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage4_unit1_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage4_unit1_relu1 = nn.Activation(activation="relu")
            self.stage4_unit1_conv1 = nn.Conv2D(channels=512, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0),use_bias=False)

            self.stage4_unit1_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage4_unit1_relu2 = nn.Activation(activation="relu")
            self.stage4_unit1_conv2 = nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), use_bias=False)

            self.stage4_unit1_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage4_unit1_relu3 = nn.Activation(activation="relu")
            self.stage4_unit1_conv3 = nn.Conv2D(channels=2048, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)
            self.stage4_unit1_sc = nn.Conv2D(channels=2048, kernel_size=(1, 1), strides=(2, 2), padding=(0, 0), use_bias=False)

            self.stage4_unit2_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage4_unit2_relu1 = nn.Activation(activation="relu")
            self.stage4_unit2_conv1 = nn.Conv2D(channels=512, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage4_unit2_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage4_unit2_relu2 = nn.Activation(activation="relu")
            self.stage4_unit2_conv2 = nn.Conv2D(channels=512, kernel_size=(3, 3),
                                                    strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage4_unit2_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage4_unit2_relu3 = nn.Activation(activation="relu")
            self.stage4_unit2_conv3 = nn.Conv2D(channels=2048, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage4_unit3_bn1 = nn.BatchNorm(epsilon=2e-05)
            self.stage4_unit3_relu1 = nn.Activation(activation="relu")
            self.stage4_unit3_conv1 = nn.Conv2D(channels=512, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.stage4_unit3_bn2 = nn.BatchNorm(epsilon=2e-05)
            self.stage4_unit3_relu2 = nn.Activation(activation="relu")
            self.stage4_unit3_conv2 = nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), use_bias=False)

            self.stage4_unit3_bn3 = nn.BatchNorm(epsilon=2e-05)
            self.stage4_unit3_relu3 = nn.Activation(activation="relu")
            self.stage4_unit3_conv3 = nn.Conv2D(channels=2048, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)

            self.bn1 = nn.BatchNorm(epsilon=2e-05)
            self.relu1 = nn.Activation(activation="relu")

            self.ssh_c3_lateral = nn.Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
            self.ssh_c3_lateral_bn = nn.BatchNorm(epsilon=2e-05)
            self.ssh_c3_lateral_relu = nn.Activation(activation="relu")

            self.ssh_m3_det_conv1 = nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_m3_det_conv1_bn = nn.BatchNorm(epsilon=2e-05)

            self.ssh_m3_det_context_conv1 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_m3_det_context_conv1_bn = nn.BatchNorm(epsilon=2e-05)
            self.ssh_m3_det_context_conv1_relu = nn.Activation(activation="relu")

            self.ssh_m3_det_context_conv2 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1),  padding=(1, 1))
            self.ssh_m3_det_context_conv2_bn = nn.BatchNorm(epsilon=2e-05)

            self.ssh_m3_det_context_conv3_1 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1),padding=(1, 1))
            self.ssh_m3_det_context_conv3_1_bn = nn.BatchNorm(epsilon=2e-05)
            self.ssh_m3_det_context_conv3_1_relu = nn.Activation(activation="relu")

            self.ssh_m3_det_context_conv3_2 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_m3_det_context_conv3_2_bn = nn.BatchNorm(epsilon=2e-05)

            self.ssh_m3_det_concat_relu = nn.Activation(activation="relu")

            self.face_rpn_cls_score_stride32 = nn.Conv2D(channels=4, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
            self.face_rpn_bbox_pred_stride32 = nn.Conv2D(channels=8, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
            self.face_rpn_landmark_pred_stride32 = nn.Conv2D(channels=20, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))

            self.ssh_c2_lateral = nn.Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
            self.ssh_c2_lateral_bn = nn.BatchNorm(epsilon=2e-05)
            self.ssh_c2_lateral_relu = nn.Activation(activation="relu")

            self.ssh_c2_aggr = nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_c2_aggr_bn = nn.BatchNorm(epsilon=2e-05)
            self.ssh_c2_aggr_relu = nn.Activation(activation="relu")

            self.ssh_m2_det_conv1 = nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_m2_det_conv1_bn = nn.BatchNorm(epsilon=2e-05)

            self.ssh_m2_det_context_conv1 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_m2_det_context_conv1_bn = nn.BatchNorm(epsilon=2e-05)
            self.ssh_m2_det_context_conv1_relu = nn.Activation(activation="relu")

            self.ssh_m2_det_context_conv2 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_m2_det_context_conv2_bn = nn.BatchNorm(epsilon=2e-05)

            self.ssh_m2_det_context_conv3_1 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_m2_det_context_conv3_1_bn = nn.BatchNorm(epsilon=2e-05)
            self.ssh_m2_det_context_conv3_1_relu = nn.Activation(activation="relu")

            self.ssh_m2_det_context_conv3_2 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_m2_det_context_conv3_2_bn = nn.BatchNorm(epsilon=2e-05)

            self.ssh_m2_det_concat_relu = nn.Activation(activation="relu")

            self.face_rpn_cls_score_stride16 = nn.Conv2D(channels=4, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
            self.face_rpn_bbox_pred_stride16 = nn.Conv2D(channels=8, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
            self.face_rpn_landmark_pred_stride16 = nn.Conv2D(channels=20, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))

            self.ssh_m1_red_conv = nn.Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
            self.ssh_m1_red_conv_bn = nn.BatchNorm(epsilon=2e-05)
            self.ssh_m1_red_conv_relu = nn.Activation(activation="relu")

            self.ssh_c1_aggr = nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_c1_aggr_bn = nn.BatchNorm(epsilon=2e-05)
            self.ssh_c1_aggr_relu = nn.Activation(activation="relu")

            self.ssh_m1_det_conv1 = nn.Conv2D(channels=256, kernel_size=(3, 3),  strides=(1, 1), padding=(1, 1))
            self.ssh_m1_det_conv1_bn =nn.BatchNorm(epsilon=2e-05)

            self.ssh_m1_det_context_conv1 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_m1_det_context_conv1_bn = nn.BatchNorm(epsilon=2e-05)
            self.ssh_m1_det_context_conv1_relu = nn.Activation(activation="relu")

            self.ssh_m1_det_context_conv2 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_m1_det_context_conv2_bn =nn.BatchNorm(epsilon=2e-05)

            self.ssh_m1_det_context_conv3_1 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_m1_det_context_conv3_1_bn = nn.BatchNorm(epsilon=2e-05)
            self.ssh_m1_det_context_conv3_1_relu = nn.Activation(activation="relu")

            self.ssh_m1_det_context_conv3_2 = nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            self.ssh_m1_det_context_conv3_2_bn = nn.BatchNorm(epsilon=2e-05)

            self.ssh_m1_det_concat_relu = nn.Activation(activation="relu")

            self.face_rpn_cls_score_stride8 = nn.Conv2D(channels=4, kernel_size=(1, 1),strides=(1, 1), padding=(0, 0))
            self.face_rpn_bbox_pred_stride8 = nn.Conv2D(channels=8, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
            self.face_rpn_landmark_pred_stride8 = nn.Conv2D(channels=20, kernel_size=(1, 1),  strides=(1, 1), padding=(0, 0))

    def hybrid_forward(self, F, x):
        x = self.bn_data(x)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pooling0(x)

        x = self.stage1_unit1_bn1(x)
        stage1_unit1_relu1 = self.stage1_unit1_relu1(x)
        x = self.stage1_unit1_conv1(stage1_unit1_relu1)

        x = self.stage1_unit1_bn2(x)
        x = self.stage1_unit1_relu2(x)
        x = self.stage1_unit1_conv2(x)

        x = self.stage1_unit1_bn3(x)
        x = self.stage1_unit1_relu3(x)
        x = self.stage1_unit1_conv3(x)

        stage1_unit1_sc = self.stage1_unit1_sc(stage1_unit1_relu1)
        _plus0 = F.broadcast_add(lhs=x, rhs=stage1_unit1_sc)

        x = self.stage1_unit2_bn1(_plus0)
        x = self.stage1_unit2_relu1(x)
        x = self.stage1_unit2_conv1(x)

        x = self.stage1_unit2_bn2(x)
        x = self.stage1_unit2_relu2(x)
        x = self.stage1_unit2_conv2(x)

        x = self.stage1_unit2_bn3(x)
        x = self.stage1_unit2_relu3(x)
        x = self.stage1_unit2_conv3(x)
        _plus1 = F.broadcast_add(lhs=x, rhs=stage1_unit1_sc)

        x = self.stage1_unit3_bn1(_plus1)
        x = self.stage1_unit3_relu1(x)
        x = self.stage1_unit3_conv1(x)

        x = self.stage1_unit3_bn2(x)
        x = self.stage1_unit3_relu2(x)
        x = self.stage1_unit3_conv2(x)

        x = self.stage1_unit3_bn3(x)
        x = self.stage1_unit3_relu3(x)
        x = self.stage1_unit3_conv3(x)
        _plus2 = F.broadcast_add(lhs=x, rhs=_plus1)

        x = self.stage2_unit1_bn1(_plus2)
        stage2_unit1_relu1 = self.stage2_unit1_relu1(x)
        x = self.stage2_unit1_conv1(stage2_unit1_relu1)

        x = self.stage2_unit1_bn2(x)
        x = self.stage2_unit1_relu2(x)
        x = self.stage2_unit1_conv2(x)

        x = self.stage2_unit1_bn3(x)
        x = self.stage2_unit1_relu3(x)
        x = self.stage2_unit1_conv3(x)

        stage2_unit1_sc = self.stage2_unit1_sc(stage2_unit1_relu1)
        _plus3 = F.broadcast_add(lhs=x, rhs=stage2_unit1_sc)

        x = self.stage2_unit2_bn1(_plus3)
        x = self.stage2_unit2_relu1(x)
        x = self.stage2_unit2_conv1(x)

        x = self.stage2_unit2_bn2(x)
        x = self.stage2_unit2_relu2(x)
        x = self.stage2_unit2_conv2(x)

        x = self.stage2_unit2_bn3(x)
        x = self.stage2_unit2_relu3(x)
        x = self.stage2_unit2_conv3(x)

        _plus4 = F.broadcast_add(lhs=x, rhs=_plus3)

        x = self.stage2_unit3_bn1(_plus4)
        x = self.stage2_unit3_relu1(x)
        x = self.stage2_unit3_conv1(x)

        x = self.stage2_unit3_bn2(x)
        x = self.stage2_unit3_relu2(x)
        x = self.stage2_unit3_conv2(x)

        x = self.stage2_unit3_bn3(x)
        x = self.stage2_unit3_relu3(x)
        x = self.stage2_unit3_conv3(x)
        _plus5 = F.broadcast_add(lhs=x, rhs=_plus4)

        x = self.stage2_unit4_bn1(_plus5)
        x = self.stage2_unit4_relu1(x)
        x = self.stage2_unit4_conv1(x)

        x = self.stage2_unit4_bn2(x)
        x = self.stage2_unit4_relu2(x)
        x = self.stage2_unit4_conv2(x)

        x = self.stage2_unit4_bn3(x)
        x = self.stage2_unit4_relu3(x)
        x = self.stage2_unit4_conv3(x)
        _plus6 = F.broadcast_add(lhs=x, rhs=_plus5)

        x = self.stage3_unit1_bn1(_plus6)
        stage3_unit1_relu1 = self.stage3_unit1_relu1(x)
        x = self.stage3_unit1_conv1(stage3_unit1_relu1)

        x = self.stage3_unit1_bn2(x)
        stage3_unit1_relu2 = self.stage3_unit1_relu2(x)
        x = self.stage3_unit1_conv2(stage3_unit1_relu2)

        x = self.stage3_unit1_bn3(x)
        x = self.stage3_unit1_relu3(x)
        x = self.stage3_unit1_conv3(x)
        stage3_unit1_sc = self.stage3_unit1_sc(stage3_unit1_relu1)
        _plus7 = F.broadcast_add(lhs=x, rhs=stage3_unit1_sc)

        x = self.stage3_unit2_bn1(_plus7)
        x = self.stage3_unit2_relu1(x)
        x = self.stage3_unit2_conv1(x)

        x = self.stage3_unit2_bn2(x)
        x = self.stage3_unit1_relu2(x)
        x = self.stage3_unit2_conv2(x)

        x = self.stage3_unit2_bn3(x)
        x = self.stage3_unit2_relu3(x)
        x = self.stage3_unit2_conv3(x)
        _plus8 = F.broadcast_add(lhs=x, rhs=_plus7)

        x = self.stage3_unit3_bn1(_plus8)
        x = self.stage3_unit3_relu1(x)
        x = self.stage3_unit3_conv1(x)

        x = self.stage3_unit3_bn2(x)
        x = self.stage3_unit3_relu2(x)
        x = self.stage3_unit3_conv2(x)

        x = self.stage3_unit3_bn3(x)
        x = self.stage3_unit3_relu3(x)
        x = self.stage3_unit3_conv3(x)
        _plus9 = F.broadcast_add(lhs=x, rhs=_plus8)

        x = self.stage3_unit4_bn1(_plus9)
        x = self.stage3_unit4_relu1(x)
        x = self.stage3_unit4_conv1(x)

        x = self.stage3_unit4_bn2(x)
        x = self.stage3_unit4_relu2(x)
        x = self.stage3_unit4_conv2(x)

        x = self.stage3_unit4_bn3(x)
        x = self.stage3_unit4_relu3(x)
        x = self.stage3_unit4_conv3(x)
        _plus10 = F.broadcast_add(lhs=x, rhs=_plus9)

        x = self.stage3_unit5_bn1(_plus10)
        x = self.stage3_unit5_relu1(x)
        x = self.stage3_unit5_conv1(x)

        x = self.stage3_unit5_bn2(x)
        x = self.stage3_unit5_relu2(x)
        x = self.stage3_unit5_conv2(x)

        x = self.stage3_unit5_bn3(x)
        x = self.stage3_unit5_relu3(x)
        x = self.stage3_unit5_conv3(x)
        _plus11 = F.broadcast_add(lhs=x, rhs=_plus10)

        x = self.stage3_unit6_bn1(_plus11)
        x = self.stage3_unit6_relu1(x)
        x = self.stage3_unit6_conv1(x)

        x = self.stage3_unit6_bn2(x)
        x = self.stage3_unit6_relu2(x)
        x = self.stage3_unit6_conv2(x)

        x = self.stage3_unit6_bn3(x)
        x = self.stage3_unit6_relu3(x)
        x = self.stage3_unit6_conv3(x)
        _plus12 = F.broadcast_add(lhs=x, rhs=_plus11)

        x = self.stage4_unit1_bn1(_plus12)
        stage4_unit1_relu1 = self.stage4_unit1_relu1(x)
        x = self.stage4_unit1_conv1(stage4_unit1_relu1)

        x = self.stage4_unit1_bn2(x)
        stage4_unit1_relu2 = self.stage4_unit1_relu2(x)
        x = self.stage4_unit1_conv2(stage4_unit1_relu2)

        x = self.stage4_unit1_bn3(x)
        x = self.stage4_unit1_relu3(x)
        x = self.stage4_unit1_conv3(x)
        stage4_unit1_sc = self.stage4_unit1_sc(stage4_unit1_relu1)
        _plus13 = F.broadcast_add(lhs=x, rhs=stage4_unit1_sc)

        x = self.stage4_unit2_bn1(_plus13)
        x = self.stage4_unit2_relu1(x)
        x = self.stage4_unit2_conv1(x)

        x = self.stage4_unit2_bn2(x)
        x = self.stage4_unit2_relu2(x)
        x = self.stage4_unit2_conv2(x)

        x = self.stage4_unit2_bn3(x)
        x = self.stage4_unit2_relu3(x)
        x = self.stage4_unit2_conv3(x)
        _plus14 = F.broadcast_add(lhs=x, rhs=_plus13)

        x = self.stage4_unit3_bn1(_plus14)
        x = self.stage4_unit3_relu1(x)
        x = self.stage4_unit3_conv1(x)

        x = self.stage4_unit3_bn2(x)
        x = self.stage4_unit3_relu2(x)
        x = self.stage4_unit3_conv2(x)

        x = self.stage4_unit3_bn3(x)
        x = self.stage4_unit3_relu3(x)
        x = self.stage4_unit3_conv3(x)
        _plus15 = F.broadcast_add(lhs=x, rhs=_plus14)

        x = self.bn1(_plus15)
        x = self.relu1(x)

        x = self.ssh_c3_lateral(x)
        x = self.ssh_c3_lateral_bn(x)
        ssh_c3_lateral_relu = self.ssh_c3_lateral_relu(x)

        x = self.ssh_m3_det_conv1(ssh_c3_lateral_relu)
        ssh_m3_det_conv1_bn = self.ssh_m3_det_conv1_bn(x)

        x = self.ssh_m3_det_context_conv1(ssh_c3_lateral_relu)
        x = self.ssh_m3_det_context_conv1_bn(x)
        ssh_m3_det_context_conv1_relu = self.ssh_m3_det_context_conv1_relu(x)

        x = self.ssh_m3_det_context_conv2(ssh_m3_det_context_conv1_relu)
        ssh_m3_det_context_conv2_bn = self.ssh_m3_det_context_conv2_bn(x)

        x = self.ssh_m3_det_context_conv3_1(ssh_m3_det_context_conv1_relu)
        x = self.ssh_m3_det_context_conv3_1_bn(x)
        x = self.ssh_m3_det_context_conv3_1_relu(x)

        x = self.ssh_m3_det_context_conv3_2(x)
        x = self.ssh_m3_det_context_conv3_2_bn(x)

        x = F.concat(ssh_m3_det_conv1_bn, ssh_m3_det_context_conv2_bn, x, dim=1)
        ssh_m3_det_concat_relu = self.ssh_m3_det_concat_relu(x)

        x = self.face_rpn_cls_score_stride32(ssh_m3_det_concat_relu)
        x = F.reshape(x, shape=(0, 2, -1, 0))
        x = F.SoftmaxActivation(x, mode="channel")
        face_rpn_cls_prob_reshape_stride32 = F.reshape(x, shape=(0, 4, -1, 0))
        face_rpn_bbox_pred_stride32 = self.face_rpn_bbox_pred_stride32(ssh_m3_det_concat_relu)
        face_rpn_landmark_pred_stride32 = self.face_rpn_landmark_pred_stride32(ssh_m3_det_concat_relu)

        x = self.ssh_c2_lateral(stage4_unit1_relu2)
        x = self.ssh_c2_lateral_bn(x)
        ssh_c2_lateral_relu = self.ssh_c2_lateral_relu(x)
        x = F.UpSampling(ssh_c3_lateral_relu, sample_type="nearest", scale=2)
        x = F.Crop(*[x, ssh_c2_lateral_relu])
        _plus16 = F.broadcast_add(lhs=ssh_c2_lateral_relu, rhs=x)

        x = self.ssh_c2_aggr(_plus16)
        x = self.ssh_c2_aggr_bn(x)
        ssh_c2_aggr_relu = self.ssh_c2_aggr_relu(x)

        x = self.ssh_m2_det_conv1(ssh_c2_aggr_relu)
        ssh_m2_det_conv1_bn = self.ssh_m2_det_conv1_bn(x)

        x = self.ssh_m2_det_context_conv1(ssh_c2_aggr_relu)
        x = self.ssh_m2_det_context_conv1_bn(x)
        ssh_m2_det_context_conv1_relu = self.ssh_m2_det_context_conv1_relu(x)

        x = self.ssh_m2_det_context_conv2(ssh_m2_det_context_conv1_relu)
        ssh_m2_det_context_conv2_bn = self.ssh_m2_det_context_conv2_bn(x)

        x = self.ssh_m2_det_context_conv3_1(ssh_m2_det_context_conv1_relu)
        x = self.ssh_m2_det_context_conv3_1_bn(x)
        x = self.ssh_m2_det_context_conv3_1_relu(x)

        x = self.ssh_m2_det_context_conv3_2(x)
        x = self.ssh_m2_det_context_conv3_2_bn(x)

        x = F.concat(ssh_m2_det_conv1_bn, ssh_m2_det_context_conv2_bn, x, dim=1)
        ssh_m2_det_concat_relu = self.ssh_m2_det_concat_relu(x)

        x = self.face_rpn_cls_score_stride16(ssh_m2_det_concat_relu)
        x = F.reshape(x, shape=(0, 2, -1, 0))
        x = F.SoftmaxActivation(x, mode="channel")
        face_rpn_cls_prob_reshape_stride16 = F.reshape(x, shape=(0, 4, -1, 0))
        face_rpn_bbox_pred_stride16 = self.face_rpn_bbox_pred_stride16(ssh_m2_det_concat_relu)
        face_rpn_landmark_pred_stride16 = self.face_rpn_landmark_pred_stride16(ssh_m2_det_concat_relu)

        x = self.ssh_m1_red_conv(stage3_unit1_relu2)
        x = self.ssh_m1_red_conv_bn(x)
        ssh_m1_red_conv_relu = self.ssh_m1_red_conv_relu(x)

        x = F.UpSampling(ssh_c2_aggr_relu, sample_type="nearest", scale=2)
        x = F.Crop(*[x, ssh_m1_red_conv_relu])
        _plus17 = F.broadcast_add(lhs=ssh_m1_red_conv_relu, rhs=x)

        x = self.ssh_c1_aggr(_plus17)
        x = self.ssh_c1_aggr_bn(x)
        ssh_c1_aggr_relu = self.ssh_c1_aggr_relu(x)

        x = self.ssh_m1_det_conv1(ssh_c1_aggr_relu)
        ssh_m1_det_conv1_bn = self.ssh_m1_det_conv1_bn(x)

        x = self.ssh_m1_det_context_conv1(ssh_c1_aggr_relu)
        x = self.ssh_m1_det_context_conv1_bn(x)
        ssh_m1_det_context_conv1_relu = self.ssh_m1_det_context_conv1_relu(x)

        x = self.ssh_m1_det_context_conv2(ssh_m1_det_context_conv1_relu)
        ssh_m1_det_context_conv2_bn = self.ssh_m1_det_context_conv2_bn(x)

        x = self.ssh_m1_det_context_conv3_1(ssh_m1_det_context_conv1_relu)
        x = self.ssh_m1_det_context_conv3_1_bn(x)
        x = self.ssh_m1_det_context_conv3_1_relu(x)

        x = self.ssh_m1_det_context_conv3_2(x)
        x = self.ssh_m1_det_context_conv3_2_bn(x)

        x = F.concat(ssh_m1_det_conv1_bn, ssh_m1_det_context_conv2_bn, x, dim=1)
        ssh_m1_det_concat_relu = self.ssh_m1_det_concat_relu(x)

        x = self.face_rpn_cls_score_stride8(ssh_m1_det_concat_relu)
        x = F.reshape(x, shape=(0, 2, -1, 0))
        x = F.SoftmaxActivation(x, mode="channel")
        face_rpn_cls_prob_reshape_stride8 = F.reshape(x, shape=(0, 4, -1, 0))

        face_rpn_bbox_pred_stride8 = self.face_rpn_bbox_pred_stride8(ssh_m1_det_concat_relu)
        face_rpn_landmark_pred_stride8 = self.face_rpn_landmark_pred_stride8(ssh_m1_det_concat_relu)

        ret_group = []
        ret_group.append(face_rpn_cls_prob_reshape_stride32)
        ret_group.append(face_rpn_bbox_pred_stride32)
        ret_group.append(face_rpn_landmark_pred_stride32)

        ret_group.append(face_rpn_cls_prob_reshape_stride16)
        ret_group.append(face_rpn_bbox_pred_stride16)
        ret_group.append(face_rpn_landmark_pred_stride16)

        ret_group.append(face_rpn_cls_prob_reshape_stride8)
        ret_group.append(face_rpn_bbox_pred_stride8)
        ret_group.append(face_rpn_landmark_pred_stride8)
        # print('face_rpn_cls_prob_reshape_stride32', face_rpn_cls_prob_reshape_stride32.shape)
        # print('face_rpn_bbox_pred_stride32', face_rpn_bbox_pred_stride32.shape)
        # print('face_rpn_landmark_pred_stride32', face_rpn_landmark_pred_stride32.shape)
        #
        # print('face_rpn_cls_prob_reshape_stride16', face_rpn_cls_prob_reshape_stride16.shape)
        # print('face_rpn_bbox_pred_stride16', face_rpn_bbox_pred_stride16.shape)
        # print('face_rpn_landmark_pred_stride16', face_rpn_landmark_pred_stride16.shape)
        #
        # print('face_rpn_cls_prob_reshape_stride8', face_rpn_cls_prob_reshape_stride8.shape)
        # print('face_rpn_bbox_pred_stride8', face_rpn_bbox_pred_stride8.shape)
        # print('face_rpn_landmark_pred_stride8', face_rpn_landmark_pred_stride8.shape)
        return ret_group



if __name__ == '__main__':
    gluon_net = RetinfaceGluon()
    gluon_net.load_parameters('gluon_net.params')



















































