import mxnet as mx
import numpy as np



def get_r50_symbol(in_data):
    bn_data = mx.sym.BatchNorm(data=in_data, fix_gamma=True, use_global_stats=False, eps=2e-05, momentum=0.9, name='bn_data')

    conv0 = mx.sym.Convolution(data=bn_data, num_filter=64, kernel=(7,7), stride=(2,2), pad=(3,3),dilate=(1,1),
                               num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='conv0')
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, use_global_stats=False, eps=2e-05, momentum=0.9,name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type="relu", name='relu0')
    pooling0 = mx.sym.Pooling(data=relu0, kernel=(3,3), stride=(2,2), pad=(1,1), global_pool=False, pool_type='max', name='pooling0')

    stage1_unit1_bn1 = mx.sym.BatchNorm(data=pooling0, fix_gamma=False, use_global_stats=False, eps=2e-05, momentum=0.9, name='stage1_unit1_bn1')
    stage1_unit1_relu1 = mx.sym.Activation(data=stage1_unit1_bn1, act_type="relu", name='stage1_unit1_relu1')
    stage1_unit1_conv1 = mx.sym.Convolution(data=stage1_unit1_relu1, num_filter=64, kernel=(1,1), stride=(1,1), pad=(0,0), dilate=(1,1),
                               num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage1_unit1_conv1')

    stage1_unit1_bn2 = mx.sym.BatchNorm(data=stage1_unit1_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05, momentum=0.9, name='stage1_unit1_bn2')
    stage1_unit1_relu2 = mx.sym.Activation(data=stage1_unit1_bn2, act_type="relu", name='stage1_unit1_relu2')
    stage1_unit1_conv2 = mx.sym.Convolution(data=stage1_unit1_relu2, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), dilate=(1,1),
                               num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage1_unit1_conv2')

    stage1_unit1_bn3 = mx.sym.BatchNorm(data=stage1_unit1_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05, momentum=0.9,name='stage1_unit1_bn3')
    stage1_unit1_relu3 = mx.sym.Activation(data=stage1_unit1_bn3, act_type="relu", name='stage1_unit1_relu3')
    stage1_unit1_conv3 = mx.sym.Convolution(data=stage1_unit1_relu3, num_filter=256, kernel=(1,1), stride=(1, 1),pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256,name='stage1_unit1_conv3')
    stage1_unit1_sc = mx.sym.Convolution(data=stage1_unit1_relu1, num_filter=256, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage1_unit1_sc')

    _plus0 = mx.sym.elemwise_add(lhs=stage1_unit1_conv3, rhs=stage1_unit1_sc, name='_plus0')

    stage1_unit2_bn1 = mx.sym.BatchNorm(data=_plus0, fix_gamma=False, use_global_stats=False, eps=2e-05, momentum=0.9, name='stage1_unit2_bn1')
    stage1_unit2_relu1 = mx.sym.Activation(data=stage1_unit2_bn1, act_type="relu", name='stage1_unit2_relu1')
    stage1_unit2_conv1 = mx.sym.Convolution(data=stage1_unit2_relu1, num_filter=64, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage1_unit2_conv1')

    stage1_unit2_bn2 = mx.sym.BatchNorm(data=stage1_unit2_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05, momentum=0.9,
                                        name='stage1_unit2_bn2')
    stage1_unit2_relu2 = mx.sym.Activation(data=stage1_unit2_bn2, act_type="relu", name='stage1_unit2_relu2')
    stage1_unit2_conv2 = mx.sym.Convolution(data=stage1_unit2_relu2, num_filter=64, kernel=(3,3), stride=(1, 1), pad=(1, 1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256,name='stage1_unit2_conv2')

    stage1_unit2_bn3 = mx.sym.BatchNorm(data=stage1_unit2_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05, momentum=0.9,
                                        name='stage1_unit2_bn3')
    stage1_unit2_relu3 = mx.sym.Activation(data=stage1_unit2_bn3, act_type="relu", name='stage1_unit2_relu3')
    stage1_unit2_conv3 = mx.sym.Convolution(data=stage1_unit2_relu3, num_filter=256, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage1_unit2_conv3')

    _plus1 = mx.sym.elemwise_add(lhs=stage1_unit2_conv3, rhs=stage1_unit1_sc, name='_plus1')

    stage1_unit3_bn1 = mx.sym.BatchNorm(data=_plus1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage1_unit3_bn1')
    stage1_unit3_relu1 = mx.sym.Activation(data=stage1_unit3_bn1, act_type="relu", name='stage1_unit3_relu1')
    stage1_unit3_conv1 = mx.sym.Convolution(data=stage1_unit3_relu1, num_filter=64, kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256,name='stage1_unit3_conv1')

    stage1_unit3_bn2 = mx.sym.BatchNorm(data=stage1_unit3_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage1_unit3_bn2')
    stage1_unit3_relu2 = mx.sym.Activation(data=stage1_unit3_bn2, act_type="relu", name='stage1_unit3_relu2')
    stage1_unit3_conv2 = mx.sym.Convolution(data=stage1_unit3_relu2, num_filter=64, kernel=(3,3), stride=(1, 1), pad=(1,1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage1_unit3_conv2')

    stage1_unit3_bn3 = mx.sym.BatchNorm(data=stage1_unit3_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage1_unit3_bn3')
    stage1_unit3_relu3 = mx.sym.Activation(data=stage1_unit3_bn3, act_type="relu", name='stage1_unit3_relu3')
    stage1_unit3_conv3 = mx.sym.Convolution(data=stage1_unit3_relu3, num_filter=256, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage1_unit3_conv3')
    _plus2 = mx.sym.elemwise_add(lhs=stage1_unit3_conv3, rhs=_plus1, name='_plus2')

    stage2_unit1_bn1 = mx.sym.BatchNorm(data=_plus2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage2_unit1_bn1')
    stage2_unit1_relu1 = mx.sym.Activation(data=stage2_unit1_bn1, act_type="relu", name='stage2_unit1_relu1')
    stage2_unit1_conv1 = mx.sym.Convolution(data=stage2_unit1_relu1, num_filter=128, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit1_conv1')

    stage2_unit1_bn2 = mx.sym.BatchNorm(data=stage2_unit1_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage2_unit1_bn2')
    stage2_unit1_relu2 = mx.sym.Activation(data=stage2_unit1_bn2, act_type="relu", name='stage2_unit1_relu2')
    stage2_unit1_conv2 = mx.sym.Convolution(data=stage2_unit1_relu2, num_filter=128, kernel=(3,3), stride=(2,2), pad=(1,1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit1_conv2')

    stage2_unit1_bn3 = mx.sym.BatchNorm(data=stage2_unit1_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage2_unit1_bn3')
    stage2_unit1_relu3 = mx.sym.Activation(data=stage2_unit1_bn3, act_type="relu", name='stage2_unit1_relu3')
    stage2_unit1_conv3 = mx.sym.Convolution(data=stage2_unit1_relu3, num_filter=512, kernel=(1,1), stride=(1,1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit1_conv3')

    stage2_unit1_sc = mx.sym.Convolution(data=stage2_unit1_relu1, num_filter=512, kernel=(1, 1), stride=(2,2), pad=(0, 0), dilate=(1, 1),
                                         num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit1_sc')
    _plus3 = mx.sym.elemwise_add(lhs=stage2_unit1_conv3, rhs=stage2_unit1_sc, name='_plus3')

    stage2_unit2_bn1 = mx.sym.BatchNorm(data=_plus3, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage2_unit2_bn1')
    stage2_unit2_relu1 = mx.sym.Activation(data=stage2_unit2_bn1, act_type="relu", name='stage2_unit2_relu1')
    stage2_unit2_conv1 = mx.sym.Convolution(data=stage2_unit2_relu1, num_filter=128, kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit2_conv1')

    stage2_unit2_bn2 = mx.sym.BatchNorm(data=stage2_unit2_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage2_unit2_bn2')
    stage2_unit2_relu2 = mx.sym.Activation(data=stage2_unit2_bn2, act_type="relu", name='stage2_unit2_relu2')
    stage2_unit2_conv2 = mx.sym.Convolution(data=stage2_unit2_relu2, num_filter=128, kernel=(3,3), stride=(1, 1), pad=(1,1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit2_conv2')

    stage2_unit2_bn3 = mx.sym.BatchNorm(data=stage2_unit2_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage2_unit2_bn3')
    stage2_unit2_relu3 = mx.sym.Activation(data=stage2_unit2_bn3, act_type="relu", name='stage2_unit2_relu3')
    stage2_unit2_conv3 = mx.sym.Convolution(data=stage2_unit2_relu3, num_filter=512, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit2_conv3')

    _plus4 = mx.sym.elemwise_add(lhs=stage2_unit2_conv3, rhs=_plus3, name='_plus4')

    stage2_unit3_bn1 = mx.sym.BatchNorm(data=_plus4, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage2_unit3_bn1')
    stage2_unit3_relu1 = mx.sym.Activation(data=stage2_unit3_bn1, act_type="relu", name='stage2_unit3_relu1')
    stage2_unit3_conv1 = mx.sym.Convolution(data=stage2_unit3_relu1, num_filter=128, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit3_conv1')

    stage2_unit3_bn2 = mx.sym.BatchNorm(data=stage2_unit3_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage2_unit3_bn2')
    stage2_unit3_relu2 = mx.sym.Activation(data=stage2_unit3_bn2, act_type="relu", name='stage2_unit3_relu2')
    stage2_unit3_conv2 = mx.sym.Convolution(data=stage2_unit3_relu2, num_filter=128, kernel=(3,3), stride=(1, 1), pad=(1,1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit3_conv2')

    stage2_unit3_bn3 = mx.sym.BatchNorm(data=stage2_unit3_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage2_unit3_bn3')
    stage2_unit3_relu3 = mx.sym.Activation(data=stage2_unit3_bn3, act_type="relu", name='stage2_unit3_relu3')
    stage2_unit3_conv3 = mx.sym.Convolution(data=stage2_unit3_relu3, num_filter=512, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit3_conv3')

    _plus5 = mx.sym.elemwise_add(lhs=stage2_unit3_conv3, rhs=_plus4, name='_plus5')

    stage2_unit4_bn1 = mx.sym.BatchNorm(data=_plus5, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage2_unit4_bn1')
    stage2_unit4_relu1 = mx.sym.Activation(data=stage2_unit4_bn1, act_type="relu", name='stage2_unit4_relu1')
    stage2_unit4_conv1 = mx.sym.Convolution(data=stage2_unit4_relu1, num_filter=128, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit4_conv1')

    stage2_unit4_bn2 = mx.sym.BatchNorm(data=stage2_unit4_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage2_unit4_bn2')
    stage2_unit4_relu2 = mx.sym.Activation(data=stage2_unit4_bn2, act_type="relu", name='stage2_unit4_relu2')
    stage2_unit4_conv2 = mx.sym.Convolution(data=stage2_unit4_relu2, num_filter=128, kernel=(3,3), stride=(1, 1), pad=(1, 1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit4_conv2')

    stage2_unit4_bn3 = mx.sym.BatchNorm(data=stage2_unit4_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage2_unit4_bn3')
    stage2_unit4_relu3 = mx.sym.Activation(data=stage2_unit4_bn3, act_type="relu", name='stage2_unit4_relu3')
    stage2_unit4_conv3 = mx.sym.Convolution(data=stage2_unit4_relu3, num_filter=512, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage2_unit4_conv3')
    _plus6 = mx.sym.elemwise_add(lhs=stage2_unit4_conv3, rhs=_plus5, name='_plus6')


    stage3_unit1_bn1 = mx.sym.BatchNorm(data=_plus6, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit1_bn1')
    stage3_unit1_relu1 = mx.sym.Activation(data=stage3_unit1_bn1, act_type="relu", name='stage3_unit1_relu1')
    stage3_unit1_conv1 = mx.sym.Convolution(data=stage3_unit1_relu1, num_filter=256, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit1_conv1')

    stage3_unit1_bn2 = mx.sym.BatchNorm(data=stage3_unit1_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit1_bn2')
    stage3_unit1_relu2 = mx.sym.Activation(data=stage3_unit1_bn2, act_type="relu", name='stage3_unit1_relu2')
    stage3_unit1_conv2 = mx.sym.Convolution(data=stage3_unit1_relu2, num_filter=256, kernel=(3,3), stride=(2,2), pad=(1, 1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit1_conv2')

    stage3_unit1_bn3 = mx.sym.BatchNorm(data=stage3_unit1_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit1_bn3')
    stage3_unit1_relu3 = mx.sym.Activation(data=stage3_unit1_bn3, act_type="relu", name='stage3_unit1_relu3')
    stage3_unit1_conv3 = mx.sym.Convolution(data=stage3_unit1_relu3, num_filter=1024, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit1_conv3')
    stage3_unit1_sc = mx.sym.Convolution(data=stage3_unit1_relu1, num_filter=1024, kernel=(1, 1), stride=(2,2), pad=(0, 0), dilate=(1, 1),
                                         num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256,
                                         name='stage3_unit1_sc')
    _plus7 = mx.sym.elemwise_add(lhs=stage3_unit1_conv3, rhs=stage3_unit1_sc, name='_plus7')

    stage3_unit2_bn1 = mx.sym.BatchNorm(data=_plus7, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit2_bn1')
    stage3_unit2_relu1 = mx.sym.Activation(data=stage3_unit2_bn1, act_type="relu", name='stage3_unit2_relu1')
    stage3_unit2_conv1 = mx.sym.Convolution(data=stage3_unit2_relu1, num_filter=256, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit2_conv1')

    stage3_unit2_bn2 = mx.sym.BatchNorm(data=stage3_unit2_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit2_bn2')
    stage3_unit2_relu2 = mx.sym.Activation(data=stage3_unit2_bn2, act_type="relu", name='stage3_unit2_relu2')
    stage3_unit2_conv2 = mx.sym.Convolution(data=stage3_unit2_relu2, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1, 1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit2_conv2')

    stage3_unit2_bn3 = mx.sym.BatchNorm(data=stage3_unit2_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit2_bn3')
    stage3_unit2_relu3 = mx.sym.Activation(data=stage3_unit2_bn3, act_type="relu", name='stage3_unit2_relu3')
    stage3_unit2_conv3 = mx.sym.Convolution(data=stage3_unit2_relu3, num_filter=1024, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit2_conv3')

    _plus8 = mx.sym.elemwise_add(lhs=stage3_unit2_conv3, rhs=_plus7, name='_plus8')

    stage3_unit3_bn1 = mx.sym.BatchNorm(data=_plus8, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit3_bn1')
    stage3_unit3_relu1 = mx.sym.Activation(data=stage3_unit3_bn1, act_type="relu", name='stage3_unit3_relu1')
    stage3_unit3_conv1 = mx.sym.Convolution(data=stage3_unit3_relu1, num_filter=256, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit3_conv1')

    stage3_unit3_bn2 = mx.sym.BatchNorm(data=stage3_unit3_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit3_bn2')
    stage3_unit3_relu2 = mx.sym.Activation(data=stage3_unit3_bn2, act_type="relu", name='stage3_unit3_relu2')
    stage3_unit3_conv2 = mx.sym.Convolution(data=stage3_unit3_relu2, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1, 1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit3_conv2')

    stage3_unit3_bn3 = mx.sym.BatchNorm(data=stage3_unit3_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit3_bn3')
    stage3_unit3_relu3 = mx.sym.Activation(data=stage3_unit3_bn3, act_type="relu", name='stage3_unit3_relu3')
    stage3_unit3_conv3 = mx.sym.Convolution(data=stage3_unit3_relu3, num_filter=1024, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit3_conv3')

    _plus9 = mx.sym.elemwise_add(lhs=stage3_unit3_conv3, rhs=_plus8, name='_plus9')

    stage3_unit4_bn1 = mx.sym.BatchNorm(data=_plus9, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit4_bn1')
    stage3_unit4_relu1 = mx.sym.Activation(data=stage3_unit4_bn1, act_type="relu", name='stage3_unit4_relu1')
    stage3_unit4_conv1 = mx.sym.Convolution(data=stage3_unit4_relu1, num_filter=256, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit4_conv1')

    stage3_unit4_bn2 = mx.sym.BatchNorm(data=stage3_unit4_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit4_bn2')
    stage3_unit4_relu2 = mx.sym.Activation(data=stage3_unit4_bn2, act_type="relu", name='stage3_unit4_relu2')
    stage3_unit4_conv2 = mx.sym.Convolution(data=stage3_unit4_relu2, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1, 1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit4_conv2')

    stage3_unit4_bn3 = mx.sym.BatchNorm(data=stage3_unit4_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit4_bn3')
    stage3_unit4_relu3 = mx.sym.Activation(data=stage3_unit4_bn3, act_type="relu", name='stage3_unit4_relu3')
    stage3_unit4_conv3 = mx.sym.Convolution(data=stage3_unit4_relu3, num_filter=1024, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit4_conv3')

    _plus10 = mx.sym.elemwise_add(lhs=stage3_unit4_conv3, rhs=_plus9, name='_plus10')

    stage3_unit5_bn1 = mx.sym.BatchNorm(data=_plus10, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit5_bn1')
    stage3_unit5_relu1 = mx.sym.Activation(data=stage3_unit5_bn1, act_type="relu", name='stage3_unit5_relu1')
    stage3_unit5_conv1 = mx.sym.Convolution(data=stage3_unit5_relu1, num_filter=256, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit5_conv1')

    stage3_unit5_bn2 = mx.sym.BatchNorm(data=stage3_unit5_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit5_bn2')
    stage3_unit5_relu2 = mx.sym.Activation(data=stage3_unit5_bn2, act_type="relu", name='stage3_unit5_relu2')
    stage3_unit5_conv2 = mx.sym.Convolution(data=stage3_unit5_relu2, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1, 1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit5_conv2')

    stage3_unit5_bn3 = mx.sym.BatchNorm(data=stage3_unit5_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit5_bn3')
    stage3_unit5_relu3 = mx.sym.Activation(data=stage3_unit5_bn3, act_type="relu", name='stage3_unit5_relu3')
    stage3_unit5_conv3 = mx.sym.Convolution(data=stage3_unit5_relu3, num_filter=1024, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit5_conv3')

    _plus11 = mx.sym.elemwise_add(lhs=stage3_unit5_conv3, rhs=_plus10, name='_plus11')

    stage3_unit6_bn1 = mx.sym.BatchNorm(data=_plus11, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit6_bn1')
    stage3_unit6_relu1 = mx.sym.Activation(data=stage3_unit6_bn1, act_type="relu", name='stage3_unit6_relu1')
    stage3_unit6_conv1 = mx.sym.Convolution(data=stage3_unit6_relu1, num_filter=256, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit6_conv1')

    stage3_unit6_bn2 = mx.sym.BatchNorm(data=stage3_unit6_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit6_bn2')
    stage3_unit6_relu2 = mx.sym.Activation(data=stage3_unit6_bn2, act_type="relu", name='stage3_unit6_relu2')
    stage3_unit6_conv2 = mx.sym.Convolution(data=stage3_unit6_relu2, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1, 1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit6_conv2')

    stage3_unit6_bn3 = mx.sym.BatchNorm(data=stage3_unit6_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage3_unit6_bn3')
    stage3_unit6_relu3 = mx.sym.Activation(data=stage3_unit6_bn3, act_type="relu", name='stage3_unit6_relu3')
    stage3_unit6_conv3 = mx.sym.Convolution(data=stage3_unit6_relu3, num_filter=1024, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage3_unit6_conv3')

    _plus12 = mx.sym.elemwise_add(lhs=stage3_unit6_conv3, rhs=_plus11, name='_plus12')

    stage4_unit1_bn1 = mx.sym.BatchNorm(data=_plus12, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage4_unit1_bn1')
    stage4_unit1_relu1 = mx.sym.Activation(data=stage4_unit1_bn1, act_type="relu", name='stage4_unit1_relu1')
    stage4_unit1_conv1 = mx.sym.Convolution(data=stage4_unit1_relu1, num_filter=512, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage4_unit1_conv1')

    stage4_unit1_bn2 = mx.sym.BatchNorm(data=stage4_unit1_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage4_unit1_bn2')
    stage4_unit1_relu2 = mx.sym.Activation(data=stage4_unit1_bn2, act_type="relu", name='stage4_unit1_relu2')
    stage4_unit1_conv2 = mx.sym.Convolution(data=stage4_unit1_relu2, num_filter=512, kernel=(3,3), stride=(2,2), pad=(1, 1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage4_unit1_conv2')

    stage4_unit1_bn3 = mx.sym.BatchNorm(data=stage4_unit1_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage4_unit1_bn3')
    stage4_unit1_relu3 = mx.sym.Activation(data=stage4_unit1_bn3, act_type="relu", name='stage4_unit1_relu3')
    stage4_unit1_conv3 = mx.sym.Convolution(data=stage4_unit1_relu3, num_filter=2048, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage4_unit1_conv3')
    stage4_unit1_sc = mx.sym.Convolution(data=stage4_unit1_relu1, num_filter=2048, kernel=(1, 1), stride=(2, 2),
                                         pad=(0, 0), dilate=(1, 1),
                                         num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256,
                                         name='stage4_unit1_sc')
    _plus13 = mx.sym.elemwise_add(lhs=stage4_unit1_conv3, rhs=stage4_unit1_sc, name='_plus13')

    stage4_unit2_bn1 = mx.sym.BatchNorm(data=_plus13, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage4_unit2_bn1')
    stage4_unit2_relu1 = mx.sym.Activation(data=stage4_unit2_bn1, act_type="relu", name='stage4_unit2_relu1')
    stage4_unit2_conv1 = mx.sym.Convolution(data=stage4_unit2_relu1, num_filter=512, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage4_unit2_conv1')

    stage4_unit2_bn2 = mx.sym.BatchNorm(data=stage4_unit2_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage4_unit2_bn2')
    stage4_unit2_relu2 = mx.sym.Activation(data=stage4_unit2_bn2, act_type="relu", name='stage4_unit2_relu2')
    stage4_unit2_conv2 = mx.sym.Convolution(data=stage4_unit2_relu2, num_filter=512, kernel=(3,3), stride=(1,1), pad=(1, 1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage4_unit2_conv2')

    stage4_unit2_bn3 = mx.sym.BatchNorm(data=stage4_unit2_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage4_unit2_bn3')
    stage4_unit2_relu3 = mx.sym.Activation(data=stage4_unit2_bn3, act_type="relu", name='stage4_unit2_relu3')
    stage4_unit2_conv3 = mx.sym.Convolution(data=stage4_unit2_relu3, num_filter=2048, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage4_unit2_conv3')

    _plus14 = mx.sym.elemwise_add(lhs=stage4_unit2_conv3, rhs=_plus13, name='_plus14')

    stage4_unit3_bn1 = mx.sym.BatchNorm(data=_plus14, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage4_unit3_bn1')
    stage4_unit3_relu1 = mx.sym.Activation(data=stage4_unit3_bn1, act_type="relu", name='stage4_unit3_relu1')
    stage4_unit3_conv1 = mx.sym.Convolution(data=stage4_unit3_relu1, num_filter=512, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage4_unit3_conv1')

    stage4_unit3_bn2 = mx.sym.BatchNorm(data=stage4_unit3_conv1, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage4_unit3_bn2')
    stage4_unit3_relu2 = mx.sym.Activation(data=stage4_unit3_bn2, act_type="relu", name='stage4_unit3_relu2')
    stage4_unit3_conv2 = mx.sym.Convolution(data=stage4_unit3_relu2, num_filter=512, kernel=(3,3), stride=(1,1), pad=(1, 1), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage4_unit3_conv2')

    stage4_unit3_bn3 = mx.sym.BatchNorm(data=stage4_unit3_conv2, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='stage4_unit3_bn3')
    stage4_unit3_relu3 = mx.sym.Activation(data=stage4_unit3_bn3, act_type="relu", name='stage4_unit3_relu3')
    stage4_unit3_conv3 = mx.sym.Convolution(data=stage4_unit3_relu3, num_filter=2048, kernel=(1,1), stride=(1, 1), pad=(0,0), dilate=(1, 1),
                                            num_group=1, no_bias=True, cudnn_tune='limited_workspace', workspace=256, name='stage4_unit3_conv3')

    _plus15 = mx.sym.elemwise_add(lhs=stage4_unit3_conv3, rhs=_plus14, name='_plus15')

    bn1 = mx.sym.BatchNorm(data=_plus15, fix_gamma=False, use_global_stats=False, eps=2e-05,
                                        momentum=0.9, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type="relu", name='relu1')

    ssh_c3_lateral = mx.sym.Convolution(data=relu1, num_filter=256, kernel=(1,1), stride=(1, 1), pad=(0,0), name='ssh_c3_lateral')
    ssh_c3_lateral_bn = mx.sym.BatchNorm(data=ssh_c3_lateral, fix_gamma=False, eps=2e-05,
                                        momentum=0.9, name='ssh_c3_lateral_bn')
    ssh_c3_lateral_relu = mx.sym.Activation(data=ssh_c3_lateral_bn, act_type="relu", name='ssh_c3_lateral_relu')

    ssh_m3_det_conv1 = mx.sym.Convolution(data=ssh_c3_lateral_relu, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1, 1), name='ssh_m3_det_conv1')
    ssh_m3_det_conv1_bn = mx.sym.BatchNorm(data=ssh_m3_det_conv1, fix_gamma=False, eps=2e-05, momentum=0.9, name='ssh_m3_det_conv1_bn')

    ssh_m3_det_context_conv1 = mx.sym.Convolution(data=ssh_c3_lateral_relu, num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='ssh_m3_det_context_conv1')
    ssh_m3_det_context_conv1_bn = mx.sym.BatchNorm(data=ssh_m3_det_context_conv1, fix_gamma=False, eps=2e-05, momentum=0.9,
                                           name='ssh_m3_det_context_conv1_bn')
    ssh_m3_det_context_conv1_relu = mx.sym.Activation(data=ssh_m3_det_context_conv1_bn, act_type="relu", name='ssh_m3_det_context_conv1_relu')

    ssh_m3_det_context_conv2 = mx.sym.Convolution(data=ssh_m3_det_context_conv1_relu, num_filter=128, kernel=(3, 3), stride=(1, 1),
                                          pad=(1, 1), name='ssh_m3_det_context_conv2')
    ssh_m3_det_context_conv2_bn = mx.sym.BatchNorm(data=ssh_m3_det_context_conv2, fix_gamma=False, eps=2e-05, momentum=0.9,
                                           name='ssh_m3_det_context_conv2_bn')

    ssh_m3_det_context_conv3_1 = mx.sym.Convolution(data=ssh_m3_det_context_conv1_relu, num_filter=128, kernel=(3, 3), stride=(1, 1),
                                          pad=(1, 1), name='ssh_m3_det_context_conv3_1')
    ssh_m3_det_context_conv3_1_bn = mx.sym.BatchNorm(data=ssh_m3_det_context_conv3_1, fix_gamma=False, eps=2e-05, momentum=0.9,
                                           name='ssh_m3_det_context_conv3_1_bn')
    ssh_m3_det_context_conv3_1_relu = mx.sym.Activation(data=ssh_m3_det_context_conv3_1_bn, act_type="relu",
                                                      name='ssh_m3_det_context_conv3_1_relu')

    ssh_m3_det_context_conv3_2 = mx.sym.Convolution(data=ssh_m3_det_context_conv3_1_relu, num_filter=128, kernel=(3, 3),
                                                    stride=(1, 1),
                                                    pad=(1, 1), name='ssh_m3_det_context_conv3_2')
    ssh_m3_det_context_conv3_2_bn = mx.sym.BatchNorm(data=ssh_m3_det_context_conv3_2, fix_gamma=False, eps=2e-05,
                                                     momentum=0.9, name='ssh_m3_det_context_conv3_2_bn')

    ssh_m3_det_concat = mx.sym.Concat(ssh_m3_det_conv1_bn, ssh_m3_det_context_conv2_bn, ssh_m3_det_context_conv3_2_bn,
                                     dim=1,  name='ssh_m3_det_concat')

    ssh_m3_det_concat_relu = mx.sym.Activation(data=ssh_m3_det_concat, act_type="relu", name='ssh_m3_det_concat_relu')


    face_rpn_cls_score_stride32 = mx.sym.Convolution(data=ssh_m3_det_concat_relu, num_filter=4, kernel=(1, 1),
                                                    stride=(1, 1),pad=(0, 0), name='face_rpn_cls_score_stride32')
    face_rpn_cls_score_reshape_stride32 = mx.sym.Reshape(data=face_rpn_cls_score_stride32,shape=(0, 2, -1, 0), name='face_rpn_cls_score_reshape_stride32')
    face_rpn_cls_prob_stride32 = mx.sym.SoftmaxActivation(data=face_rpn_cls_score_reshape_stride32, mode="channel",
                                               name='face_rpn_cls_prob_stride32')
    face_rpn_cls_prob_reshape_stride32 = mx.sym.Reshape(data=face_rpn_cls_prob_stride32, shape=(0, 4, -1, 0),
                                                         name='face_rpn_cls_prob_reshape_stride32')
    face_rpn_bbox_pred_stride32 = mx.sym.Convolution(data=ssh_m3_det_concat_relu, num_filter=8, kernel=(1, 1),
                                                     stride=(1, 1),pad=(0, 0), name='face_rpn_bbox_pred_stride32')
    face_rpn_landmark_pred_stride32 = mx.sym.Convolution(data=ssh_m3_det_concat_relu, num_filter=20, kernel=(1, 1),
                                                     stride=(1, 1), pad=(0, 0), name='face_rpn_landmark_pred_stride32')

    ssh_c2_lateral = mx.sym.Convolution(data=stage4_unit1_relu2, num_filter=256, kernel=(1, 1),
                                                     stride=(1, 1), pad=(0, 0), name='ssh_c2_lateral')
    ssh_c2_lateral_bn = mx.sym.BatchNorm(data=ssh_c2_lateral, fix_gamma=False, eps=2e-05,momentum=0.9,name='ssh_c2_lateral_bn')
    ssh_c2_lateral_relu = mx.sym.Activation(data=ssh_c2_lateral_bn, act_type="relu", name='ssh_c2_lateral_relu')
    ssh_c3_up = mx.sym.UpSampling(ssh_c3_lateral_relu,sample_type= "nearest", scale=2,workspace=512, name='ssh_c3_up')

    crop0 = mx.sym.Crop(*[ssh_c3_up, ssh_c2_lateral_relu], name='crop0')
    _plus16 = mx.sym.elemwise_add(lhs=ssh_c2_lateral_relu, rhs=crop0, name='_plus16')

    ssh_c2_aggr = mx.sym.Convolution(data=_plus16, num_filter=256, kernel=(3, 3),
                                        stride=(1, 1), pad=(1, 1), name='ssh_c2_aggr')
    ssh_c2_aggr_bn = mx.sym.BatchNorm(data=ssh_c2_aggr, fix_gamma=False, eps=2e-05, momentum=0.9,
                                         name='ssh_c2_aggr_bn')
    ssh_c2_aggr_relu = mx.sym.Activation(data=ssh_c2_aggr_bn, act_type="relu", name='ssh_c2_aggr_relu')

    ssh_m2_det_conv1 = mx.sym.Convolution(data=ssh_c2_aggr_relu, num_filter=256, kernel=(3, 3),
                                     stride=(1, 1), pad=(1, 1), name='ssh_m2_det_conv1')
    ssh_m2_det_conv1_bn = mx.sym.BatchNorm(data=ssh_m2_det_conv1, fix_gamma=False, eps=2e-05, momentum=0.9,
                                      name='ssh_m2_det_conv1_bn')

    ssh_m2_det_context_conv1 = mx.sym.Convolution(data=ssh_c2_aggr_relu, num_filter=128, kernel=(3, 3),
                                     stride=(1, 1), pad=(1, 1), name='ssh_m2_det_context_conv1')
    ssh_m2_det_context_conv1_bn = mx.sym.BatchNorm(data=ssh_m2_det_context_conv1, fix_gamma=False, eps=2e-05, momentum=0.9,
                                      name='ssh_m2_det_context_conv1_bn')
    ssh_m2_det_context_conv1_relu = mx.sym.Activation(data=ssh_m2_det_context_conv1_bn, act_type="relu", name='ssh_m2_det_context_conv1_relu')

    ssh_m2_det_context_conv2 = mx.sym.Convolution(data=ssh_m2_det_context_conv1_relu, num_filter=128, kernel=(3, 3),
                                     stride=(1, 1), pad=(1, 1), name='ssh_m2_det_context_conv2')
    ssh_m2_det_context_conv2_bn = mx.sym.BatchNorm(data=ssh_m2_det_context_conv2, fix_gamma=False, eps=2e-05, momentum=0.9,
                                      name='ssh_m2_det_context_conv2_bn')

    ssh_m2_det_context_conv3_1 = mx.sym.Convolution(data=ssh_m2_det_context_conv1_relu, num_filter=128, kernel=(3, 3),
                                     stride=(1, 1), pad=(1, 1), name='ssh_m2_det_context_conv3_1')
    ssh_m2_det_context_conv3_1_bn = mx.sym.BatchNorm(data=ssh_m2_det_context_conv3_1, fix_gamma=False, eps=2e-05, momentum=0.9,
                                      name='ssh_m2_det_context_conv3_1_bn')
    ssh_m2_det_context_conv3_1_relu = mx.sym.Activation(data=ssh_m2_det_context_conv3_1_bn, act_type="relu", name='ssh_m2_det_context_conv3_1_relu')
    ssh_m2_det_context_conv3_2 = mx.sym.Convolution(data=ssh_m2_det_context_conv3_1_relu, num_filter=128, kernel=(3, 3),
                                     stride=(1, 1), pad=(1, 1), name='ssh_m2_det_context_conv3_2')
    ssh_m2_det_context_conv3_2_bn = mx.sym.BatchNorm(data=ssh_m2_det_context_conv3_2, fix_gamma=False, eps=2e-05, momentum=0.9,
                                      name='ssh_m2_det_context_conv3_2_bn')

    ssh_m2_det_concat = mx.sym.Concat(ssh_m2_det_conv1_bn, ssh_m2_det_context_conv2_bn,ssh_m2_det_context_conv3_2_bn,
                                      dim=1,  name='ssh_m2_det_concat')
    ssh_m2_det_concat_relu = mx.sym.Activation(data=ssh_m2_det_concat, act_type="relu", name='ssh_m2_det_concat_relu')

    face_rpn_cls_score_stride16 = mx.sym.Convolution(data=ssh_m2_det_concat_relu, num_filter=4, kernel=(1, 1),
                                     stride=(1, 1), pad=(0, 0), name='face_rpn_cls_score_stride16')

    face_rpn_cls_score_reshape_stride16 = mx.sym.Reshape(data=face_rpn_cls_score_stride16, shape=(0, 2, -1, 0),
                                                         name='face_rpn_cls_score_reshape_stride16')
    face_rpn_cls_prob_stride16 = mx.sym.SoftmaxActivation(data=face_rpn_cls_score_reshape_stride16, mode="channel",
                                               name='face_rpn_cls_prob_stride16')
    face_rpn_cls_prob_reshape_stride16 = mx.sym.Reshape(data=face_rpn_cls_prob_stride16, shape=(0, 4, -1, 0),
                                                         name='face_rpn_cls_prob_reshape_stride16')


    face_rpn_bbox_pred_stride16 = mx.sym.Convolution(data=ssh_m2_det_concat_relu, num_filter=8, kernel=(1, 1),
                                     stride=(1, 1), pad=(0, 0), name='face_rpn_bbox_pred_stride16')
    face_rpn_landmark_pred_stride16 = mx.sym.Convolution(data=ssh_m2_det_concat_relu, num_filter=20, kernel=(1, 1),
                                     stride=(1, 1), pad=(0, 0), name='face_rpn_landmark_pred_stride16')
    ssh_m1_red_conv = mx.sym.Convolution(data=stage3_unit1_relu2, num_filter=256, kernel=(1, 1),
                                     stride=(1, 1), pad=(0, 0), name='ssh_m1_red_conv')
    ssh_m1_red_conv_bn = mx.sym.BatchNorm(data=ssh_m1_red_conv, fix_gamma=False, eps=2e-05, momentum=0.9,
                                      name='ssh_m1_red_conv_bn')
    ssh_m1_red_conv_relu = mx.sym.Activation(data=ssh_m1_red_conv_bn, act_type="relu", name='ssh_m1_red_conv_relu')
    ssh_m2_red_up = mx.sym.UpSampling(ssh_c2_aggr_relu,sample_type= "nearest", scale=2,workspace=512, name='ssh_m2_red_up')
    crop1 = mx.sym.Crop(*[ssh_m2_red_up, ssh_m1_red_conv_relu], name='crop1')
    _plus17 = mx.sym.elemwise_add(lhs=ssh_m1_red_conv_relu, rhs=crop1, name='_plus17')

    ssh_c1_aggr = mx.sym.Convolution(data=_plus17, num_filter=256, kernel=(3, 3),
                                         stride=(1, 1), pad=(1, 1), name='ssh_c1_aggr')
    ssh_c1_aggr_bn = mx.sym.BatchNorm(data=ssh_c1_aggr, fix_gamma=False, eps=2e-05, momentum=0.9,
                                         name='ssh_c1_aggr_bn')
    ssh_c1_aggr_relu = mx.sym.Activation(data=ssh_c1_aggr_bn, act_type="relu", name='ssh_c1_aggr_relu')

    ssh_m1_det_conv1 = mx.sym.Convolution(data=ssh_c1_aggr_relu, num_filter=256, kernel=(3, 3),
                                     stride=(1, 1), pad=(1, 1), name='ssh_m1_det_conv1')
    ssh_m1_det_conv1_bn = mx.sym.BatchNorm(data=ssh_m1_det_conv1, fix_gamma=False, eps=2e-05, momentum=0.9,
                                      name='ssh_m1_det_conv1_bn')

    weight = mx.symbol.Variable(name="{}_weight".format("ssh_m1_det_context_conv1"),
        init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
    bias = mx.symbol.Variable(name="{}_bias".format("ssh_m1_det_context_conv1"),
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(0.0)})

    ssh_m1_det_context_conv1 = mx.sym.Convolution(data=ssh_c1_aggr_relu, num_filter=128, kernel=(3, 3),
                                     stride=(1, 1), pad=(1, 1), name='ssh_m1_det_context_conv1', weight=weight, bias=bias)
    ssh_m1_det_context_conv1_bn = mx.sym.BatchNorm(data=ssh_m1_det_context_conv1, fix_gamma=False, eps=2e-05, momentum=0.9,
                                           name='ssh_m1_det_context_conv1_bn')
    ssh_m1_det_context_conv1_relu = mx.sym.Activation(data=ssh_m1_det_context_conv1_bn, act_type="relu", name='ssh_m1_det_context_conv1_relu')

    ssh_m1_det_context_conv2 = mx.sym.Convolution(data=ssh_m1_det_context_conv1_relu, num_filter=128, kernel=(3, 3),
                                                  stride=(1, 1), pad=(1, 1), name='ssh_m1_det_context_conv2')
    ssh_m1_det_context_conv2_bn = mx.sym.BatchNorm(data=ssh_m1_det_context_conv2, fix_gamma=False, eps=2e-05, momentum=0.9,
                                           name='ssh_m1_det_context_conv2_bn')
    ssh_m1_det_context_conv3_1 = mx.sym.Convolution(data=ssh_m1_det_context_conv1_relu, num_filter=128, kernel=(3, 3),
                                                  stride=(1, 1), pad=(1, 1), name='ssh_m1_det_context_conv3_1')
    ssh_m1_det_context_conv3_1_bn = mx.sym.BatchNorm(data=ssh_m1_det_context_conv3_1, fix_gamma=False, eps=2e-05, momentum=0.9,
                                           name='ssh_m1_det_context_conv3_1_bn')
    ssh_m1_det_context_conv3_1_relu = mx.sym.Activation(data=ssh_m1_det_context_conv3_1_bn, act_type="relu", name='ssh_m1_det_context_conv3_1_relu')

    ssh_m1_det_context_conv3_2 = mx.sym.Convolution(data=ssh_m1_det_context_conv3_1_relu, num_filter=128, kernel=(3, 3),
                                                  stride=(1, 1), pad=(1, 1), name='ssh_m1_det_context_conv3_2')
    ssh_m1_det_context_conv3_2_bn = mx.sym.BatchNorm(data=ssh_m1_det_context_conv3_2, fix_gamma=False, eps=2e-05, momentum=0.9,
                                           name='ssh_m1_det_context_conv3_2_bn')

    ssh_m1_det_concat = mx.sym.Concat(ssh_m1_det_conv1_bn, ssh_m1_det_context_conv2_bn,ssh_m1_det_context_conv3_2_bn,
                                      dim=1,  name='ssh_m1_det_concat')
    ssh_m1_det_concat_relu = mx.sym.Activation(data=ssh_m1_det_concat, act_type="relu", name='ssh_m1_det_concat_relu')
    face_rpn_cls_score_stride8 = mx.sym.Convolution(data=ssh_m1_det_concat_relu, num_filter=4, kernel=(1, 1),
                                                  stride=(1, 1), pad=(0, 0), name='face_rpn_cls_score_stride8')
    face_rpn_cls_score_reshape_stride8 = mx.sym.Reshape(data=face_rpn_cls_score_stride8, shape=(0, 2, -1, 0),
                                                         name='face_rpn_cls_score_reshape_stride8')
    face_rpn_cls_prob_stride8 = mx.sym.SoftmaxActivation(data=face_rpn_cls_score_reshape_stride8, mode="channel",
                                               name='face_rpn_cls_prob_stride8')
    face_rpn_cls_prob_reshape_stride8 = mx.sym.Reshape(data=face_rpn_cls_prob_stride8, shape=(0, 4, -1, 0),
                                                         name='face_rpn_cls_prob_reshape_stride8')
    face_rpn_bbox_pred_stride8 = mx.sym.Convolution(data=ssh_m1_det_concat_relu, num_filter=8, kernel=(1, 1),
                                                  stride=(1, 1), pad=(0, 0), name='face_rpn_bbox_pred_stride8')
    face_rpn_landmark_pred_stride8 = mx.sym.Convolution(data=ssh_m1_det_concat_relu, num_filter=20, kernel=(1, 1),
                                                  stride=(1, 1), pad=(0, 0), name='face_rpn_landmark_pred_stride8')
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


    return mx.sym.Group(ret_group)

def save_symbol():
    in_data = mx.symbol.Variable(name='data')

    symbol = get_r50_symbol(in_data)
    symbol.save("test-symbol.json")


if __name__ == '__main__':
    # get_feature_symbol_mobileface_v1()
    save_symbol()