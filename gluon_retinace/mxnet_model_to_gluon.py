import mxnet as mx
from gluon_retinace.retinaface_gluon import RetinfaceGluon
from mxnet.gluon import nn

data = mx.nd.ones((1,3,512,512), ctx=mx.gpu())

json_path = '../model/R50-symbol.json'
param_path = '../model/R50-0000.params'
net = nn.SymbolBlock.imports(json_path, ['data'], param_file=param_path, ctx=mx.gpu())

gluon_net = RetinfaceGluon()
gluon_net_params = gluon_net.collect_params()._params
for param in net.collect_params()._params.items():
    if 'bn_data_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm0_gamma']._data = param[1]._data
    if 'bn_data_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm0_beta']._data = param[1]._data
    if 'bn_data_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm0_running_mean']._data = param[1]._data
    if 'bn_data_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm0_running_var']._data = param[1]._data

    if 'conv0_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv0_weight']._data = param[1]._data

    if 'bn0_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm1_gamma']._data = param[1]._data
    if 'bn0_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm1_beta']._data = param[1]._data
    if 'bn0_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm1_running_mean']._data = param[1]._data
    if 'bn0_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm1_running_var']._data = param[1]._data

    if 'stage1_unit1_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm2_gamma']._data = param[1]._data
    if 'stage1_unit1_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm2_beta']._data = param[1]._data
    if 'stage1_unit1_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm2_running_mean']._data = param[1]._data
    if 'stage1_unit1_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm2_running_var']._data = param[1]._data
    if 'stage1_unit1_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv1_weight']._data = param[1]._data

    if 'stage1_unit1_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm3_gamma']._data = param[1]._data
    if 'stage1_unit1_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm3_beta']._data = param[1]._data
    if 'stage1_unit1_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm3_running_mean']._data = param[1]._data
    if 'stage1_unit1_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm3_running_var']._data = param[1]._data
    if 'stage1_unit1_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv2_weight']._data = param[1]._data

    if 'stage1_unit1_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm4_gamma']._data = param[1]._data
    if 'stage1_unit1_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm4_beta']._data = param[1]._data
    if 'stage1_unit1_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm4_running_mean']._data = param[1]._data
    if 'stage1_unit1_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm4_running_var']._data = param[1]._data
    if 'stage1_unit1_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv3_weight']._data = param[1]._data
    if 'stage1_unit1_sc_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv4_weight']._data = param[1]._data

    if 'stage1_unit2_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm5_gamma']._data = param[1]._data
    if 'stage1_unit2_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm5_beta']._data = param[1]._data
    if 'stage1_unit2_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm5_running_mean']._data = param[1]._data
    if 'stage1_unit2_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm5_running_var']._data = param[1]._data
    if 'stage1_unit2_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv5_weight']._data = param[1]._data

    if 'stage1_unit2_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm6_gamma']._data = param[1]._data
    if 'stage1_unit2_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm6_beta']._data = param[1]._data
    if 'stage1_unit2_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm6_running_mean']._data = param[1]._data
    if 'stage1_unit2_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm6_running_var']._data = param[1]._data
    if 'stage1_unit2_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv6_weight']._data = param[1]._data

    if 'stage1_unit2_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm7_gamma']._data = param[1]._data
    if 'stage1_unit2_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm7_beta']._data = param[1]._data
    if 'stage1_unit2_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm7_running_mean']._data = param[1]._data
    if 'stage1_unit2_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm7_running_var']._data = param[1]._data
    if 'stage1_unit2_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv7_weight']._data = param[1]._data

    if 'stage1_unit3_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm8_gamma']._data = param[1]._data
    if 'stage1_unit3_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm8_beta']._data = param[1]._data
    if 'stage1_unit3_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm8_running_mean']._data = param[1]._data
    if 'stage1_unit3_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm8_running_var']._data = param[1]._data
    if 'stage1_unit3_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv8_weight']._data = param[1]._data

    if 'stage1_unit3_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm9_gamma']._data = param[1]._data
    if 'stage1_unit3_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm9_beta']._data = param[1]._data
    if 'stage1_unit3_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm9_running_mean']._data = param[1]._data
    if 'stage1_unit3_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm9_running_var']._data = param[1]._data
    if 'stage1_unit3_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv9_weight']._data = param[1]._data

    if 'stage1_unit3_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm10_gamma']._data = param[1]._data
    if 'stage1_unit3_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm10_beta']._data = param[1]._data
    if 'stage1_unit3_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm10_running_mean']._data = param[1]._data
    if 'stage1_unit3_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm10_running_var']._data = param[1]._data
    if 'stage1_unit3_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv10_weight']._data = param[1]._data

    if 'stage2_unit1_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm11_gamma']._data = param[1]._data
    if 'stage2_unit1_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm11_beta']._data = param[1]._data
    if 'stage2_unit1_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm11_running_mean']._data = param[1]._data
    if 'stage2_unit1_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm11_running_var']._data = param[1]._data
    if 'stage2_unit1_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv11_weight']._data = param[1]._data

    if 'stage2_unit1_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm12_gamma']._data = param[1]._data
    if 'stage2_unit1_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm12_beta']._data = param[1]._data
    if 'stage2_unit1_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm12_running_mean']._data = param[1]._data
    if 'stage2_unit1_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm12_running_var']._data = param[1]._data
    if 'stage2_unit1_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv12_weight']._data = param[1]._data

    if 'stage2_unit1_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm13_gamma']._data = param[1]._data
    if 'stage2_unit1_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm13_beta']._data = param[1]._data
    if 'stage2_unit1_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm13_running_mean']._data = param[1]._data
    if 'stage2_unit1_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm13_running_var']._data = param[1]._data
    if 'stage2_unit1_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv13_weight']._data = param[1]._data
    if 'stage2_unit1_sc_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv14_weight']._data = param[1]._data

    if 'stage2_unit2_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm14_gamma']._data = param[1]._data
    if 'stage2_unit2_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm14_beta']._data = param[1]._data
    if 'stage2_unit2_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm14_running_mean']._data = param[1]._data
    if 'stage2_unit2_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm14_running_var']._data = param[1]._data
    if 'stage2_unit2_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv15_weight']._data = param[1]._data

    if 'stage2_unit2_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm15_gamma']._data = param[1]._data
    if 'stage2_unit2_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm15_beta']._data = param[1]._data
    if 'stage2_unit2_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm15_running_mean']._data = param[1]._data
    if 'stage2_unit2_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm15_running_var']._data = param[1]._data
    if 'stage2_unit2_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv16_weight']._data = param[1]._data

    if 'stage2_unit2_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm16_gamma']._data = param[1]._data
    if 'stage2_unit2_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm16_beta']._data = param[1]._data
    if 'stage2_unit2_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm16_running_mean']._data = param[1]._data
    if 'stage2_unit2_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm16_running_var']._data = param[1]._data
    if 'stage2_unit2_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv17_weight']._data = param[1]._data

    if 'stage2_unit3_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm17_gamma']._data = param[1]._data
    if 'stage2_unit3_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm17_beta']._data = param[1]._data
    if 'stage2_unit3_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm17_running_mean']._data = param[1]._data
    if 'stage2_unit3_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm17_running_var']._data = param[1]._data
    if 'stage2_unit3_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv18_weight']._data = param[1]._data

    if 'stage2_unit3_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm18_gamma']._data = param[1]._data
    if 'stage2_unit3_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm18_beta']._data = param[1]._data
    if 'stage2_unit3_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm18_running_mean']._data = param[1]._data
    if 'stage2_unit3_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm18_running_var']._data = param[1]._data
    if 'stage2_unit3_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv19_weight']._data = param[1]._data

    if 'stage2_unit3_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm19_gamma']._data = param[1]._data
    if 'stage2_unit3_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm19_beta']._data = param[1]._data
    if 'stage2_unit3_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm19_running_mean']._data = param[1]._data
    if 'stage2_unit3_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm19_running_var']._data = param[1]._data
    if 'stage2_unit3_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv20_weight']._data = param[1]._data

    if 'stage2_unit4_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm20_gamma']._data = param[1]._data
    if 'stage2_unit4_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm20_beta']._data = param[1]._data
    if 'stage2_unit4_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm20_running_mean']._data = param[1]._data
    if 'stage2_unit4_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm20_running_var']._data = param[1]._data
    if 'stage2_unit4_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv21_weight']._data = param[1]._data

    if 'stage2_unit4_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm21_gamma']._data = param[1]._data
    if 'stage2_unit4_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm21_beta']._data = param[1]._data
    if 'stage2_unit4_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm21_running_mean']._data = param[1]._data
    if 'stage2_unit4_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm21_running_var']._data = param[1]._data
    if 'stage2_unit4_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv22_weight']._data = param[1]._data

    if 'stage2_unit4_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm22_gamma']._data = param[1]._data
    if 'stage2_unit4_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm22_beta']._data = param[1]._data
    if 'stage2_unit4_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm22_running_mean']._data = param[1]._data
    if 'stage2_unit4_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm22_running_var']._data = param[1]._data
    if 'stage2_unit4_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv23_weight']._data = param[1]._data

    if 'stage3_unit1_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm23_gamma']._data = param[1]._data
    if 'stage3_unit1_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm23_beta']._data = param[1]._data
    if 'stage3_unit1_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm23_running_mean']._data = param[1]._data
    if 'stage3_unit1_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm23_running_var']._data = param[1]._data
    if 'stage3_unit1_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv24_weight']._data = param[1]._data

    if 'stage3_unit1_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm24_gamma']._data = param[1]._data
    if 'stage3_unit1_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm24_beta']._data = param[1]._data
    if 'stage3_unit1_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm24_running_mean']._data = param[1]._data
    if 'stage3_unit1_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm24_running_var']._data = param[1]._data
    if 'stage3_unit1_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv25_weight']._data = param[1]._data

    if 'stage3_unit1_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm25_gamma']._data = param[1]._data
    if 'stage3_unit1_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm25_beta']._data = param[1]._data
    if 'stage3_unit1_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm25_running_mean']._data = param[1]._data
    if 'stage3_unit1_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm25_running_var']._data = param[1]._data
    if 'stage3_unit1_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv26_weight']._data = param[1]._data
    if 'stage3_unit1_sc_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv27_weight']._data = param[1]._data

    if 'stage3_unit2_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm26_gamma']._data = param[1]._data
    if 'stage3_unit2_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm26_beta']._data = param[1]._data
    if 'stage3_unit2_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm26_running_mean']._data = param[1]._data
    if 'stage3_unit2_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm26_running_var']._data = param[1]._data
    if 'stage3_unit2_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv28_weight']._data = param[1]._data

    if 'stage3_unit2_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm27_gamma']._data = param[1]._data
    if 'stage3_unit2_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm27_beta']._data = param[1]._data
    if 'stage3_unit2_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm27_running_mean']._data = param[1]._data
    if 'stage3_unit2_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm27_running_var']._data = param[1]._data
    if 'stage3_unit2_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv29_weight']._data = param[1]._data

    if 'stage3_unit2_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm28_gamma']._data = param[1]._data
    if 'stage3_unit2_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm28_beta']._data = param[1]._data
    if 'stage3_unit2_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm28_running_mean']._data = param[1]._data
    if 'stage3_unit2_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm28_running_var']._data = param[1]._data
    if 'stage3_unit2_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv30_weight']._data = param[1]._data

    if 'stage3_unit3_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm29_gamma']._data = param[1]._data
    if 'stage3_unit3_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm29_beta']._data = param[1]._data
    if 'stage3_unit3_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm29_running_mean']._data = param[1]._data
    if 'stage3_unit3_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm29_running_var']._data = param[1]._data
    if 'stage3_unit3_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv31_weight']._data = param[1]._data

    if 'stage3_unit3_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm30_gamma']._data = param[1]._data
    if 'stage3_unit3_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm30_beta']._data = param[1]._data
    if 'stage3_unit3_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm30_running_mean']._data = param[1]._data
    if 'stage3_unit3_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm30_running_var']._data = param[1]._data
    if 'stage3_unit3_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv32_weight']._data = param[1]._data

    if 'stage3_unit3_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm31_gamma']._data = param[1]._data
    if 'stage3_unit3_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm31_beta']._data = param[1]._data
    if 'stage3_unit3_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm31_running_mean']._data = param[1]._data
    if 'stage3_unit3_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm31_running_var']._data = param[1]._data
    if 'stage3_unit3_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv33_weight']._data = param[1]._data

    if 'stage3_unit4_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm32_gamma']._data = param[1]._data
    if 'stage3_unit4_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm32_beta']._data = param[1]._data
    if 'stage3_unit4_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm32_running_mean']._data = param[1]._data
    if 'stage3_unit4_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm32_running_var']._data = param[1]._data
    if 'stage3_unit4_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv34_weight']._data = param[1]._data

    if 'stage3_unit4_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm33_gamma']._data = param[1]._data
    if 'stage3_unit4_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm33_beta']._data = param[1]._data
    if 'stage3_unit4_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm33_running_mean']._data = param[1]._data
    if 'stage3_unit4_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm33_running_var']._data = param[1]._data
    if 'stage3_unit4_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv35_weight']._data = param[1]._data

    if 'stage3_unit4_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm34_gamma']._data = param[1]._data
    if 'stage3_unit4_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm34_beta']._data = param[1]._data
    if 'stage3_unit4_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm34_running_mean']._data = param[1]._data
    if 'stage3_unit4_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm34_running_var']._data = param[1]._data
    if 'stage3_unit4_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv36_weight']._data = param[1]._data

    if 'stage3_unit5_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm35_gamma']._data = param[1]._data
    if 'stage3_unit5_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm35_beta']._data = param[1]._data
    if 'stage3_unit5_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm35_running_mean']._data = param[1]._data
    if 'stage3_unit5_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm35_running_var']._data = param[1]._data
    if 'stage3_unit5_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv37_weight']._data = param[1]._data

    if 'stage3_unit5_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm36_gamma']._data = param[1]._data
    if 'stage3_unit5_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm36_beta']._data = param[1]._data
    if 'stage3_unit5_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm36_running_mean']._data = param[1]._data
    if 'stage3_unit5_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm36_running_var']._data = param[1]._data
    if 'stage3_unit5_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv38_weight']._data = param[1]._data

    if 'stage3_unit5_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm37_gamma']._data = param[1]._data
    if 'stage3_unit5_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm37_beta']._data = param[1]._data
    if 'stage3_unit5_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm37_running_mean']._data = param[1]._data
    if 'stage3_unit5_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm37_running_var']._data = param[1]._data
    if 'stage3_unit5_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv39_weight']._data = param[1]._data

    if 'stage3_unit6_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm38_gamma']._data = param[1]._data
    if 'stage3_unit6_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm38_beta']._data = param[1]._data
    if 'stage3_unit6_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm38_running_mean']._data = param[1]._data
    if 'stage3_unit6_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm38_running_var']._data = param[1]._data
    if 'stage3_unit6_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv40_weight']._data = param[1]._data

    if 'stage3_unit6_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm39_gamma']._data = param[1]._data
    if 'stage3_unit6_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm39_beta']._data = param[1]._data
    if 'stage3_unit6_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm39_running_mean']._data = param[1]._data
    if 'stage3_unit6_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm39_running_var']._data = param[1]._data
    if 'stage3_unit6_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv41_weight']._data = param[1]._data

    if 'stage3_unit6_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm40_gamma']._data = param[1]._data
    if 'stage3_unit6_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm40_beta']._data = param[1]._data
    if 'stage3_unit6_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm40_running_mean']._data = param[1]._data
    if 'stage3_unit6_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm40_running_var']._data = param[1]._data
    if 'stage3_unit6_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv42_weight']._data = param[1]._data

    if 'stage4_unit1_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm41_gamma']._data = param[1]._data
    if 'stage4_unit1_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm41_beta']._data = param[1]._data
    if 'stage4_unit1_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm41_running_mean']._data = param[1]._data
    if 'stage4_unit1_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm41_running_var']._data = param[1]._data
    if 'stage4_unit1_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv43_weight']._data = param[1]._data

    if 'stage4_unit1_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm42_gamma']._data = param[1]._data
    if 'stage4_unit1_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm42_beta']._data = param[1]._data
    if 'stage4_unit1_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm42_running_mean']._data = param[1]._data
    if 'stage4_unit1_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm42_running_var']._data = param[1]._data
    if 'stage4_unit1_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv44_weight']._data = param[1]._data

    if 'stage4_unit1_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm43_gamma']._data = param[1]._data
    if 'stage4_unit1_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm43_beta']._data = param[1]._data
    if 'stage4_unit1_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm43_running_mean']._data = param[1]._data
    if 'stage4_unit1_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm43_running_var']._data = param[1]._data
    if 'stage4_unit1_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv45_weight']._data = param[1]._data
    if 'stage4_unit1_sc_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv46_weight']._data = param[1]._data

    if 'stage4_unit2_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm44_gamma']._data = param[1]._data
    if 'stage4_unit2_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm44_beta']._data = param[1]._data
    if 'stage4_unit2_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm44_running_mean']._data = param[1]._data
    if 'stage4_unit2_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm44_running_var']._data = param[1]._data
    if 'stage4_unit2_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv47_weight']._data = param[1]._data

    if 'stage4_unit2_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm45_gamma']._data = param[1]._data
    if 'stage4_unit2_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm45_beta']._data = param[1]._data
    if 'stage4_unit2_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm45_running_mean']._data = param[1]._data
    if 'stage4_unit2_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm45_running_var']._data = param[1]._data
    if 'stage4_unit2_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv48_weight']._data = param[1]._data

    if 'stage4_unit2_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm46_gamma']._data = param[1]._data
    if 'stage4_unit2_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm46_beta']._data = param[1]._data
    if 'stage4_unit2_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm46_running_mean']._data = param[1]._data
    if 'stage4_unit2_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm46_running_var']._data = param[1]._data
    if 'stage4_unit2_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv49_weight']._data = param[1]._data

    if 'stage4_unit3_bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm47_gamma']._data = param[1]._data
    if 'stage4_unit3_bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm47_beta']._data = param[1]._data
    if 'stage4_unit3_bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm47_running_mean']._data = param[1]._data
    if 'stage4_unit3_bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm47_running_var']._data = param[1]._data
    if 'stage4_unit3_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv50_weight']._data = param[1]._data

    if 'stage4_unit3_bn2_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm48_gamma']._data = param[1]._data
    if 'stage4_unit3_bn2_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm48_beta']._data = param[1]._data
    if 'stage4_unit3_bn2_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm48_running_mean']._data = param[1]._data
    if 'stage4_unit3_bn2_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm48_running_var']._data = param[1]._data
    if 'stage4_unit3_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv51_weight']._data = param[1]._data

    if 'stage4_unit3_bn3_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm49_gamma']._data = param[1]._data
    if 'stage4_unit3_bn3_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm49_beta']._data = param[1]._data
    if 'stage4_unit3_bn3_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm49_running_mean']._data = param[1]._data
    if 'stage4_unit3_bn3_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm49_running_var']._data = param[1]._data
    if 'stage4_unit3_conv3_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv52_weight']._data = param[1]._data

    if 'bn1_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm50_gamma']._data = param[1]._data
    if 'bn1_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm50_beta']._data = param[1]._data
    if 'bn1_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm50_running_mean']._data = param[1]._data
    if 'bn1_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm50_running_var']._data = param[1]._data

    if 'ssh_c3_lateral_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv53_weight']._data = param[1]._data
    if 'ssh_c3_lateral_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv53_bias']._data = param[1]._data
    if 'ssh_c3_lateral_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm51_gamma']._data = param[1]._data
    if 'ssh_c3_lateral_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm51_beta']._data = param[1]._data
    if 'ssh_c3_lateral_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm51_running_mean']._data = param[1]._data
    if 'ssh_c3_lateral_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm51_running_var']._data = param[1]._data

    if 'ssh_m3_det_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv54_weight']._data = param[1]._data
    if 'ssh_m3_det_conv1_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv54_bias']._data = param[1]._data
    if 'ssh_m3_det_conv1_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm52_gamma']._data = param[1]._data
    if 'ssh_m3_det_conv1_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm52_beta']._data = param[1]._data
    if 'ssh_m3_det_conv1_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm52_running_mean']._data = param[1]._data
    if 'ssh_m3_det_conv1_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm52_running_var']._data = param[1]._data

    if 'ssh_m3_det_context_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv55_weight']._data = param[1]._data
    if 'ssh_m3_det_context_conv1_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv55_bias']._data = param[1]._data
    if 'ssh_m3_det_context_conv1_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm53_gamma']._data = param[1]._data
    if 'ssh_m3_det_context_conv1_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm53_beta']._data = param[1]._data
    if 'ssh_m3_det_context_conv1_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm53_running_mean']._data = param[1]._data
    if 'ssh_m3_det_context_conv1_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm53_running_var']._data = param[1]._data

    if 'ssh_m3_det_context_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv56_weight']._data = param[1]._data
    if 'ssh_m3_det_context_conv2_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv56_bias']._data = param[1]._data
    if 'ssh_m3_det_context_conv2_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm54_gamma']._data = param[1]._data
    if 'ssh_m3_det_context_conv2_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm54_beta']._data = param[1]._data
    if 'ssh_m3_det_context_conv2_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm54_running_mean']._data = param[1]._data
    if 'ssh_m3_det_context_conv2_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm54_running_var']._data = param[1]._data

    if 'ssh_m3_det_context_conv3_1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv57_weight']._data = param[1]._data
    if 'ssh_m3_det_context_conv3_1_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv57_bias']._data = param[1]._data
    if 'ssh_m3_det_context_conv3_1_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm55_gamma']._data = param[1]._data
    if 'ssh_m3_det_context_conv3_1_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm55_beta']._data = param[1]._data
    if 'ssh_m3_det_context_conv3_1_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm55_running_mean']._data = param[1]._data
    if 'ssh_m3_det_context_conv3_1_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm55_running_var']._data = param[1]._data

    if 'ssh_m3_det_context_conv3_2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv58_weight']._data = param[1]._data
    if 'ssh_m3_det_context_conv3_2_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv58_bias']._data = param[1]._data
    if 'ssh_m3_det_context_conv3_2_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm56_gamma']._data = param[1]._data
    if 'ssh_m3_det_context_conv3_2_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm56_beta']._data = param[1]._data
    if 'ssh_m3_det_context_conv3_2_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm56_running_mean']._data = param[1]._data
    if 'ssh_m3_det_context_conv3_2_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm56_running_var']._data = param[1]._data

    if 'face_rpn_cls_score_stride32_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv59_weight']._data = param[1]._data
    if 'face_rpn_cls_score_stride32_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv59_bias']._data = param[1]._data

    if 'face_rpn_bbox_pred_stride32_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv60_weight']._data = param[1]._data
    if 'face_rpn_bbox_pred_stride32_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv60_bias']._data = param[1]._data

    if 'face_rpn_landmark_pred_stride32_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv61_weight']._data = param[1]._data
    if 'face_rpn_landmark_pred_stride32_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv61_bias']._data = param[1]._data

    if 'ssh_c2_lateral_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv62_weight']._data = param[1]._data
    if 'ssh_c2_lateral_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv62_bias']._data = param[1]._data
    if 'ssh_c2_lateral_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm57_gamma']._data = param[1]._data
    if 'ssh_c2_lateral_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm57_beta']._data = param[1]._data
    if 'ssh_c2_lateral_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm57_running_mean']._data = param[1]._data
    if 'ssh_c2_lateral_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm57_running_var']._data = param[1]._data

    if 'ssh_c2_aggr_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv63_weight']._data = param[1]._data
    if 'ssh_c2_aggr_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv63_bias']._data = param[1]._data
    if 'ssh_c2_aggr_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm58_gamma']._data = param[1]._data
    if 'ssh_c2_aggr_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm58_beta']._data = param[1]._data
    if 'ssh_c2_aggr_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm58_running_mean']._data = param[1]._data
    if 'ssh_c2_aggr_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm58_running_var']._data = param[1]._data

    if 'ssh_m2_det_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv64_weight']._data = param[1]._data
    if 'ssh_m2_det_conv1_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv64_bias']._data = param[1]._data
    if 'ssh_m2_det_conv1_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm59_gamma']._data = param[1]._data
    if 'ssh_m2_det_conv1_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm59_beta']._data = param[1]._data
    if 'ssh_m2_det_conv1_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm59_running_mean']._data = param[1]._data
    if 'ssh_m2_det_conv1_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm59_running_var']._data = param[1]._data

    if 'ssh_m2_det_context_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv65_weight']._data = param[1]._data
    if 'ssh_m2_det_context_conv1_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv65_bias']._data = param[1]._data
    if 'ssh_m2_det_context_conv1_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm60_gamma']._data = param[1]._data
    if 'ssh_m2_det_context_conv1_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm60_beta']._data = param[1]._data
    if 'ssh_m2_det_context_conv1_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm60_running_mean']._data = param[1]._data
    if 'ssh_m2_det_context_conv1_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm60_running_var']._data = param[1]._data

    if 'ssh_m2_det_context_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv66_weight']._data = param[1]._data
    if 'ssh_m2_det_context_conv2_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv66_bias']._data = param[1]._data
    if 'ssh_m2_det_context_conv2_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm61_gamma']._data = param[1]._data
    if 'ssh_m2_det_context_conv2_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm61_beta']._data = param[1]._data
    if 'ssh_m2_det_context_conv2_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm61_running_mean']._data = param[1]._data
    if 'ssh_m2_det_context_conv2_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm61_running_var']._data = param[1]._data

    if 'ssh_m2_det_context_conv3_1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv67_weight']._data = param[1]._data
    if 'ssh_m2_det_context_conv3_1_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv67_bias']._data = param[1]._data
    if 'ssh_m2_det_context_conv3_1_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm62_gamma']._data = param[1]._data
    if 'ssh_m2_det_context_conv3_1_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm62_beta']._data = param[1]._data
    if 'ssh_m2_det_context_conv3_1_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm62_running_mean']._data = param[1]._data
    if 'ssh_m2_det_context_conv3_1_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm62_running_var']._data = param[1]._data

    if 'ssh_m2_det_context_conv3_2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv68_weight']._data = param[1]._data
    if 'ssh_m2_det_context_conv3_2_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv68_bias']._data = param[1]._data
    if 'ssh_m2_det_context_conv3_2_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm63_gamma']._data = param[1]._data
    if 'ssh_m2_det_context_conv3_2_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm63_beta']._data = param[1]._data
    if 'ssh_m2_det_context_conv3_2_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm63_running_mean']._data = param[1]._data
    if 'ssh_m2_det_context_conv3_2_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm63_running_var']._data = param[1]._data

    if 'face_rpn_cls_score_stride16_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv69_weight']._data = param[1]._data
    if 'face_rpn_cls_score_stride16_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv69_bias']._data = param[1]._data

    if 'face_rpn_bbox_pred_stride16_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv70_weight']._data = param[1]._data
    if 'face_rpn_bbox_pred_stride16_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv70_bias']._data = param[1]._data

    if 'face_rpn_landmark_pred_stride16_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv71_weight']._data = param[1]._data
    if 'face_rpn_landmark_pred_stride16_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv71_bias']._data = param[1]._data

    if 'ssh_m1_red_conv_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv72_weight']._data = param[1]._data
    if 'ssh_m1_red_conv_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv72_bias']._data = param[1]._data
    if 'ssh_m1_red_conv_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm64_gamma']._data = param[1]._data
    if 'ssh_m1_red_conv_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm64_beta']._data = param[1]._data
    if 'ssh_m1_red_conv_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm64_running_mean']._data = param[1]._data
    if 'ssh_m1_red_conv_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm64_running_var']._data = param[1]._data

    if 'ssh_c1_aggr_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv73_weight']._data = param[1]._data
    if 'ssh_c1_aggr_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv73_bias']._data = param[1]._data
    if 'ssh_c1_aggr_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm65_gamma']._data = param[1]._data
    if 'ssh_c1_aggr_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm65_beta']._data = param[1]._data
    if 'ssh_c1_aggr_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm65_running_mean']._data = param[1]._data
    if 'ssh_c1_aggr_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm65_running_var']._data = param[1]._data

    if 'ssh_m1_det_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv74_weight']._data = param[1]._data
    if 'ssh_m1_det_conv1_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv74_bias']._data = param[1]._data
    if 'ssh_m1_det_conv1_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm66_gamma']._data = param[1]._data
    if 'ssh_m1_det_conv1_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm66_beta']._data = param[1]._data
    if 'ssh_m1_det_conv1_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm66_running_mean']._data = param[1]._data
    if 'ssh_m1_det_conv1_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm66_running_var']._data = param[1]._data

    if 'ssh_m1_det_context_conv1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv75_weight']._data = param[1]._data
    if 'ssh_m1_det_context_conv1_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv75_bias']._data = param[1]._data
    if 'ssh_m1_det_context_conv1_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm67_gamma']._data = param[1]._data
    if 'ssh_m1_det_context_conv1_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm67_beta']._data = param[1]._data
    if 'ssh_m1_det_context_conv1_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm67_running_mean']._data = param[1]._data
    if 'ssh_m1_det_context_conv1_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm67_running_var']._data = param[1]._data

    if 'ssh_m1_det_context_conv2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv76_weight']._data = param[1]._data
    if 'ssh_m1_det_context_conv2_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv76_bias']._data = param[1]._data
    if 'ssh_m1_det_context_conv2_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm68_gamma']._data = param[1]._data
    if 'ssh_m1_det_context_conv2_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm68_beta']._data = param[1]._data
    if 'ssh_m1_det_context_conv2_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm68_running_mean']._data = param[1]._data
    if 'ssh_m1_det_context_conv2_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm68_running_var']._data = param[1]._data

    if 'ssh_m1_det_context_conv3_1_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv77_weight']._data = param[1]._data
    if 'ssh_m1_det_context_conv3_1_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv77_bias']._data = param[1]._data
    if 'ssh_m1_det_context_conv3_1_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm69_gamma']._data = param[1]._data
    if 'ssh_m1_det_context_conv3_1_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm69_beta']._data = param[1]._data
    if 'ssh_m1_det_context_conv3_1_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm69_running_mean']._data = param[1]._data
    if 'ssh_m1_det_context_conv3_1_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm69_running_var']._data = param[1]._data

    if 'ssh_m1_det_context_conv3_2_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv78_weight']._data = param[1]._data
    if 'ssh_m1_det_context_conv3_2_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv78_bias']._data = param[1]._data
    if 'ssh_m1_det_context_conv3_2_bn_gamma' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm70_gamma']._data = param[1]._data
    if 'ssh_m1_det_context_conv3_2_bn_beta' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm70_beta']._data = param[1]._data
    if 'ssh_m1_det_context_conv3_2_bn_moving_mean' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm70_running_mean']._data = param[1]._data
    if 'ssh_m1_det_context_conv3_2_bn_moving_var' == param[0]:
        gluon_net_params['retinfacegluon0_batchnorm70_running_var']._data = param[1]._data

    if 'face_rpn_cls_score_stride8_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv79_weight']._data = param[1]._data
    if 'face_rpn_cls_score_stride8_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv79_bias']._data = param[1]._data

    if 'face_rpn_bbox_pred_stride8_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv80_weight']._data = param[1]._data
    if 'face_rpn_bbox_pred_stride8_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv80_bias']._data = param[1]._data

    if 'face_rpn_landmark_pred_stride8_weight' == param[0]:
        gluon_net_params['retinfacegluon0_conv81_weight']._data = param[1]._data
    if 'face_rpn_landmark_pred_stride8_bias' == param[0]:
        gluon_net_params['retinfacegluon0_conv81_bias']._data = param[1]._data


gluon_net.save_parameters('gluon_net.params')








