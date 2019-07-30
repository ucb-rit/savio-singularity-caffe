import caffe
import numpy as np
from google.protobuf import text_format

caffe.set_mode_gpu()
# select GPU device number
caffe.set_device(0)

# Load the Reference CaffeNet model
model_path_caffenet = './'
net_fn_caffenet   = model_path_caffenet + 'deploy.prototxt'
param_fn_caffenet = model_path_caffenet + 'bvlc_reference_caffenet.caffemodel'

# Patching model to be able to compute gradients.
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn_caffenet).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

# construct a classifier based on the Reference CaffeNet model
net_caffenet = caffe.Classifier('tmp.prototxt', param_fn_caffenet,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
