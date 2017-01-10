# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from array import array
import os
# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
#caffe_root = '../project/Caffe/'
caffe_root='/home/jarvis/normal/caffe/'
#sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0,caffe_root+'python')
import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
#model_weights = 'new_net.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')

mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'bgr', zip('BGR', mu)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

# create transformer for the input called 'data'

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)

labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
          
labels = np.loadtxt(labels_file, str, delimiter='\t')

#for i in range(0,3):
    #image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
  #image=caffe.io.load_image('football.jpg')
#try:
image = caffe.io.load_image('/home/jarvis/normal/jarvisBoardCode/'+sys.argv[1])
#temp= '/home/iplab_apu/Desktop/ankit/drive-download-20160823T093738Z/ISBI2016_ISIC_Part1_Training_Data/ISIC_'+index +'.jpg'   
#print temp
image = caffe.io.resize(image, (227,227))
transformed_image=[]
transformed_image = transformer.preprocess('data', image)

#plt.imshow(image)
net.blobs['data'].data[...] = transformed_image
t=time.clock()
output = net.forward()
t2=time.clock()
output_prob = output['prob'][0]

#listFeatureVector8.append(net.blobs['data'].data[0])
print "time",(t2-t)
print 'output label:', labels[output_prob.argmax()]

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:'
for i in top_inds:
    print output_prob[i], labels[i]

#except(Exception):
 # print "Didn't find ",i
count=0
jsonOutput={}
jsonOutput['results']=[]
for i in top_inds:
	res={}
	count=count+1
	res[str(count)]=labels[i][10:]
	jsonOutput['results'].append(res)
with open('/home/jarvis/normal/jarvisBoardCode/predictions.txt','w') as outfile:
	json.dump(jsonOutput,outfile)
os.chmod("/home/jarvis/normal/jarvisBoardCode/predictions.txt", 0o777)
print top_inds
print jsonOutput
