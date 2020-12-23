from django.shortcuts import render
from django.core.files.storage import FileSystemStorage


import io
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import base64
import tensorflow as tf
import skimage.draw
from PIL import Image
import base64
from io import BytesIO
from keras import backend as K

# Root directory of the project
ROOT_DIR = os.path.abspath("/app/")

sys.path.append("/app/Floor_Mask_RCNN/")
sys.path.append("/app/Floor_Mask_RCNN/mrcnn/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "Floor_Mask_RCNN/samples/coco/"))  # To find local version

#import coco
from Floor_Mask_RCNN.samples import coco

from Floor_Mask_RCNN.samples.floor import floor

# Directory to save logs and trained model
#/home/sky/Desktop/Floor_Project/
MODEL_DIR = os.path.join("Floor_Mask_RCNN", "logs")


# Local path to trained weights file
#/home/sky/Desktop/Floor_Project/
FLOOR_MODEL_PATH = os.path.join("Floor_Mask_RCNN", "mask_rcnn_floor_0087.h5")

# Directory of images to run detection on
#IMAGE_DIR = os.path.join("/home/sky/Desktop/Floor_Project/Floor_Mask_RCNN", "floor_imgs_test")



config = floor.FloorConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create your views here.



def index(request):
	context = {"a": "hello"}

	return render(request, "index.html", context)



def predictImage(request):
	class_names = ['BG', 'floor']

	fileObj = request.FILES['filepath']

	fs=FileSystemStorage()
	filePathName = fs.save(fileObj.name, fileObj)
	filePathName = fs.url(filePathName)
	testimage = '.'+filePathName
	image = skimage.io.imread(testimage)

	K.clear_session()
	# Run detection
	with tf.Graph().as_default():

		# Create model object in inference mode.
		model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
		# Load weights trained on MS-COCO
		model.load_weights(FLOOR_MODEL_PATH, by_name=True)

		results = model.detect([image])
		
		# Visualize results
		r = results[0]

		

############################################################################################################

	def color_splash(image, mask, floor_image=None):
	    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
	    # Copy color pixels from the original color image where mask is set
	    if mask.shape[-1] > 0:
	    # We're treating all instances as one, so collapse the mask into one layer
	    	mask = (np.sum(mask, -1, keepdims=True) >= 1)
	    	splash = np.where(mask, image, gray).astype(np.uint8)
	    else:
	    	splash = gray.astype(np.uint8)
	    return splash
	    #floor = cv2.imread("./media/floor_img/37021.jpg")
	    #floor = cv2.resize(floor, image.shape[1::-1])
	    #floor_image = cv2.cvtColor(floor, cv2.COLOR_BGR2RGB)
	 
	    
	    #if mask.shape[-1] > 0:
	        #mask = (np.sum(mask, -1, keepdims=True) >= 1)
	        #splash = np.where(mask, floor_image, image).astype(np.uint8)
	    #else:
	        #splash = floor_image.astype(np.uint8)
	    #return splash


	def to_image(numpy_img):
		img = Image.fromarray(numpy_img, 'RGB')
		return img


	def to_data_uri(pil_img):
	    data = BytesIO()
	    pil_img.save(data, "JPEG") # pick your format
	    data64 = base64.b64encode(data.getvalue())
	    return u'data:img/jpeg;base64,'+data64.decode('utf-8')
	   

	splash= color_splash(image, r['masks'])    

	pil_img = to_image(splash)
	img_uri = to_data_uri(pil_img)
	
	


############################################################################################################
	

	return render(request, "index.html" , { "pred_image": img_uri })

	

