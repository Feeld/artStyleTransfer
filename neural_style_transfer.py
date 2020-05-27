# import the necessary packages
import imutils
import numpy as np
import cv2
from PIL import Image

def style_trans( model, pil_image ):
    # load the neural style transfer model from disk
    net = cv2.dnn.readNetFromTorch(model)
    
    # load the input image, resize it to have a width of 600 pixels, and
    # then grab the image dimensions
    
    # use numpy to convert the pil_image into a numpy array
    numpy_image=np.array(pil_image)  

    # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that 
    # the color is converted from RGB to BGR format
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    image = imutils.resize(opencv_image, width=600)
    (h, w) = image.shape[:2]
    
    # construct a blob from the image, set the input, and then perform a
    # forward pass of the network
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    # start = time.time()
    output = net.forward()
    # end = time.time()

    # reshape the output tensor, add back in the mean subtraction, and
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)
    
    # then swap the channel ordering
    output = output[:,:,::-1]

    # Clip the range b/w 0 and 255 to prevent saturation
    output = np.clip(output * 255, 0, 255) # proper [0..255] range
    # Steps to convert array to PIL image format
    output = output.astype(np.uint8) # safe conversion
    output = Image.fromarray(output)

    return output
