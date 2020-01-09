
#imports:
import numpy as np
import matplotlib.pyplot as plt
import extras

#load files:
train_im_file = open('./datasets/train_images', 'rb')
train_label_file = open('./datasets/train_labels', 'rb')

train_im_file.read(16)#first 16 bytes are not required
train_label_file.read(8)# 8 here...

image_size = 28
num_images = 60000

buf = train_im_file.read(num_images * image_size * image_size)
feat_mat = np.frombuffer(buf, dtype= np.uint8)
feat_mat = feat_mat.reshape(num_images, image_size * image_size)
#print(feat_mat.shape): 60000, 28, 28, 1

buf = train_label_file.read(num_images, )
target_vec = np.frombuffer(buf, dtype=np.uint8)
target_vec = target_vec.reshape(num_images)

#plot the data to test:
#extras.draw_sample(feat_mat, target_vec, 5)

#---------------------------------------------------------#
input_layer = image_size * image_size
hidden_layer = 25
num_labels = 10

Theta1 = np.random.rand(hidden_layer,input_layer + 1)
Theta2 = np.random.rand(num_labels, hidden_layer + 1)

J, Theta1, Theta2 = extras.cost(Theta1,
                                Theta2,
                                input_layer,
                                hidden_layer,
                                feat_mat, 
                                target_vec,
                                learning_rate= 0.1)


#---------------------------------------------------#
#load image:

from PIL import Image
import PIL.ImageOps

im = Image.open('test.png').convert('L')
im = im.resize((28, 28), Image.ANTIALIAS)
im = PIL.ImageOps.invert(im)

test_im = np.array(im)
test_im = test_im.reshape(1, image_size * image_size)

test_im = feat_mat[0].reshape(1, image_size * image_size)
predicition = extras.predict(test_im, Theta1, Theta2)

print(predicition)

