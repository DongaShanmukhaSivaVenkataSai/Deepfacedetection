import tensorflow as tf
print(tf.__version__)

#Avoid OOM errors by setting GPU Memory Consumption Growth
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
tf.config.list_physical_devices('GPU')



#Load Image into TF Data Pipeline
images=tf.data.Dataset.list_files('Data\\Images\\*.jpg')#tf.data.Dataset→This is TensorFlow’s way of building efficient input pipelines.
#.list_files()→A method that lists all files that match a specified pattern.
images.as_numpy_iterator().next()
#To convert the TensorFlow dataset (images) into a Python iterator that returns the file paths as NumPy strings (i.e., byte strings like b'Data\\Images\\img1.jpg').



#This defines a Python function named load_image, which takes in one argument x.
#In this case,x will be a TensorFlow string tensor — the file path of the image.
def load_image(x):
    byte_img=tf.io.read_file(x)#tf.io.read_file(x) reads the image file in raw bytes.
    #A Tensor containing the raw bytes of the image (still compressed .jpg format).
    img=tf.io.decode_jpeg(byte_img)#Decodes the JPEG bytes into an actual image tensor (a 3D array of pixels).
    return img
images=images.map(load_image)
#Applies your load_image function to every element in the dataset.
#images is currently a dataset of file paths (like b'Data\\Images\\1.jpg')
#images.map(load_image) turns it into a dataset of image tensors
#Now, each item is the actual image data, not the file path anymore.

images.as_numpy_iterator().next()
#Converts the TensorFlow dataset into a Python iterator that yields NumPy arrays (image tensors).
#.next() gets the first image in the dataset as a NumPy array.

type(images)

import matplotlib.pyplot as plt


#view raw images with matplotlib
image_generator=images.batch(4).as_numpy_iterator()
plot_images=image_generator.next()
fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx,image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()