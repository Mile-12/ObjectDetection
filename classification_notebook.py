

credentials = {
  "bucket": "fish-detection1.0",
  "access_key_id": "ad44d4e7417b4741b1bacc980eb372d2",
  "secret_access_key": "84e6d5d95aa282f367bb91c29bb0fd69fdfaf5d91f2a4e8d",
  "endpoint_url": "https://s3.us.cloud-object-storage.appdomain.cloud"
}

"""## Install Prerequisites

We use Keras/Tensorflow to build the classification model, and visualize the process with matplotlib.
"""

# Import required libraries
import os
import uuid
import shutil
import json
#from botocore.client import Config
import ibm_boto3
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

"""## Read The Data

Here we build simple wrapper functions to read in the data from our cloud object storage buckets and extract it.

A download function from IBM Cloud
"""

def download_file_cos(credentials, local_file_name, key): 
    '''
    Wrapper function to download a file from cloud object storage using the
    credential dict provided and loading it into memory
    '''
    cos = ibm_boto3.client(
        service_name='s3',
        aws_access_key_id=credentials['access_key_id'],
        aws_secret_access_key=credentials['secret_access_key'],
        endpoint_url=credentials['endpoint_url'])
    try:
        res=cos.download_file(Bucket=credentials['bucket'], Key=key, Filename=local_file_name)
    except Exception as e:
        print(Exception, e)
    else:
        print('File Downloaded')

def get_annotations(credentials): 
    cos = ibm_boto3.client(
        service_name='s3',
        aws_access_key_id=credentials['access_key_id'],
        aws_secret_access_key=credentials['secret_access_key'],
        endpoint_url=credentials['endpoint_url'])
    try:
        return json.loads(cos.get_object(Bucket=credentials['bucket'], Key='_annotations.json')['Body'].read())
    except Exception as e:
        print(Exception, e)

base_path = 'data'
if os.path.exists(base_path) and os.path.isdir(base_path):
    shutil.rmtree(base_path)
os.makedirs(base_path, exist_ok=True)

annotations = get_annotations(credentials)

for i, image in enumerate(annotations['annotations'].keys()):
    label = annotations['annotations'][image][0]['label']
    os.makedirs(os.path.join(base_path, label), exist_ok=True)
    _, extension = os.path.splitext(image)
    local_path = os.path.join(base_path, label, str(uuid.uuid4()) + extension)
    download_file_cos(credentials, local_path, image)


"""## Build the Model

We start with a [MobileNetV2](https://arxiv.org/abs/1801.04381) architecture as the backbone [pretrained feature extractor](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet). We then add a couple of dense layers and a softmax layer to perfom the classification. We freeze the MobileNetV2 backbone with weights trained on ImageNet dataset and only train the dense layers and softmax layer that we have added.
"""

base_model=tf.keras.applications.MobileNetV2(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
x=base_model.output
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dense(512,activation='relu')(x) #dense layer 1
x=tf.keras.layers.Dense(256,activation='relu')(x) #dense layer 2
preds=tf.keras.layers.Dense(6,activation='softmax')(x) #final layer with softmax activation

model=tf.keras.Model(inputs=base_model.input,outputs=preds)

#Freeze layers from MobileNetV2 backbone (not to be trained)
for layer in base_model.layers:
    layer.trainable=False

#Prepare the training dataset as a data generator object
train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('data',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=10,
                                                 class_mode='categorical',
                                                 shuffle=True)

"""### Using Adam, categorical_crossentropy and accuracy as optimization method, loss function and metrics, respectively"""

# Build the model
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

"""## Train the model"""

from tensorflow import set_random_seed
set_random_seed(2)
step_size_train=5
log_file = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=4)

"""## Figure of Training Loss and Accuracy"""

# Model accuracy and loss vs epoch
plt.plot(log_file.history['acc'], '-bo', label="train_accuracy")
plt.plot(log_file.history['loss'], '-r*', label="train_loss")
plt.title('Training Loss and Accuracy')
plt.ylabel('Loss/Accuracy')
plt.xlabel('Epoch #')
plt.legend(loc='center right')
plt.show()


# Mapping labels 
label_map = (train_generator.class_indices)


# Creating a sample inference function
def prediction(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    preds = model.predict(x)
    #print('Predictions', preds)
    
    for pred, value in label_map.items():    
        if value == np.argmax(preds):
            print('Predicted class is:', pred)
            print('With a confidence score of: ', np.max(preds))
    
    return np.argmax(preds)
