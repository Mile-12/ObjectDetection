import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('Model.h5')

label_map ={'Crab': 0,
 'Haddock': 1,
 'Hake': 2,
 'Ling': 3,
 'Octopus': 4,
 'Plaice, European': 5,
 'Ray': 6,
 'Sardine': 7}


def prediction(image_path):
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
            return {
                "Prediction" : pred, 
                "Confidence" : np.max(preds)
            }
            
