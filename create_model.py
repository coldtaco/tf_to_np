import pickle
import tensorflow as tf

from Layers import *

def create_model(tf_model_path, save_path='model'):
    """
    tf_model_path:(str) Path to trained tf model
    save_path:(str) Pickle save destination
    """
    l = []
    model = tf.keras.models.load_model('model_casia_run1.h5')
    for a in model.layers:
        if 'conv2d' in a.name:
            l.append(Conv2D(*[w.numpy() for w in a.weights], a._build_input_shape))
        elif 'max_pooling2d'in a.name:        
            l.append(MaxPool2D(a.strides))
        elif 'flatten' in a.name:
            l.append(Flatten())
        elif 'dense' in a.name:
            l.append(Dense(*[w.numpy() for w in a.weights]))        
        if 'activation' in a.__dict__:
            act = str(a.activation)
            if 'relu' in act:
                l.append(ReLU())
            elif 'softmax' in act:
                l.append(Softmax())
    m = Model(l)
    with open(save_path, 'wb') as f:
        pickle.dump(m, f)

if __name__ == '__main__':
    tf_model_path = 'model_casia_run1.h5'
    save_path = 'model'
    create_model(tf_model_path, save_path)