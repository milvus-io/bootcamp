import tensorflow as tf
from keras.applications.vgg16 import VGG16
from tensorflow.python.keras.backend import set_session
graph = None
model = None
sess = None

# def load_model():
#     global graph
#     graph = tf.get_default_graph()

#     global model
#     model = VGG16(weights='imagenet',
#                   input_shape=(224, 224, 3),
#                   pooling='max',
#                   include_top=False)

# config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# # config.gpu_options.per_process_gpu_memory_fraction = 0.5
# global sess
# sess = tf.Session(config=config)
# set_session(sess)

# load_model()
