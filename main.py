import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.api.models import load_model

number_reader = load_model('Number Reader.keras')
