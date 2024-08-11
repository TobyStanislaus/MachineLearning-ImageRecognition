import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from makeModel import use_model, build_model

if __name__ == 'main':
    use_model('data\\test')
    print('finished')