import os
from keras.utils.data_utils import get_file

dir_ = os.path.dirname(os.path.realpath(__file__))
w_path = dir_ + '\Input'
m_path = dir_ + '\model\model'

wiki_path = 'https://www.dropbox.com/s/mwa404sonmyu884/wiki.zh.vec?dl=1'
file = get_file('wiki.zh.vec', wiki_path, cache_subdir = w_path)

model_path = 'https://www.dropbox.com/s/jilocc8hjofdhtb/model_epoch_16_step_42.data-00000-of-00001?dl=1'
file = get_file('model_epoch_16_step_42.data-00000-of-00001', model_path, cache_subdir = m_path)

