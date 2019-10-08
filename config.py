'''
import models 
from config import DefaultConfig

opt = DefaultConfig()
lr = opt.lr
model = getattr(models, opt.model)
dataset = DogCat(opt.train_data_root)
--------

for modify 
opt = DefaultConifg()
new_config = {'lr':0.1,'use_gpu':False}
opt.parse(new_config)
'''
import warnings
import torch as t
class DefaultConfig(object):
    env = 'default'
    model = 'AlexNet'

    train_data_root = './data/train/'
    test_data_root = './data/test'
    load_model_path = 'checkpoints/model.pty'

    batch_size = 128
    use_gpu = True
    num_workers = 4 # how many workers for loading data
    print_freq = 20 # print every  N batch

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1 
    lr_decay = 0.95
    weight_decy = 1e-4
    def parse(self,kwargs):  # kwargs  字典集合 ex. ke=1
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning : opt has not attribut")
            setattr(self,k,v)
        print('user config:')
        for k,v in self.__class__.__dict__.items():
                if not k.startswith('__'):
                    print(k.getattr(self,k))
opt = DefaultConfig()