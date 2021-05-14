import os
from utils import load_data
from model.fcn.net import *


def initialization(epoch, batch_size, data_root, save_path, load_path, cache=True, train=True, unity=False):
    print("Initializing model...")
    # conf['epoch'] = epoch
    # conf['batch_size'] = batch_size
    # conf['save_path'] = save_path
    # conf['load_path'] = load_path
    if unity:
        train_source = None
        test_source = None
    elif train:
        train_source = load_data(os.path.join(data_root, "Train"), cache=cache)
        test_source = load_data(os.path.join(data_root, "Test"), cache=cache)
    else:
        train_source = load_data(data_root, cache=cache)
        test_source = train_source
    # conf['train_source'] = train_source
    # conf['test_source'] = test_source

    # model = Model(**conf)
    # da_rnn_kwargs = {"batch_size": 64, "T": 10}
    # _, model = da_rnn(None, 0, .001, da_rnn_kwargs)

    model = Model()
    print("Model initialization complete.")
    return model
