import os
from model.fcn.net import *


def initialization(epoch, batch_size, data_root, save_path, load_path, cache=True, train=True, unity=False):
    print("Initializing model...")

    if unity:
        train_source = None
        test_source = None

    model = Model()
    print("Model initialization complete.")
    return model
