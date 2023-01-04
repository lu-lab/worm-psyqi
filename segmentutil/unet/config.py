import numpy as np


class Config:

    batch = 1
    weight_decay = 1e-4
    epoch = 200
    lr = 1e-3
    weight = np.array([1, 5])
    num_workers = 4
    num_channels = 1
    num_classes = 2
    num_planes = 7
    border = 10
    prob = 0.5
    beta = 5  # relative importance of green to red
    model_name = 'UNET25D_BASIC'
