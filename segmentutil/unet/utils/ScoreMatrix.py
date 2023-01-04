import numpy as np


class Stats:
    def __init__(self, stats_list):
        self.stats_dict = {}
        for stats in stats_list:
            self.stats_dict[stats] = []

    def Update(self, new_stats):
        for item, value in new_stats.items():
            if item in self.stats_dict.keys():
                self.stats_dict[item].append(value)


def Compute_IOU(prediction, labels):
    """
    Currently 2 classes only
    :param prediction: 1d array
    :param labels: 1d array
    :return: IOU
    """

    prediction = np.squeeze(np.array(prediction))
    labels = np.squeeze(np.array(labels))

    assert prediction.shape == labels.shape, 'shapes do not match'
    intersection = np.sum(prediction*labels)
    union = np.sum(prediction) + np.sum(labels) - intersection
    ep = 1e-5
    IOU = intersection/(union + ep)
    return IOU


def Compute_Dice(prediction, labels):
    pass


if __name__ == '__main__':
    pass
