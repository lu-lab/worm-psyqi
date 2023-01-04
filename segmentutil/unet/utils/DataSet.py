import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import h5py as h5


class TestData(Dataset):
    def __init__(self, directory, mode='training', border=None, prob=0.5):
        self.prob = (prob, )*3
        self.image = []
        if mode == 'training':
            if not border:
                for patch_h5_dir in glob.glob(directory + '/*'):
                    patch_hr = h5.File(patch_h5_dir, 'r')
                    oriImg = patch_hr['ori'][()]
                    labImg = patch_hr['lab'][()]
                    mean_value = patch_hr['mean'][()]
                    std_value = patch_hr['std'][()]
                    oriImg = (oriImg - mean_value)/std_value
                    patch_hr.close()
                    self.image.append((torch.tensor(oriImg, dtype=torch.float32), torch.tensor(labImg, dtype=torch.float32)))
            else:
                for patch_h5_dir in glob.glob(directory + '/*'):
                    patch_hr = h5.File(patch_h5_dir, 'r')
                    oriImg = patch_hr['ori'][()]
                    labImg = patch_hr['lab'][()].astype(np.uint8)
                    if border is not None:
                        labImg = labImg[border:-border, border:-border]
                    mean_value = patch_hr['mean'][()]
                    std_value = patch_hr['std'][()]
                    oriImg = (oriImg - mean_value)/std_value
                    patch_hr.close()
                    self.image.append((torch.tensor(oriImg, dtype=torch.float32), torch.tensor(labImg, dtype=torch.float32)))

        else:
            raise ValueError('enter a right mode')

    def __getitem__(self, idx):
        prob_list = np.random.rand(3)
        img = self.image[idx]

        if prob_list[0] > self.prob[0]:
            img = self.Rotate(img)

        if prob_list[1] > self.prob[1]:
            img = self.FlipHorizontal(img)

        if prob_list[2] > self.prob[2]:
            img = self.FlipVertical(img)

        return img

    def __len__(self):
        return len(self.image)

    def Rotate(self, img):
        rotate_time = np.random.randint(low=1, high=4)
        rotate_img = torch.rot90(img[0], k=rotate_time, dims=[1, 2])
        rotate_label = torch.rot90(img[1], k=rotate_time, dims=[0, 1])
        return rotate_img, rotate_label

    def FlipHorizontal(self, img):
        return torch.flip(img[0], dims=[2]), torch.flip(img[1], dims=[1])

    def FlipVertical(self, img):
        return torch.flip(img[0], dims=[1]), torch.flip(img[1], dims=[0])

    def Split(self, fold: int = 5) -> tuple:
        """
        To randomly split training and validation data and generate sequence
        :param fold: k-fold validation, e.g. 10-fold, default is 5
        :return: tuple of sample passed into DataLoader -> (train_sampler, validation_sampler)
        """

        length = self.__len__()
        allIdx = np.array(range(length))
        np.random.shuffle(allIdx)  # Shuffle index to randomize
        splitIdx = np.floor(length / fold).astype(np.int)

        validationIdx = allIdx[:splitIdx]
        trainingIdx = allIdx[splitIdx:]

        train_sampler = sampler.SubsetRandomSampler(trainingIdx)
        validation_sampler = sampler.SubsetRandomSampler(validationIdx)

        return train_sampler, validation_sampler


class TestData_MultiScale(Dataset):
    def __init__(self, directory, mode='training', border=None, prob=0.5):
        self.prob = (prob, )*3
        self.image = []
        if mode == 'training':
            if not border:
                for patch_h5_dir in glob.glob(directory + '/*'):
                    patch_hr = h5.File(patch_h5_dir, 'r')
                    oriImg = patch_hr['ori'][()]
                    labImg = patch_hr['lab'][()]
                    mean_value_red = patch_hr['red_mean'][()]
                    std_value_red = patch_hr['red_std'][()]
                    mean_value_green = patch_hr['green_mean'][()]
                    std_value_green = patch_hr['green_std'][()]
                    oriImg_red = (oriImg[0, ...] - mean_value_red)/std_value_red
                    oriImg_green = (oriImg[1, ...] - mean_value_green) / std_value_green
                    patch_hr.close()
                    self.image.append((torch.tensor(np.stack((oriImg_red, oriImg_green), axis=0), dtype=torch.float32), torch.tensor(labImg, dtype=torch.float32)))
            else:
                for patch_h5_dir in glob.glob(directory + '/*'):
                    patch_hr = h5.File(patch_h5_dir, 'r')
                    oriImg = patch_hr['ori'][()]
                    labImg = patch_hr['lab'][()].astype(np.uint8)
                    if border is not None:
                        labImg = labImg[:, border:-border, border:-border]
                    mean_value_red = patch_hr['red_mean'][()]
                    std_value_red = patch_hr['red_std'][()]
                    mean_value_green = patch_hr['green_mean'][()]
                    std_value_green = patch_hr['green_std'][()]
                    oriImg_red = (oriImg[0, ...] - mean_value_red) / std_value_red
                    oriImg_green = (oriImg[1, ...] - mean_value_green) / std_value_green
                    patch_hr.close()
                    gg = torch.tensor(np.stack((oriImg_red, oriImg_green), axis=0), dtype=torch.float32)
                    self.image.append((torch.tensor(np.stack((oriImg_red, oriImg_green), axis=0), dtype=torch.float32), torch.tensor(labImg, dtype=torch.float32)))

        else:
            raise ValueError('enter a right mode')

    def __getitem__(self, idx):
        prob_list = np.random.rand(3)
        img = self.image[idx]

        if prob_list[0] < self.prob[0]:
            img = self.Rotate(img)

        if prob_list[1] < self.prob[1]:
            img = self.FlipHorizontal(img)

        if prob_list[2] < self.prob[2]:
            img = self.FlipVertical(img)

        return img

    def __len__(self):
        return len(self.image)

    def Rotate(self, img):
        rotate_time = np.random.randint(low=1, high=4)
        rotate_img = torch.rot90(img[0], k=rotate_time, dims=[1, 2])
        rotate_label = torch.rot90(img[1], k=rotate_time, dims=[0, 1])
        return rotate_img, rotate_label

    def FlipHorizontal(self, img):
        return torch.flip(img[0], dims=[2]), torch.flip(img[1], dims=[1])

    def FlipVertical(self, img):
        return torch.flip(img[0], dims=[1]), torch.flip(img[1], dims=[0])

    def Split(self, fold: int = 5) -> tuple:
        """
        To randomly split training and validation data and generate sequence
        :param fold: k-fold validation, e.g. 10-fold, default is 5
        :return: tuple of sample passed into DataLoader -> (train_sampler, validation_sampler)
        """

        length = self.__len__()
        allIdx = np.array(range(length))
        np.random.shuffle(allIdx)  # Shuffle index to randomize
        splitIdx = np.floor(length / fold).astype(np.int)

        validationIdx = allIdx[:splitIdx]
        trainingIdx = allIdx[splitIdx:]

        train_sampler = sampler.SubsetRandomSampler(trainingIdx)
        validation_sampler = sampler.SubsetRandomSampler(validationIdx)

        return train_sampler, validation_sampler



if __name__ == '__main__':
    gen = '/home/admin-kzhang91/888/1234'
    aa = TestData_MultiScale(directory=gen, border=10, prob=0)
    aa_loader = DataLoader(dataset=aa, batch_size=4, shuffle=True)
    print(aa.__getitem__())