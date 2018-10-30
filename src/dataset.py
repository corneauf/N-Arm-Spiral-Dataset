import numpy as np
from torch.utils.data import Dataset

class NArmSpiral(Dataset):
    """
    `torch.utils.data.Dataset` subclass for the NArmSpiral dataset

    `NArmSpiral` can be used to provide additional control over simply loading the .csv file
    directly. This class can be pass to a `torch.utils.data.DataLoader` object to iterate
    through the dataset. It also automatically separate the dataset into two parts: the test
    dataset and the train dataset. The train dataset takes 80% of the points for itself while
    the test dataset uses the last 20%.

    Parameters
    ----------
    filename: str
        .csv file containing the dataset.
    train: bool
        Specify of the dataset should contain training data or test data.

    Attributes
    ----------
    classes : list
        List of classes inside the dataset. Classes start at 0.
    data: ndarray
        Contains the points


    """
    def __init__(self, filename, train=True):
        self._file_data = np.loadtxt(filename, delimiter=';', dtype=np.float32)

        # Empty array to store the splices for training or test
        self.data = np.empty((0, self._file_data.shape[-1]), dtype=np.float32)

        # List of classes name and count for individual classes
        self.classes, _samples = np.unique(self._file_data[:, 2], return_counts=True)

        # We assume the classes have the same amount of samples
        self._sample_count = _samples[0]
        
        # Split the file data into array of each classe
        split_classes = np.split(self._file_data, len(self.classes))

        # Divide the classes into 80% training samples and 20% test samples
        part = int(self._sample_count * 0.8)

        for _class in split_classes:
            if train:
                self.data = np.concatenate((self.data, _class[:part, :]))
            else:
                self.data = np.concatenate((self.data, _class[part:, :]))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :2], self.data[index, 2].astype(np.int64)
