import numpy as np
from torch.utils.data import Dataset

class NArmSpiral(Dataset):

    def __init__(self, filename, train=True):
        self.file_data = np.loadtxt(filename, delimiter=';', dtype=np.float32)

        # Empty array to store the splices for training or test
        self.data = np.empty((0, self.file_data.shape[-1]), dtype=np.float32)

        # List of classes name and count for individual classes
        self.classes, _samples = np.unique(self.file_data[:, 2], return_counts=True)

        # We assume the classes have the same amount of samples
        self._sample_count = _samples[0]
        
        # Split the file data into array of each classe
        split_classes = np.split(self.file_data, len(self.classes))

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
