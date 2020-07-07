from torch.utils.data.dataset import Dataset
"""
Wraps HRI data around a PyTorch Dataset
"""
class HRIDataset(Dataset):
    """
    Creates a HRI dataset from HRI data
    """

    def __init__(self, data, labels):
        """
        :param data: an array of the shape (batch, seq_len, nb_feat)
        :param labels: array of labels corresponding to sequences in data - SED (1), no SED (0)
        """
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        assert index < self.__len__(), "Error: index out of range"
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)