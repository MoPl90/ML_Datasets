import torch
from sklearn.preprocessing import StandardScaler


class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, batch_size=32, scale_data=True):
        if not torch.is_tensor(X):
            # Apply scaling if necessary
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)

        else:
            self.X = X

        if torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

        self.batch_size = batch_size

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def dataloader(self, shuffle=True):

        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )
