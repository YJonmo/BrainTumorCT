import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class BrainTumorDataset(Dataset):
    def __init__(self, images, labels, no_classes=4, image_size=128):
        # Images
        self.X = images
        # Corresponding Labels
        self.Y = labels

        # Convert original image numpy array to PIL image and then to a tensor
        self.transform = T.Compose([T.ToPILImage(),
                                             T.Resize((image_size, image_size)),
                                             T.ToTensor()
                                             ])
        self.no_classes = no_classes

    def __len__(self):
        # Returns # of images
        return len(self.X)

    def __getitem__(self, idx):
        # Transformations for one image of X at a time
        # Original image as a tensor
        img = self.transform(self.X[idx])

        labels = torch.zeros(self.no_classes, dtype=torch.float32)

        for i in range(self.no_classes):
          labels[i] = self.Y[idx][i]

        labels= [labels]
        img = [img]
        # print(np.shape(img[0]))
        # print(labels)
        return (torch.stack(labels), torch.stack(img))
