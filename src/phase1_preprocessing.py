import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.phase1_config import IMAGE_SIZE, BATCH_SIZE


class SignLanguageDataset(Dataset):
    def __init__(self, dataframe, label_to_index=None, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

        if label_to_index is None:
            labels = sorted(self.dataframe["label"].unique())
            self.label_to_index = {label: index for index, label in enumerate(labels)}
        else:
            self.label_to_index = label_to_index

        self.index_to_label = {index: label for label, index in self.label_to_index.items()}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        label = self.label_to_index[row["label"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_phase1_transforms(image_size=IMAGE_SIZE):
    base_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return base_transform


def build_dataloaders(train_df, val_df, test_df, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    label_to_index = {
        label: index for index, label in enumerate(sorted(train_df["label"].unique()))
    }

    transform = get_phase1_transforms(image_size=image_size)

    train_dataset = SignLanguageDataset(
        train_df,
        label_to_index=label_to_index,
        transform=transform,
    )
    val_dataset = SignLanguageDataset(
        val_df,
        label_to_index=label_to_index,
        transform=transform,
    )
    test_dataset = SignLanguageDataset(
        test_df,
        label_to_index=label_to_index,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader, label_to_index
