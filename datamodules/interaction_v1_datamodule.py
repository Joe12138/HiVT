from typing import Callable, Optional
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader

from datasets import InteractionDataset


class InteractionV1DataModule(LightningDataModule):
    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 local_radius: float = 50,
                 save_dir: str = "/home/joe/Desktop/HiVT/test_data") -> None:
        super(InteractionV1DataModule, self).__init__()
        self.root = root
        self.save_dir = save_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transforms = val_transform
        self.local_radius = local_radius

    def prepare_data(self) -> None:
        InteractionDataset(root=self.root,
                           save_dir=self.save_dir,
                           split="train",
                           transform=self.train_transform,
                           local_radius=self.local_radius)

        InteractionDataset(root=self.root,
                           save_dir=self.save_dir,
                           split="val",
                           transform=self.val_transforms,
                           local_radius=self.local_radius)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = InteractionDataset(root=self.root,
                                                save_dir=self.save_dir,
                                                split="train",
                                                transform=self.train_transform,
                                                local_radius=self.local_radius)

        self.val_dataset = InteractionDataset(root=self.root,
                                              save_dir=self.save_dir,
                                              split="val",
                                              transform=self.val_transforms,
                                              local_radius=self.local_radius)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)