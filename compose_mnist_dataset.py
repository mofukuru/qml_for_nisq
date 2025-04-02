import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

class ComposeMnistDataset:
    def __init__(
            self,
            num_trainset: int | None = None,
            num_testset: int | None = None,
            train_batch_size: int = 64,
            test_batch_size: int = 64,
            num_comp: int = 0,
            visualize: bool = False
        ):

        self.num_trainset = num_trainset
        self.num_testset = num_testset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.visualize = visualize
        self.num_comp = num_comp

    def reduce_size(self, size):
        return transforms.Resize(size=size)

    def filter_classes(self, dataset, classes=[0, 1]):
        indices = np.where(np.isin(dataset.targets, classes))[0]
        dataset.targets = dataset.targets[indices]
        dataset.data = dataset.data[indices]
        return dataset

    def compress_by_pca(self, train_dataset, test_dataset):
        pca = PCA(n_components=self.num_comp)

        train_data_pca = pca.fit_transform(train_dataset.data)
        test_data_pca = pca.transform(test_dataset.data)

        train_data_reconstruction = pca.inverse_transform(train_data_pca)
        test_data_reconstruction = pca.inverse_transform(test_data_pca)

        train_data_pca = torch.tensor(train_data_pca, dtype=torch.float32)
        test_data_pca = torch.tensor(test_data_pca, dtype=torch.float32)

        self.train_data_pca = train_data_pca
        self.test_data_pca = test_data_pca
        self.train_labels = train_dataset.targets
        self.test_labels = test_dataset.targets

        if self.visualize and self.num_comp == 2:
            m = self.plot_pca_features()
            m.show()

        return train_data_pca, test_data_pca, train_data_reconstruction, test_data_reconstruction

    def print_dataset_info(self, train_dataset, test_dataset):
        print("==== dataset infomation ====")
        print(f"# of train datasets: {len(train_dataset)}")
        print(f"# of test datasets: {len(test_dataset)}")

        sample_data, sample_label = train_dataset[0]
        print(f"shape of data: {sample_data.shape}")
        print(f"data type: {sample_data.dtype}")

        unique_classes, counts = train_dataset.targets.clone().detach().unique(return_counts=True)
        print("class distribution(train data): ")
        for cls, count in zip(unique_classes, counts):
            print(f"    class {cls.item()}: {count.item()}")

        unique_classes, counts = test_dataset.targets.clone().detach().unique(return_counts=True)
        print("class distribution(test data): ")
        for cls, count in zip(unique_classes, counts):
            print(f"    class {cls.item()}: {count.item()}")

    def plot_pca_features(self):
        markers = ['o', 's', '^', 'D', 'x', 'P', '*', '+', '|', '_']
        plt.figure(figsize=(8, 6))

        unique_labels = torch.unique(self.train_labels)
        for i, label in enumerate(unique_labels):
            mask_train = self.train_labels == label
            mask_test = self.test_labels == label
            plt.scatter(
                self.train_data_pca[mask_train, 0],
                self.train_data_pca[mask_train, 1],
                label=f"Train {label.item()}",
                marker=markers[i % len(markers)],
                alpha=0.5
                )
            plt.scatter(
                self.test_data_pca[mask_test, 0],
                self.test_data_pca[mask_test, 1],
                label=f"Test {label.item()}",
                marker=markers[i % len(markers)],
                alpha=0.5,
                edgecolors='black'
                )

        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.title("PCA Feature Distribution by Label")
        return plt

    def show_sample_images(self, num_comp, dataset, original_dataset, num_samples=5):
        fig, axes = plt.subplots(num_samples, 2, figsize=(5, num_samples * 2.5))
        for i in range(num_samples):
            if num_comp == 0:
                image, label = dataset[i]
                if isinstance(image, torch.Tensor):
                    image = image.squeeze().numpy()
                axes[i, 0].imshow(image, cmap="gray")
                axes[i, 0].set_title(f"Label: {label}")
                axes[i, 0].axis("off")

                axes[i, 1].axis("off")
            else:
                image = dataset[i]
                original_image, original_label = original_dataset[i]
                original_image = original_image.reshape(28, 28)
                image = image.reshape(28, 28)
                if isinstance(original_image, torch.Tensor):
                    original_image = original_image.squeeze().numpy()
                if isinstance(image, torch.Tensor):
                    image = image.squeeze().numpy()
                axes[i, 0].imshow(original_image, cmap="gray")
                axes[i, 0].set_title(f"Original (Label: {original_label})")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(image, cmap="gray")
                axes[i, 1].set_title("Reconstructed")
                axes[i, 1].axis("off")
        plt.tight_layout()
        plt.show()

    def exec(self, filter: list | None, size: tuple):
        transform_list = []

        transform_list.append(transforms.ToTensor())

        if self.num_comp != 0:
            transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
        else:
            # TODO: revise the params of normalization
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
            transform_list.append(self.reduce_size(size))

        transform = transforms.Compose(transform_list)

        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

        if filter is not None:
            train_dataset = self.filter_classes(train_dataset, filter)
            test_dataset = self.filter_classes(test_dataset, filter)

        original_train_dataset = copy.deepcopy(train_dataset)
        # original_test_dataset = copy.deepcopy(test_dataset)

        if self.num_comp != 0:
            train_dataset.data = train_dataset.data.view(-1, 28*28).float() / 255.0
            test_dataset.data = test_dataset.data.view(-1, 28*28).float() / 255.0

            train_dataset.data, test_dataset.data, train_reconstruction, test_reconstruction = \
                self.compress_by_pca(train_dataset, test_dataset)

        if self.num_trainset is not None:
            self.num_trainset = min(self.num_trainset, len(train_dataset.data))
            train_dataset.data = train_dataset.data[:self.num_trainset]
            train_dataset.targets = train_dataset.targets[:self.num_trainset]

        if self.num_testset is not None:
            self.num_testset = min(self.num_testset, len(test_dataset.data))
            test_dataset.data = test_dataset.data[:self.num_testset]
            test_dataset.targets = test_dataset.targets[:self.num_testset]

        if self.visualize:
            self.print_dataset_info(train_dataset, test_dataset)

            if self.num_comp == 0:
                self.show_sample_images(self.num_comp, train_dataset, None, num_samples=5)
            else:
                self.show_sample_images(self.num_comp, train_reconstruction, original_train_dataset, num_samples=5)

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(
                self.train_batch_size,
                self.num_trainset if self.num_trainset is not None else len(train_dataset.data)
                ),
            shuffle=True
            )
        test_loader = DataLoader(
            test_dataset,
            batch_size=min(
                self.test_batch_size,
                self.num_testset if self.num_testset is not None else len(test_dataset.data)
                ),
            shuffle=True
            )

        return train_loader, test_loader
