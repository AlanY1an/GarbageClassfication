import os
import random
import shutil


class DataPartitioner:
    def __init__(self, dataset_path, output_path="data_split", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """初始化数据划分器
        :param dataset_path: 原始数据集路径
        :param output_path: 输出的划分数据集存放路径
        :param train_ratio: 训练集比例
        :param val_ratio: 验证集比例
        :param test_ratio: 测试集比例
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Create new directories for partitioned data
        self.train_path = os.path.join(self.output_path, "train")
        self.val_path = os.path.join(self.output_path, "val")
        self.test_path = os.path.join(self.output_path, "test")
        self.create_dirs([self.train_path, self.val_path, self.test_path])

    def create_dirs(self, directories):
        """Create directories if they do not exist."""
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def get_classes(self):
        """
        Retrieve class names from the dataset.
        :return: A list of class names (subdirectories).
        """
        return [cls for cls in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, cls))]

    def partition_data(self):
        """Partition the dataset into training, validation, and test sets."""
        classes = self.get_classes()

        for cls in classes:
            cls_path = os.path.join(self.dataset_path, cls)
            images = os.listdir(cls_path)
            # Randomly shuffle before split
            random.shuffle(images)

            # Compute partition indices
            total = len(images)
            train_end = int(total * self.train_ratio)
            val_end = train_end + int(total * self.val_ratio)

            train_images = images[:train_end]
            val_images = images[train_end:val_end]
            test_images = images[val_end:]

            # Copy images to respective directories
            self.copy_images(train_images, cls_path, os.path.join(self.train_path, cls))
            self.copy_images(val_images, cls_path, os.path.join(self.val_path, cls))
            self.copy_images(test_images, cls_path, os.path.join(self.test_path, cls))

            print(f"Class [{cls}] done: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

        print("All processing done!")

    def copy_images(self, images, src_dir, dst_dir):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for img in images:
            shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))


if __name__ == "__main__":
    dataset_path = "../data/raw-data" # Path to your original dataset
    output_path = "../data/split-data" # Path to the new partitioned dataset

    partitioner = DataPartitioner(dataset_path, output_path)
    partitioner.partition_data()
