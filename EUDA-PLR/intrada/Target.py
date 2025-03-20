import numpy as np
from utils import project_root
from dataset.base_dataset import BaseDataset

DEFAULT_INFO_PATH = project_root / 'advent/dataset/target_list/info.json'

class TargetDataSet(BaseDataset):
    def __init__(self, root, list_path, set='train',
                 max_iters=None,
                 crop_size=(512, 512), mean=(128, 128, 128),
                 load_labels=True, labels_size=None):
        # pdb.set_trace()
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)

        self.load_labels = load_labels

    def get_metadata(self, name):
        img_file = self.root / 'images' / self.set / name
        label_file = self.root / 'labels' / self.set / name
        return img_file, label_file

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        label = self.get_pseudo_labels(label_file)
        image = self.get_image(img_file)
        image = self.preprocess(image)
        return image.copy(), label, np.array(image.shape), name
