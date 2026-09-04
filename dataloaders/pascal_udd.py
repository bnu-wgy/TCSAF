from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 base_dir=None,
                 split='train',
                 transform=None
                 ):

        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line)
                _cat = os.path.join(self._cat_dir, line.replace('JPG', 'png'))
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat.replace('JPG', 'png'))
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        _target = np.array(Image.open(self.categories[index])).astype(np.int32)
        _target = np.where(_target==1, 1, 0)


        return _img, _target

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'

