import os
import sys
import numpy as np
import cv2
import torch
import torch.utils.data as data
from util.config import train_cfg
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from data.preproc import preproc
TEST = 1


def val_data(root):
    imgs = []
    targets = []
    CARD_CLASSES = get_anno()
    class_to_ind = dict(
        zip(CARD_CLASSES, range(len(CARD_CLASSES))))
    lists = os.listdir(root)
    for list in lists:
        label = class_to_ind[list]
        img_list_path = os.path.join(root, list)
        img_lists = os.listdir(img_list_path)

        for img_list in img_lists:
            img_label = []
            img_path = img_list_path + "/" + img_list
            imgs.append(img_path)
            img_label.append(int(label))
            targets.append(img_label)

    return imgs, targets


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    def __init__(self, class_to_ind=None):
        CARD_CLASSES = get_anno()
        self.class_to_ind = class_to_ind or dict(
            zip(CARD_CLASSES, range(len(CARD_CLASSES))))

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        bndbox = []
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)

        return bndbox


def get_anno():
    labels = []
    data_root = train_cfg['data_set_dir']
    label_anno = data_root + '/labels.txt'
    with open(label_anno, 'r') as f:
        results = f.readlines()

    for result in results:
        label = result.strip().split(':')
        labels.append(label[1])

    return labels


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to WIDER folder
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """
    def __init__(self, root, dataset_type, preproc=None, target_transform=None):
        self.root = root
        self.dataset_type = dataset_type
        self.preproc = preproc
        if dataset_type == 'train':
            self.target_transform = target_transform
            self._annopath = os.path.join(self.root, 'Annotations', '%s')
            self._imgpath = os.path.join(self.root, 'JPEGImages', '%s')
            self.ids = os.listdir(os.path.join(self.root, 'Annotations'))
        elif dataset_type == 'val':
            self.target_transform = None
            self._imgpath, self._annopath = val_data(self.root + "/val/")

    def __getitem__(self, index):
        if self.dataset_type == 'train':
            img_id = self.ids[index][:-4] + '.jpg'
            target_id = self.ids[index]
            target = ET.parse(self._annopath % target_id).getroot()
            img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        elif self.dataset_type == 'val':
            img = cv2.imread(self._imgpath[index], cv2.IMREAD_COLOR)
            target = self._annopath[index]
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.preproc is not None:
            img, target = self.preproc(img, target)
        return torch.from_numpy(img), target

    def __len__(self):
        if self.dataset_type == 'train':
            return len(self.ids)
        elif self.dataset_type == 'val':
            return (len(self._imgpath))


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup[0], int):
                # annos = torch.Tensor(tup)
                targets.append(tup[0])

    imgs = torch.stack(imgs, 0)
    targets = np.array(targets)
    targets = torch.Tensor(targets)

    return imgs, targets


if TEST == 0:
    dataset_path = train_cfg['data_set_dir']
    dataset = VOCDetection(dataset_path, preproc(224, (104, 117, 123)), AnnotationTransform())
    print(dataset.__len__())

    for iteration in range(0, dataset.__len__()//8):
        batch_iterator = iter(data.DataLoader(dataset, 8, shuffle=True, num_workers=4, collate_fn=detection_collate))
        images, labels = next(batch_iterator)
        print(images, labels)
