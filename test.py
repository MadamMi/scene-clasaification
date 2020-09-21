import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from peleenet import load_model
import numpy as np
import shutil
import sys


root = '/media/D3/scene_meetvr/find_imgs'
new_root = '/media/D3/scene_meetvr/new_data'
if not os.path.exists(new_root):
    os.mkdir(new_root)

img_lists = os.listdir(root)


def main(id):
    save_root = os.path.join(new_root, str(id))
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    count = 0
    for img_list in img_lists:
        img_path = os.path.join(root, img_list)
        image = Image.open(img_path)
        image = image.resize((224, 224))
        centre_crop = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = centre_crop(image).unsqueeze(0)
        #if torch.cuda.is_available():
        img = img.cuda()
        img = Variable(img)

        model = load_model(pretrained_model_path='models/scene_138_acc_78_38.pth.tar',
                           model_classes=138, data_classes=138)
        model = model.cuda()
        model.eval()
        out = model(img)
        out = out.cpu()
        output = out.detach().numpy()
        result = np.argmax(output[0])
        if result == id:
            img_new_path = os.path.join(save_root, str('%06d' % count) + '.jpg')
            shutil.copyfile(img_path, img_new_path)
            count += 1

        print('classID = {0}, score after softmax = {1}'.format(np.argmax(output[0]), output[0][np.argmax(output[0])]))


if __name__ == '__main__':
    id = 8
    print(id)
    main(id)

