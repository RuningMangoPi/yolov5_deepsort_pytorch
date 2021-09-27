import os
import cv2
import timm
import torch
import numpy as np
from collections import OrderedDict


class resnet_inference(object):
    def __init__(self):
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.classes_name = ["chef_hat", "other_hat", "without_hat"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = device
        self.net_name = "resnet18"
        self.model_path = "./yolov5/weights/chefhat_class_resnet18_41.pth"
        self.net = self.load_pretrain_model()

    def load_pretrain_model(self):
        # 加载网络
        net = timm.create_model(self.net_name, num_classes=len(self.classes_name), pretrained=True)
        checkpoint = torch.load(self.model_path, map_location='cuda:0')
        change = OrderedDict()
        for key, op in checkpoint.items():
            change[key.replace("module.", "", 1)] = op
        net.load_state_dict(change, strict=True)
        if "cuda" in self.device.type:
            net = net.cuda()
        return net


    def inference(self, box_list, image):
        src_img = np.int8(image)
        class_name = []
        for item, box in enumerate(box_list):
            img = src_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])].astype(np.uint8)
            image = cv2.resize(img, (112, 112))
            image = image.astype(np.float32)

            image = (image/255.0 - self.mean)/self.std
            image = image.transpose([2, 0, 1])
            if "cuda" in self.device.type:
                image = torch.from_numpy(image).to(self.device)
            else:
                image = torch.from_numpy(image)
            if image.ndimension() == 3:
                image = image.unsqueeze(0)
            # Inference
            self.net.eval()
            output = self.net(image)
            _, predict = torch.max(output.data, 1)
            class_name.append(self.classes_name[predict.item()])
        return class_name



if __name__=="__main__":
    classes_name = ["chef_hat", "other_hat", "without_hat"]
    net = resnet_inference()
    path = "C:/Users/DeepBlue/Desktop/Yolov5_DeepSort_Pytorch/images/"
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        image = cv2.imread(file_path)
        class_index = net.inference(image)
        print(classes_name[class_index])
        print("==================================")
        cv2.imshow("image", image)
        cv2.waitKey(0)


