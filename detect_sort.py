
# 1. 加载检测好的结果
# 2. 检测新的数据

import sys
sys.path.insert(0, './yolov5')
import cv2
import json
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import non_max_suppression, scale_coords
from deep_sort_pytorch.deep_sort import DeepSort
import numpy as np
from resnet18_inference import resnet_inference
import time


def save_image(box_list, image):
    for index, box in enumerate(box_list):
        crop_mat = image[box[1]: box[3], box[0]: box[2]]
        cur_time = (int(round(time.time() * 1000)))
        cv2.imwrite("C:/Users/DeepBlue/Desktop/copmat/" + str(cur_time) + ".jpg", crop_mat)




class Detect(object):
    """
    用来获得检测的结果
    Args:
        weights 是None, 则不进行目标检测，直接加载data中的数据
    """
    def __init__(self, weight=None, gpus='', data=None, conf_thres=0.4, iou_thres=0.5, classes=[0], img_size=640,
                 save=False, start_frame=0):

        if weight is None and data is None:
            print("weight 和 data 至少有一个不为空")
        self.data_only = False

        if weight is None:
            self.data_only = True
            self.pred_data = torch.load(data)
        else:
            self.device = select_device(gpus)
            self.model = attempt_load(weight, map_location=self.device)  # load FP32 model
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
            if self.half:
                self.model.half()  # to FP16

            self.augment = False
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.classes = classes
            self.agnostic_nms = False
            self.img_size = img_size
            self.chefhat_class_net = resnet_inference()  # 厨师帽分类

            # self.dataset = LoadImages(data, img_size=self.img_size)
            # self.dataset.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def __call__(self, img, im0s):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = self.model(img, augment=self.augment)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        for item_index, item in enumerate(pred):
            pred[item_index][:, :4] = scale_coords(
                img.shape[2:], item[:, :4], im0s.shape).round()
        return pred, im0s

class Sort(object):
    r"""
    获得跟踪的结果
    :prediction 不是None,直接加载prediction中的数据，不进行跟踪操作
    :sort_data 不是None, 直接加载sort_data 的数据，不再进行计算

    """
    def __init__(self, prediction=None,  reid_ckpt=None, max_dist=0.2, source=None,
                 min_confidence=0.3, nms_max_overlap=0.5, max_iou_distanve=0.7, max_age=70, n_init=3, nn_budget=100,
                 save=False):
        self.prediction = prediction
        # deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
        self.reid_ckpt = reid_ckpt
        self.max_dist = max_dist
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.max_iou_distanve = max_iou_distanve
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.save = save
        self.deepsort = DeepSort(reid_ckpt, max_dist=max_dist, min_confidence=min_confidence,
                                 nms_max_overlap=nms_max_overlap, max_iou_distance=max_iou_distanve, max_age=max_age,
                                 n_init=n_init, nn_budget=nn_budget, use_cuda=True)

    def bbox_rel(self, *xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

    def compute_color_for_labels(self, label):
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)
    #
    # def draw_boxes(self, img, bbox, identities=None, offset=(0, 0)):
    #     for i, box in enumerate(bbox):
    #         x1, y1, x2, y2 = [int(i) for i in box]
    #         x1 += offset[0]
    #         x2 += offset[0]
    #         y1 += offset[1]
    #         y2 += offset[1]
    #         # box text and bar
    #         id = int(identities[i]) if identities is not None else 0
    #         color = self.compute_color_for_labels(id)
    #         label = '{}{:d}'.format("", id)
    #         t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    #         cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    #         cv2.rectangle(
    #             img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
    #         cv2.putText(img, label, (x1, y1 +
    #                                  t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    #     return img

    def __call__(self, det_result):
        det, img = det_result[0], det_result[1]
        outputs = []
        for i, pred in enumerate(det):  # detections per image
            if pred is not None and len(pred):
                bbox_xywh = []
                confs = []
                bbox_labels = []
                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in pred:
                    x_c, y_c, bbox_w, bbox_h = self.bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    bbox_labels.append(cls.item())

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = self.deepsort.update(xywhs, confss, img)
                # if len(outputs) > 0:
                #     bbox_xyxy = outputs[:, :4]
                #     identities = outputs[:, -1]
                #     self.draw_boxes(img, bbox_xyxy, identities)
            else:
                self.deepsort.increment_ages()
        return np.array(outputs), img


class DetectSort(object):
    r"""
    :use_store 使用检测和跟踪的保存文件
    :store_file 检测跟踪结果保存文件
        文件格式：{ "帧号": {"检测结果":[ [检测得分, 标签] ], "跟踪结果"：[[xmin, ymin, xmax, ymax, id]] }}
        如：{ "57": {"det": [[0.9482421875, 1.0]], "sort": [[822, 751, 1041, 1079, 1]]} }
    :weight 检测的模型文件
    :reid_ckpt 跟踪的模型文件
    :save 是否保存结果，结果保存在 store_file 中
    :classes 设置检测跟踪哪些类别目标，如[0]， [1]， [0， 1]

    """
    def __init__(self, use_store=False, store_file=None, source=None, weight=None, reid_ckpt=None, save=False,
                 start_frame=0, classes=[0], show_img=False):
        self.frame_index = start_frame
        self.use_store = use_store
        self.store_file = store_file
        self.save = save
        self.show_img = show_img

        if not self.use_store is None and store_file is None:
            raise Exception("use_store 为真时，store_file 不能为None")

        if self.use_store is None and source is None:
            raise Exception("use_store, source 至少有一个设置")

        if self.use_store:
            self.store_data = json.load(open(store_file))
        else:
            self.store_data = {}
            self.detect = Detect(weight=weight, data=source, classes=classes, save=True, start_frame=start_frame)
            self.sort = Sort(reid_ckpt=reid_ckpt, save=True)
            self.chefhat_class_net = resnet_inference()  # 厨师帽分类

        img_size = 640
        if not source is None:
            self.dataset = LoadImages(source, img_size=img_size)
            self.dataset.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)



    def compute_color_for_labels(self, label):
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)

    def draw_boxes(self, img, bbox, class_names, identities=None, offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0

            color = self.compute_color_for_labels(id)
            # names = class_names
            label = '{}, {}'.format(id, class_names[i])
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 +
                                     t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        return img

    def __iter__(self):
        return self


    def __next__(self):
        r"""
        :return: 检测结果和跟踪结果
        格式：[[检测得分，目标标签], [xmin, ymin, xmax, ymax, 跟踪id]]
        示例：[[0.95166015625, 1.0], [100, 100, 200, 200, 1]]
        """
        img = None
        im0s = None
        if hasattr(self, "dataset"):
            for frame_idx, (path, img, im0s, vid_cap) in enumerate(self.dataset):
                break

        if not self.use_store:
            key = f"{self.frame_index}"
            det_retsult = self.detect(img, im0s)
            sort_result = self.sort(det_retsult)
            ret_det = det_retsult[0][0][:, 4:].cpu().numpy().tolist()
            ret_sort = sort_result[0].tolist()
            # save_image(ret_sort, im0s)
            ret_class = self.chefhat_class_net.inference(ret_sort, im0s)

            self.store_data[key] = {
                "det": ret_det,
                "sort": ret_sort,
                "class": ret_class
            }
            if self.save:
                json.dump(self.store_data, open(self.store_file, "w"))

            self.frame_index += 1

            if self.show_img:
                if len(sort_result[0]) > 0:
                    bbox_xyxy = sort_result[0][:, :4]
                    identities = sort_result[0][:, -1]

                    self.draw_boxes(im0s, bbox_xyxy, ret_class, identities)
                cv2.imshow("im0s", im0s)
                cv2.imwrite("C:/Users/DeepBlue/Desktop/imgs/" + str(self.frame_index) + ".jpg", im0s)
                cv2.waitKey(1)

            return ret_det, ret_sort, ret_class

        else:
            key = f"{self.frame_index}"
            if not key in list(self.store_data.keys()):
                raise StopIteration
            ret = self.store_data[f"{self.frame_index}"]
            self.frame_index += 1

            if self.show_img:
                if len(ret["sort"]) > 0:
                    bbox_xyxy = np.array(ret["sort"])[:, :4]
                    identities = np.array(ret["sort"])[:, -1]
                    self.draw_boxes(im0s, bbox_xyxy, identities)
                cv2.imshow("im0s", im0s)
                cv2.waitKey()

            return ret["det"], ret["sort"],

    def __getitem__(self, item):
        r"""
        :item 帧号
        :return: 检测结果和跟踪结果
        格式：[[检测得分，目标标签], [xmin, ymin, xmax, ymax, 跟踪id]]
        示例：[[0.95166015625, 1.0], [100, 100, 200, 200, 1]]
        """
        key = f"{item}"
        if not key in list(self.store_data.keys()):
            raise StopIteration
        ret = self.store_data[f"{self.frame_index}"]
        self.frame_index += 1
        return ret["det"], ret["sort"]


if __name__ == "__main__":
    #
    # # 加载检测和跟踪的结果
    # det_sort = DetectSort(use_store=True, store_file="save.json", source="test2_Trim.mp4", show_img=True)
    #
    # # 获得某一帧的结果
    # print(det_sort[0])
    #
    # # 迭代所有的结果
    # for index, (det, sort) in enumerate(det_sort):
    #     print(det, sort)

    # 检测和跟踪同时做
    det_sort = DetectSort(source="test2_Trim.mp4", weight="yolov5/weights/wy_person.pth",
                          reid_ckpt="deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", store_file="save.json", show_img=True)
    for index, (det, sort, class_name) in enumerate(det_sort):
        print(det, sort, class_name)

