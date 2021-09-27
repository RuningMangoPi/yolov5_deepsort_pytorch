from detect_sort import DetectSort


if __name__ == "__main__":

    # 检测和跟踪同时做
    det_sort = DetectSort(source="./test2.avi", weight="./yolov5/weights/yolov5_chefhat_detection.pth",
                          reid_ckpt="./deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", store_file="save.json",
                          show_img=True)
    dicts = {}
    for index, (det, sort, class_name) in enumerate(det_sort):
        print(det, sort, class_name)
        for item, (x1, y1, x2, y2, id) in enumerate(sort):
            if len(dicts) == 0:
                dicts[id] = []
            if (id in dicts.keys()) is False:
                dicts[id] = []
            if class_name[item] == "chef_hat":
                dicts[id].append(1)
            else:
                dicts[id].append(0)
        for key in dicts.keys():
            if len(dicts[key]) >= 20 and sum(dicts[key][:-10]) <= 2:
                # print(dicts[key])
                print("without chef hat !!!!", key)
                dicts[key] = dicts[key][-20:]
                break

