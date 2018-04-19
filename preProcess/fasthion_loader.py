from mxnet.gluon.data import dataset
from mxnet import image, nd
import csv, random, json
from base.params import *
import matplotlib.pyplot as plt
import numpy as np


class loader(dataset.Dataset):
    def __init__(self, path, json_path, transform=None):
        super(loader, self).__init__()
        self.read_csv(path % 'Annotations/label.csv')
        self.path = path
        self.json = json_path
        self.transform = transform

    def __getitem__(self, idx):
        # return:img, class weight, attribute values
        img_path, label, attr = self.items[idx]
        img = self.read_img(img_path)
        gt_labels, transf_matrix = self.read_label(label, attr)
        self.read_json(img_path)
        keypoint = self.build_canvas()
        # self.show_img(img, keypoint)
        return img, self.cls_wight, gt_labels, transf_matrix, keypoint

    def __len__(self):
        return len(self.items)

    def read_csv(self, path):
        # data list
        with open(path, 'r')as f:
            anno = csv.reader(f)
            self.items = list(anno)
        # ground truth label
        self.cls = dict(zip(classes, nums_attres))

    def read_img(self, img_path):
        img_path = self.path % img_path
        img = image.imread(img_path)
        self.img_size = img.shape
        img = image.imresize(img, img_width, img_height)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def read_label(self, label, attr):
        # wight
        self.cls_wight = [0] * len(classes)
        self.cls_index = classes.index(label)
        self.cls_wight[self.cls_index] = 1

        # vaule or gt,the first one is index of 'y'
        if 'm' not in attr:
            b = {'n': 0.0, 'y': 1.0}
            self.values = [b[i] for i in attr]
        else:
            a = 0.2  # random.uniform(0.5, 0.8)
            c = 0.8  # random.uniform(a + 0.1, 1.0)
            b = {'n': .0, 'y': c, 'm': a}
            self.values = [b[i] for i in attr]
        self.values.insert(0, attr.index('y'))
        # prepare labels
        gt_labels, transf_matrix = self.prepare_lables(self.cls_wight, self.values)
        return gt_labels, transf_matrix

    def read_json(self, img_path):
        path = img_path.split('/')
        path[1] += '_json'
        path[2] = path[2][:-4] + '_keypoints.json'
        path = '/'.join(path)
        json_path = self.json % path
        keypoint = []
        with open(json_path, 'r')as f:
            load_dict = json.load(f)
            for pose in load_dict['people']:
                keypoint.append(pose['pose_keypoints_2d'])
        new_keypoints = self.prepare_json_point(keypoint)
        self.hint = self.prepare_hints_by_point(new_keypoints, self.cls_index)

    def prepare_lables(self, weight, label):
        """
        :param gt_:global ground truth for all branches
        :param projected:transform matrix for that turn output of net to input of softmax
        :return:
        """
        gt_ = [np.zeros(n, dtype=np.bool) for n in nums_attres]
        projected = {}
        # prepare attributes
        cur_idx_branch, y_idx_gt = weight.index(1), label[0]
        cur_nums_attres = nums_attres[cur_idx_branch]
        gt_[cur_idx_branch] = (np.array(label)[1:] >= 0.8)
        # prepare transfer & the input of softmax
        # 1)visible attributes; 2)invisible attribute
        transfer_attr = np.eye(cur_nums_attres, cur_nums_attres - 1, -1)  # -1 means diagonal lower by 1
        transfer_vis = np.eye(cur_nums_attres, 2)  # all prediction be projected to 2 numbers
        transfer_vis[1:, 1] = 1.0 / (cur_nums_attres - 1)  # enhance the difference of visble & invisible
        if y_idx_gt > 0:  # in case for 'm'
            transfer_attr[1:, y_idx_gt - 1] = label[2:]
        else:
            transfer_vis[:, 0] = label[1:]  # protect the maybe attributes
            gt_[cur_idx_branch][1:] = (np.array(label)[2:] >= 0.2)
        projected['vis'] = transfer_vis
        projected['attr'] = transfer_attr
        projected['branch'] = cur_idx_branch
        abnormal_ = np.array([np.sum(x) for x in gt_])
        abnormal = np.sum(abnormal_)
        if abnormal > 1 or abnormal == 0:
            print '--abnormal in fashion_loader--'
        return gt_, projected

    def prepare_json_point(self, keypoint):
        h, w, c = self.img_size
        # h_ratio, w_ratio = 1.0 * h / img_height, 1.0 * w / img_width
        new_keypoints = dict([[i, []] for i in range(num_keypoints / 3)])
        for key in keypoint:
            for i in range(num_keypoints / 3):
                x, y, c = key[3 * i:3 * i + 3]
                if x + y <= 1:
                    continue
                # line column
                a, b = 1.0 * y / h * img_height, 1.0 * x / w * img_width
                a, b = int(round(a / stride_all)), int(round(b / stride_all))
                new_keypoints[i].append([a, b])
        # maybe can not remove all multi parts
        # check mirror parts
        for i in range(num_keypoints / 3):
            # more than 1 candidante
            if len(new_keypoints[i]) > 1:
                cur_mirror = [kp[kp.index(i) ^ 1] for kp in mirrors if i in kp]
                if cur_mirror:
                    new_keypoints[cur_mirror[0]].append(new_keypoints[i].pop())
        return new_keypoints

    def prepare_hints_by_point(self, keypoints, cls_index):
        pc = point_connection[cls_index]
        hint = []
        for i in range(len(pc) / 2):  # virtual for point pairs
            s, e = pc[2 * i:2 * i + 2]
            start, end = keypoints[s][:], keypoints[e][:]
            if not start or not end:
                continue
            s1, e1 = start.pop(0), end.pop(0)
            while True:
                res = self.draw_line(s1, e1)
                hint += zip(*res)
                hint = list(set(hint))
                if start:
                    s1 = start.pop(0)
                if end:
                    e1 = end.pop(0)
                if not start and not end:
                    break
        return nd.array(hint).T

    # all points should be and must be within the range
    def draw_line(self, s, e):
        dir = 1
        lu_l, rd_l, dir = (s[0], e[0], dir) if s[0] <= e[0] else (e[0], s[0], -dir)
        lu_c, rd_c, dir = (s[1], e[1], dir) if s[1] <= e[1] else (e[1], s[1], -dir)
        h, w = rd_l - lu_l + 1, rd_c - lu_c + 1

        mimic_mat = nd.zeros((h, w))
        if h == 1 and w == 1:
            return s, e
        if 1.0 * h / w <= 1:
            mimic_w = nd.tile(nd.arange(1, w), (h, 1))
            if dir == 1:
                mimic_h = nd.tile(nd.arange(0, h), (w - 1, 1)).T  # notice the direction & vertical
            else:
                mimic_h = nd.tile(nd.arange(h - 1, -1, -1), (w - 1, 1)).T
            start = nd.array([[h - 1]]) if dir == -1 else nd.array([[0]])
            out = nd.abs(mimic_h / mimic_w - 1.0 * h / w)
            loc_h = nd.topk(-out, axis=0)
            loc_h = nd.concat(start, loc_h, dim=1)
            a = nd.arange(w)
            mimic_mat[loc_h, a] = 1
            res_h = (loc_h + lu_l).asnumpy().astype(np.uint8).tolist()[0]
            res_w = list(range(lu_c, lu_c + w))
            return res_h, res_w
        else:
            mimic_h = nd.tile(nd.arange(1, h), (w, 1)).T
            if dir == 1:
                mimic_w = nd.tile(nd.arange(0, w), (h - 1, 1))
            else:
                mimic_w = nd.tile(nd.arange(w - 1, -1, -1), (h - 1, 1))
            start = nd.array([[w - 1]]) if dir == -1 else nd.array([[0]])
            out = nd.abs(mimic_h / mimic_w - 1.0 * h / w)
            loc_h = nd.topk(-out, axis=1).T
            loc_h = nd.concat(start, loc_h, dim=1)
            a = nd.arange(h)
            mimic_mat[a, loc_h] = 1
            res_h = list(range(lu_l, lu_l + h))
            res_w = (loc_h + lu_c).asnumpy().astype(np.uint8).tolist()[0]
            return res_h, res_w

    def build_canvas(self):
        temp = nd.zeros((int(img_height / stride_all), int(img_width / stride_all)))
        end = int(self.hint.shape[-1] * proportion_hint)
        if self.hint.size:
            keypoint = self.hint[:, :end]  # .asnumpy().astype(np.uint8)
            temp[keypoint[0], keypoint[1]] = 5.0
        return temp

    def show_img(self, img, map):
        print(classes[self.cls_index])
        plt.figure(1)
        plt.imshow(img.asnumpy()[0])
        plt.figure(2)
        # temp = np.zeros((int(img_height / stride_all), int(img_width / stride_all)))
        # keypoint = self.hint.asnumpy().astype(np.uint8)
        # temp[keypoint[0], keypoint[1]] = 1
        temp = map.asnumpy()
        plt.imshow(temp)
        plt.show()


if __name__ == "__main__":
    img_path = '/home1/%s'
    input = img_path % 'FashionAI/base/%s'
    hint_gt = img_path % 'FashionMark/base/%s'
    f_d = loader(input, hint_gt)
    idx = random.randint(0, len(f_d))
    datas = f_d[idx]

    plt.imshow(datas[0].asnumpy())
    print datas[1:]
    plt.show()
