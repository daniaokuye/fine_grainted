import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.gluon import nn
from params import *
import numpy as np
import random


class FashionNet(gluon.Block):
    def __init__(self, mode='train'):
        super(FashionNet, self).__init__()
        self.mode = mode
        # self.explore_self()
        # self.base_1, self.base_2 = \
        self.vgg_baseNet()
        self.vector = nn.Sequential()
        for i in range(len(classes)):
            self.vector.add(self.extract_vector(nums_attres[i]))

    def forward(self, *args):
        x, w, keypoint = args
        outs = []
        featues = self.base_1(x)
        by_pass_mask = self.spatical_drop_out(featues, keypoint)
        featues = self.base_2(featues)
        featues = featues * by_pass_mask
        featues = self.base_3(featues)
        for i in range(len(classes)):
            outs.append(self.vector[i](featues) * w[:, i].reshape((-1, 1)))
        return outs

    # for regression (attributes) & classification (values)
    def extract_vector(self, num_outputs):
        branch = nn.Sequential()
        branch.add(nn.Flatten())
        branch.add(nn.Dense(4096))
        branch.add(nn.BatchNorm(axis=1, center=True, scale=True))
        branch.add(nn.Activation(activation='relu'))
        branch.add(nn.Dense(4096))
        branch.add(nn.BatchNorm(axis=1, center=True, scale=True))
        branch.add(nn.Activation(activation='relu'))
        branch.add(nn.Dense(num_outputs))
        return branch

    # company 1 | VGG net as bases
    def vgg_block(self, num_convs, channels):
        out = nn.Sequential()
        for _ in range(num_convs):
            out.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1))
            out.add(nn.BatchNorm(axis=1, center=True, scale=True))
            out.add(nn.Activation(activation='relu'))
        out.add(nn.MaxPool2D(pool_size=2, strides=2))
        return out

    def vgg_stack(self, architecture):
        out = nn.Sequential()
        # with self.name_scope():
        for (num_convs, channels) in architecture:
            out.add(self.vgg_block(num_convs, channels))
        return out

    def vgg_baseNet(self):
        # num_outputs = 10
        architecture_1 = ((2, 64), (2, 128), (3, 256))
        architecture_2 = ((3, 512),)
        architecture_3 = ((3, 512),)
        # self.base_1, self.base_2 = nn.Sequential(), nn.Sequential()
        self.base_1 = self.vgg_stack(architecture_1)
        self.base_2 = self.vgg_stack(architecture_2)
        self.base_3 = self.vgg_stack(architecture_3)
        # special for bypass net
        channels = [512, 512, 256, 64, 1]
        self.by_pass = self.build_by_pass(channels)

    def build_by_pass(self, channels):
        out = nn.Sequential()
        for c in channels:
            out.add(nn.Conv2D(c, kernel_size=3, padding=1, activation='relu'))
        out.add(nn.AvgPool2D(2, 2))
        return out

    ############################
    # net -- spatical_drop_out
    #
    # drop out some details according to features activation
    # leave some activation depressed or enhanced by spatical drop out
    # or just similiar to OHEM method with random
    #
    ############################
    def spatical_drop_out(self, x, keypoints):
        if (self.mode == "train"):
            feature = self.by_pass(x)
            # 1.explore the feature; 2.random select; 3.hint
            # three all generate seeds on pooling layer
            spark = self.explore_seeds(feature)
            # b, c, h, w = feature.shape
            # new_f = nd.zeros_like(x).as_in_context(x.context)
            OHEM_mask = self.assemble_seeds_hints(spark, keypoints)
            return OHEM_mask

    def explore_seeds(self, x, proportion=0.6):
        b, _, h, w = x.shape
        k = int(h * w * proportion)
        top_k_values = nd.topk(x.flatten(), k=k, ret_typ='value')
        idx = [random.choice(range(int(k * 0.1), int(k * proportion_self)))
               for i in range(b)]
        line = list(range(b))
        threshold = top_k_values[line, idx]
        threshold = threshold.reshape((-1, 1, 1, 1))
        seeds = (x > threshold) * 5.0
        return seeds

    def assemble_seeds_hints(self, spark, key_hints):
        res = spark + key_hints
        return res.sigmoid()

    # def explore_self(self, x, proportion=0.6):
    #     # 1.explore the feature by such method:
    #     # seeds on pooling layer
    #     feature = x.detach()
    #     l = nn.AvgPool2D(2, 2)
    #     f_new = l(feature)
    #     b, _, h, w = f_new.shape
    #     idx = nd.topk(f_new.flatten(), k=int(h * w * proportion))
    #     spark = [nd.floor(idx / w), idx % w]
    #     # spark = nd.stack(*spark)
    #     # spark = nd.concat(spark, spark + nd.array([[1, 0]]).T,
    #     #                   spark + nd.array([[0, 1]]).T, spark + 1, dim=1)
    #     # new_f[i, :, loc_[0][i], loc_[1][i]] = 1
    #     return spark
    #
    # def random_self(self, section, proportion=0.2):
    #     # this method combine the method of explore_self.
    #     # there will never random action totally
    #     return random.sample(section, int(len(section) * proportion))
    #
    # # just do it, the person can not fill the whole image
    # # which branch is active for each batch
    # def assemble_self(self, explored, keypoint, new_f):
    #     loc_ = explored
    #     b, _, _, _ = new_f.shape
    #     for i in range(b):
    #         new_f[i, :, loc_[0][i], loc_[1][i]] += 5
    #     return new_f


class cls_loss(gluon.Block):
    def __init__(self):
        super(cls_loss, self).__init__()
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

    def forward(self, *args):
        outs, lable, transf = args  # weight,
        vis_outs, attr_outs, new_label = self.project(outs, transf, lable)
        loss_vis = self.vis_loss(vis_outs, new_label)
        loss_attr = self.attr_loss(attr_outs, new_label)
        return loss_attr + loss_vis

    def vis_loss(self, outs, labels):
        # visible--cross_entropy
        # loss = gluon.loss.SoftmaxCrossEntropyLoss()
        l = []
        for label, out in zip(labels, outs):
            Isvisible = [label[:, 0], nd.sum(label[:, 1:], axis=1)]
            Isvisible = nd.stack(*Isvisible).T.as_in_context(outs[0].context)
            # Isvisible = nd.array(Isvisible)
            # loss = self.cross_entropy(yhat, Isvisible)
            l.append(self.loss(out, Isvisible))
        return reduce(lambda x, y: nd.sum(x) + nd.sum(y), l)

    def attr_loss(self, outs, labels):
        # ignore this part by set the loss to 0
        # when labels is 0
        l = []
        for label, out in zip(labels, outs):
            attributes = label[:, 1:].as_in_context(outs[0].context)
            discard = nd.sum(attributes, axis=1)
            if len([1 for dis in discard if dis not in [0, 1]]) > 0:
                print "discard has errors"
            l.append(self.loss(out, attributes.T) * discard)
        return reduce(lambda x, y: nd.sum(x) + nd.sum(y), l)

    def project(self, outs, transf, label):
        # -$$- making label
        attr_outs, vis_outs, new_labels, valuable_branch = [], [], [], []
        branches_out_detect = np.array([t['branch'] for t in transf])
        # zero = nd.array([0]).as_in_context(outs[0].context)
        for i in range(len(nums_attres)):  # branches loop
            temp_out_attr, temp_out_vis = [], []
            # cur_nums_attres = nums_attres[i]
            # temp_transf_attr = nd.array(np.eye(
            #     cur_nums_attres, cur_nums_attres - 1, -1)).as_in_context(outs[i].context)
            # temp_transf_vis = nd.array(np.eye(cur_nums_attres, 2)).as_in_context(outs[i].context)
            has_transf = np.where(branches_out_detect == i)[0]
            exist_label = []
            for j in range(len(branches_out_detect)):  # batches loop
                if j in has_transf:
                    transf_attr = transf[j]['attr']
                    transf_vis = transf[j]['vis']
                    temp_out_attr.append(nd.dot(outs[i][j, :], transf_attr))
                    temp_out_vis.append(nd.dot(outs[i][j, :], transf_vis))
                    exist_label.append(nd.array(label[i][j, :]))
                # else:
                #     temp_out_attr.append(nd.dot(outs[i][j, :], temp_transf_attr))
                #     temp_out_vis.append(nd.dot(outs[i][j, :], temp_transf_vis))
            if len(has_transf) > 0:
                vis_outs.append(nd.stack(*temp_out_vis))
                attr_outs.append(nd.stack(*temp_out_attr))
                new_labels.append(nd.stack(*exist_label))
                valuable_branch.append(i)  # same with weight
            # else:
            #     vis_outs.append(zero)
            #     attr_outs.append(zero)
            # -$$- prepare label by has_transf

        return vis_outs, attr_outs, new_labels  # , valuable_branch

    # def softmax(self, y_linear):
    #     exp = nd.exp(y_linear - nd.max(y_linear, axis=1).reshape((-1, 1)))
    #     norms = nd.sum(exp, axis=1).reshape((-1, 1))
    #     return exp / norms
    #
    # def cross_entropy(self, yhat, y):
    #     return - nd.sum(y * nd.log(yhat + 1e-6))


if __name__ == "__main__":
    net = FashionNet()
    p = net.collect_params()
    print('o')
