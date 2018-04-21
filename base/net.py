import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.gluon import nn
from params import *
import numpy as np
import random
from base.units import ctx_for_tranf


class FashionNet(gluon.Block):
    def __init__(self, mode='train'):
        super(FashionNet, self).__init__()
        self.mode = mode
        # self.explore_self()
        # self.base_1, self.base_2 = \
        self.vgg_baseNet()
        self.shared = self.shared_FC()
        self.vector = nn.Sequential()
        for i in range(len(classes)):
            self.vector.add(self.extract_vector(nums_attres[i]))

    def forward(self, *args):
        x, w, keypoint = self.sv2gpu(*args)
        outs = {}
        featues = self.base_1(x)
        by_pass_mask = self.spatical_drop_out(featues, keypoint)
        featues = self.base_2(featues)
        featues = featues * by_pass_mask
        featues = self.base_3(featues)
        featues = self.shared(featues)
        for branch in w.keys():
            cur_featue = [featues[i] for i in w[branch]]
            cur_featue = nd.stack(*cur_featue)
            outs[branch] = self.vector[branch](cur_featue)
        return outs

    def sv2gpu(self, *args):
        data, weight, point, ctx = args
        if not isinstance(ctx, list):
            data = data.as_in_context(ctx)
            point = point.as_in_context(ctx)
        return data, weight, point

    def shared_FC(self):
        shared = nn.Sequential()
        with self.name_scope():
            shared.add(nn.Flatten())
            shared.add(nn.Dense(4096))
            shared.add(nn.BatchNorm(axis=1, center=True, scale=True))
            shared.add(nn.Activation(activation='relu'))
        return shared

    # for regression (attributes) & classification (values)
    def extract_vector(self, num_outputs):
        branch = nn.Sequential()
        with self.name_scope():
            branch.add(nn.Dense(4096, in_units=4096))
            branch.add(nn.BatchNorm(axis=1, in_channels=4096, center=True, scale=True))
            branch.add(nn.Activation(activation='relu'))
            branch.add(nn.Dense(num_outputs, in_units=4096))
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
        for (num_convs, channels) in architecture:
            with self.name_scope():
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
        channels = [512, 512, 256, 256, 1]
        self.by_pass = self.build_by_pass(channels)

    def build_by_pass(self, channels):
        out = nn.Sequential()
        with self.name_scope():
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
        proportion_self = get_self_handle()
        b, _, h, w = x.shape
        k = int(h * w * proportion)
        top_k_values = nd.topk(x.detach().flatten(), k=k, ret_typ='value')
        idx = [random.choice(range(int(k * (proportion_self - 0.2)), int(k * proportion_self)))
               for i in range(b)]
        idx = nd.array(idx).as_in_context(x.context)
        # line = list(range(b))
        threshold = top_k_values.pick(idx)
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
        outs, weight, lable, transf = args
        # ctx_for_tranf(transf, ctx)

        vis_outs, attr_outs, new_label = self.project(outs, weight, transf, lable)
        loss_vis = self.vis_loss(vis_outs, new_label)
        loss_attr = self.attr_loss(attr_outs, new_label)
        return loss_attr + loss_vis

    def vis_loss(self, outs, labels):
        # visible--cross_entropy
        # loss = gluon.loss.SoftmaxCrossEntropyLoss()
        l = []
        for label, out in zip(labels, outs):
            Isvisible = [label[:, 0], nd.sum(label[:, 1:], axis=1)]
            Isvisible = self.sv2gpu(nd.stack(*Isvisible).T, out.context)
            l.append(self.loss(out, Isvisible))
        l = nd.sum(l[0]) if len(l) == 1 else reduce(lambda x, y: nd.sum(x) + nd.sum(y), l)
        return l

    def attr_loss(self, outs, labels):
        # ignore this part by set the loss to 0
        # when labels is 0
        l = []
        for label, out in zip(labels, outs):
            attributes = self.sv2gpu(label[:, 1:], out.context)
            discard = nd.sum(attributes, axis=1)
            # if len([1 for dis in discard if dis not in [0, 1]]) > 0:
            #     print "discard has errors"
            l.append(self.loss(out, attributes.T) * discard)
        l = nd.sum(l[0]) if len(l) == 1 else reduce(lambda x, y: nd.sum(x) + nd.sum(y), l)
        return l

    def project(self, outs, weight, transf, label):
        # -$$- making label
        attr_outs, vis_outs, new_labels, valuable_branch = [], [], [], []
        branches_out_detect = np.array([t['branch'] for t in transf])
        valuable_branch = {}
        for batch, branch in enumerate(branches_out_detect):
            if branch in valuable_branch.keys():
                valuable_branch[branch].append(batch)
            else:
                valuable_branch[branch] = [batch]
        assert valuable_branch == weight

        # order matters nothing
        for branch in valuable_branch.keys():
            batches = valuable_branch[branch]
            temp_out_attr, temp_out_vis, exist_label = [], [], []
            ctx = outs[branch].context
            for j, batch in enumerate(batches):
                transf_attr = self.sv2gpu(transf[batch]['attr'], ctx)
                transf_vis = self.sv2gpu(transf[batch]['vis'], ctx)
                assert branch == transf[batch]['branch']
                temp_out_attr.append(nd.dot(outs[branch][j, :], transf_attr))
                temp_out_vis.append(nd.dot(outs[branch][j, :], transf_vis))
                exist_label.append(nd.array(label[batch]))
            vis_outs.append(nd.stack(*temp_out_vis))
            attr_outs.append(nd.stack(*temp_out_attr))
            new_labels.append(nd.stack(*exist_label))
        return vis_outs, attr_outs, new_labels

    def sv2gpu(self, data, ctx):
        if not isinstance(data, nd.ndarray.NDArray):
            data = nd.array(data)
        data = data.as_in_context(ctx)
        return data


class test_loss(gluon.Block):
    def __init__(self):
        # only use once to initialize paramers.
        super(test_loss, self).__init__()
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

    def forward(self, *args):
        outs, weight, lable = args
        l = []
        for branch in weight.keys():
            batches = weight[branch]
            t = [nd.array(lable[i]) for i in batches]
            gt = nd.array(*t).as_in_context(outs[branch].context)

            l.append(self.loss(outs[branch], gt))
        l = nd.sum(l[0]) if len(l) == 1 else reduce(lambda x, y: nd.sum(x) + nd.sum(y), l)
        return l


if __name__ == "__main__":
    net = FashionNet()
    p = net.collect_params()
    print('o')
