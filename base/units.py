import mxnet as mx
from mxnet import nd
import numpy as np


# this function will be used in evalution
def evaluate_accuracy(data_iterator, net, ctx):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


# Initialize parameters
def init(net, ctx):
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)


def transform(data):
    return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255


def bachfy(dataset):
    imgs, cls_wights, gt_labels, transf_matrix, keypoints = zip(*dataset)
    imgs = [img.asnumpy() for img in imgs]
    imgs = nd.array(np.stack(imgs))
    gt_labels = [np.stack(label) for label in zip(*gt_labels)]
    if sum([np.sum(np.sum(x, axis=1) > 1) for x in gt_labels]) > 0:
        print 'o'
    keypoints = nd.stack(*keypoints).expand_dims(1)
    # NDArray, NDArray, list(NDArray), tuple(dict), NDArray
    return imgs, nd.array(cls_wights), gt_labels, transf_matrix, keypoints


def ctx_for_tranf(input, ctx):
    for article in input:
        article['vis'] = nd.array(article['vis']).as_in_context(ctx)
        article['attr'] = nd.array(article['attr']).as_in_context(ctx)
