from __future__ import print_function
from mxnet.gluon import nn
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np

val = nd.array([[3, 4, 5, 1], [5, 6, 4, 2]])

w1 = nd.array(np.eye(4))
w2 = nd.array(np.eye(4))
gt = nd.array([[3, 4, 5, 1], [5, 6, 4, 2]])
b1 = nd.ones(shape=(1,))

val0 = val.detach()
val.attach_grad()
val0.attach_grad()
w1.attach_grad()
b1.attach_grad()

with autograd.record():
    val1 = val * 2
    val2 = np.zeros(val.shape)
    # a = [nd.dot(val1[0, :], w1), nd.dot(val1[1, :], w1)]
    # a[1] = val1[1, :]
    # res = nd.stack(*a)
    val2[1, :] = 1
    val3=nd.array(val2)
    res = nd.dot(val1, w1)
    res *= val3
    # res = nd.zeros_like(val1)
    # res[:, :] = val1[:, :]
    res2 = res + b1 - gt
res2.backward()
print(w1.grad)
print(val.grad)
