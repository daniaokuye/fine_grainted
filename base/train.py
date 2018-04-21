# in this process, saptical dropout will be used
import mxnet as mx
from mxnet import gluon, autograd
import numpy as np
from net import FashionNet, cls_loss, test_loss
from units import init, bachfy, transform, reload_hyper, save_hyper, ant_intelligence
from preProcess.fasthion_loader import loader
from base.params import gpus, set_db_handle


def forward_backward(net, multi_loss, data, point, weight, gts, transfs, ctx):
    with autograd.record():
        losses = [multi_loss(net(d, w, p, ctx), w, gt, transf)
                  for (d, w, p, gt, transf) in zip(data, weight, point, gts, transfs)]
        for loss in losses:
            loss.backward()
    return np.sum([loss.asscalar() for loss in losses])


def main():
    epochs = 15
    smoothing_constant = .01
    ctx = [mx.gpu(int(i)) for i in range(gpus)]
    # ctx = mx.gpu(1)
    batch_size = 6 * len(ctx)
    numworks = gpus
    moving_loss = 0

    img_path = '/home1/%s'
    input = img_path % 'FashionAI/base/%s'
    hint_gt = img_path % 'FashionMark/base/%s'
    file_model = '/home/bh/PycharmProjects/Fashion/'
    suffix = 'fashion_%d.params'
    file_hype = 'hyper.txt'

    cur_epoch, lr = reload_hyper(file_model + file_hype)

    net = FashionNet()
    init(net, ctx, file_model, suffix)
    # Optimizer
    Optimizer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})

    train_data = mx.gluon.data.DataLoader(loader(input, hint_gt, transform=transform), batch_size,
                                          last_batch='discard', num_workers=numworks,
                                          batchify_fn=bachfy, shuffle=True)
    multi_loss = cls_loss()
    ant_params = {}
    while cur_epoch < epochs:
        for i, (data, point, weight, gt, transf) in enumerate(train_data):
            data = gluon.utils.split_and_load(data, ctx)
            point = gluon.utils.split_and_load(point, ctx)
            curr_loss = forward_backward(net, multi_loss, data, point, weight, gt, transf, ctx)
            # with autograd.record():
            #     output = net(data, weight, point, ctx)
            #     loss = multi_loss(output, weight, gt, transf)
            # loss.backward()
            Optimizer.step(batch_size / gpus, ignore_stale_grad=True)
            ##########################
            #  Keep a moving average of the losses
            moving_loss = (curr_loss if ((i == 0) and (cur_epoch == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
            if i > 0 and i % 2 == 0:
                print('Batch %d. Loss: %f' % (i, moving_loss))

        ##########################
        #  adjust hyper params
        net.save_params(file_model + suffix % cur_epoch)
        save_hyper(file_model + file_hype, cur_epoch, lr)
        if cur_epoch > 10:
            set_db_handle()
            ant_intelligence(ant_params)
        # adjust lr
        if cur_epoch > 3:
            if lr <= 0.1:
                lr += 0.1 * lr
            else:
                lr -= 0.1 * lr
            Optimizer.set_learning_rate(lr)
        cur_epoch += 1


if __name__ == "__main__":
    main()
