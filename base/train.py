# in this process, saptical dropout will be used
import mxnet as mx
from mxnet import gluon, autograd, nd
from net import FashionNet, cls_loss
from units import init, bachfy, transform, ctx_for_tranf
from preProcess.fasthion_loader import loader


def main():
    epochs = 15
    smoothing_constant = .01
    gpus = '1,2,3,4'
    ctx = [mx.gpu(int(i)) for i in gpus.split(',')][0]
    batch_size = 4  # * len(ctx)
    numworks = 1
    net = FashionNet()
    init(net, ctx)

    img_path = '/home1/%s'
    input = img_path % 'FashionAI/base/%s'
    hint_gt = img_path % 'FashionMark/base/%s'

    # Optimizer
    Optimizer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .05})

    train_data = mx.gluon.data.DataLoader(loader(input, hint_gt, transform=transform), batch_size,
                                          last_batch='discard', num_workers=numworks,
                                          batchify_fn=bachfy, shuffle=True)
    # test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False),
    #                                      batch_size, shuffle=False)

    multi_loss = cls_loss()
    for e in range(epochs):
        for i, (d, w, gt, transf, point) in enumerate(train_data):
            data = d.as_in_context(ctx)
            weight = w.as_in_context(ctx)
            point = point.as_in_context(ctx)
            ctx_for_tranf(transf, ctx)
            with autograd.record():
                output = net(data, weight, point)
                loss = multi_loss(output, gt, transf)
            loss.backward()
            Optimizer.step(data.shape[0])

            ##########################
            #  Keep a moving average of the losses
            ##########################
            # curr_loss = nd.mean(loss).asscalar()
            # moving_loss = (curr_loss if ((i == 0) and (e == 0))
            #                else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
            #
            # if i > 0 and i % 200 == 0:
            #     print('Batch %d. Loss: %f' % (i, moving_loss))

        # test_accuracy = evaluate_accuracy(test_data, net)
        # train_accuracy = evaluate_accuracy(train_data, net)
        # print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))


if __name__ == "__main__":
    main()
