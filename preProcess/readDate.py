from base.params import *
import csv

img_path = '/home1/%s/%s'
input = 'FashionAI/base'
gt = 'FashionMark/base'
annotation = '/home1/FashionAI/base/Annotations/label.csv'


def read_csv(path):
    a, b, c = [], [], 0
    with open(path, 'r')as f:
        anno = csv.reader(f)
        for i, info in enumerate(anno):
            # if i > 5:
            #     break
            img_path, label, attr = info
            if attr.count('y') > 1:
                c += 1
                print info
            if 'length' in label:
                if 'm' in attr and 'y' == attr[0]:
                    a.append(info)
                    # yi = attr.index('y')
                    # mi = attr.index('m')
                    # if abs(yi - mi) > 1:
                    #     a.append(attr)
                b.append(label)
        # return list(anno)
    print  c  # set(a), set(b),


if __name__ == "__main__":
    read_csv(annotation)
