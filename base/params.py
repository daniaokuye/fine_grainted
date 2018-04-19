####################
# coco keypoints
###################
classes = ['lapel_design_labels', 'neckline_design_labels', 'collar_design_labels', 'neck_design_labels',
           'coat_length_labels', 'skirt_length_labels', 'pant_length_labels', 'sleeve_length_labels']
nums_attres = [5, 10, 5, 5, 8, 6, 6, 9]
num_keypoints = 54
mirrors = [[2, 5], [3, 6], [4, 7], [8, 11], [9, 12], [10, 13]]
# point & connection(virtual)
point_connection = [[1, 2, 1, 5, 1, 8, 1, 11], [1, 0, 1, 2, 1, 5], [1, 0, 1, 2, 1, 5], [1, 0, 1, 2, 1, 5],
                    [8, 1, 11, 1, 11, 8, 8, 9, 9, 10, 11, 12, 12, 13], [8, 9, 9, 10, 11, 12, 12, 13],
                    [8, 9, 9, 10, 11, 12, 12, 13], [2, 3, 3, 4, 5, 6, 6, 7]]

####################
# net
###################
img_width, img_height = 320, 320  # it should be times of 16
stride_all = 16.0

####################
# train
###################
proportion_step = 0.2
proportion_hint = 1.0  # the total proportion for hint
proportion_self = 0.2  # the total proportion by self

####################
# backup
###################
lapel = [1, 2, 1, 5, 1, 8, 1, 11]
neckline = [1, 0, 1, 2, 1, 5]
collar = [1, 0, 1, 2, 1, 5]
neck = [1, 0, 1, 2, 1, 5]
coat = [8, 1, 11, 1, 11, 8]
skirt = [8, 9, 9, 10, 11, 12, 12, 13]
pant = [8, 9, 9, 10, 11, 12, 12, 13]
sleeve = [2, 3, 3, 4, 5, 6, 6, 7]
