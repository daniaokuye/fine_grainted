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
img_width, img_height = 256, 256  # it should be times of 16
stride_all = 16.0

####################
# train
###################
gpus = 8
proportion_step = 0.1


class GlobalVar:
    proportion_hint = 1.0  # the total proportion for hint
    proportion_self = 0.2  # the total proportion by self


def set_db_handle():
    GlobalVar.proportion_hint -= proportion_step
    if GlobalVar.proportion_hint < 0:
        GlobalVar.proportion_hint = 0
    GlobalVar.proportion_self += proportion_step
    if GlobalVar.proportion_self < 0:
        GlobalVar.proportion_self = 0


def init_db_handle(_hint, _self):
    GlobalVar.proportion_self = _self
    GlobalVar.proportion_hint = _hint


def get_self_handle():
    return GlobalVar.proportion_self


def get_hint_handle():
    return GlobalVar.proportion_hint


'''
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

# # zero = nd.array([0]).as_in_context(outs[0].context)
# for i in range(len(nums_attres)):  # branches loop
#     temp_out_attr, temp_out_vis = [], []
#     # cur_nums_attres = nums_attres[i]
#     # temp_transf_attr = nd.array(np.eye(
#     #     cur_nums_attres, cur_nums_attres - 1, -1)).as_in_context(outs[i].context)
#     # temp_transf_vis = nd.array(np.eye(cur_nums_attres, 2)).as_in_context(outs[i].context)
#     has_transf = np.where(branches_out_detect == i)[0]
#     exist_label = []
#     for j in range(len(branches_out_detect)):  # batches loop
#         if j in has_transf:
#             transf_attr = transf[j]['attr']
#             transf_vis = transf[j]['vis']
#             temp_out_attr.append(nd.dot(outs[i][j, :], transf_attr))
#             temp_out_vis.append(nd.dot(outs[i][j, :], transf_vis))
#             exist_label.append(nd.array(label[i][j, :]))
#         # else:
#         #     temp_out_attr.append(nd.dot(outs[i][j, :], temp_transf_attr))
#         #     temp_out_vis.append(nd.dot(outs[i][j, :], temp_transf_vis))
#     if len(has_transf) > 0:
#         vis_outs.append(nd.stack(*temp_out_vis))
#         attr_outs.append(nd.stack(*temp_out_attr))
#         new_labels.append(nd.stack(*exist_label))
#         valuable_branch.append(i)  # same with weight
#     # else:
#     #     vis_outs.append(zero)
#     #     attr_outs.append(zero)
#     # -$$- prepare label by has_transf

# return vis_outs, attr_outs, new_labels  # , valuable_branch
'''
