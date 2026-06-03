# -*- coding: utf-8 -*-
from cia_consts import *
import numpy as np

def get_stim_on(on_stim, stim_info):
    p = -1
    last_on = False
    ret = []
    for on in on_stim:
        if on and not last_on:
            p += 1
        if on:
            ret.append(stim_info[:, :, p])
        last_on = on
    print(p)
    return ret

def get_stim_trigger(trigger, f, tr_range):
    ret = []
    for i, tr in enumerate(trigger):
        if tr > 0:
            ret.append(f[i+tr_range[0]+1:i+tr_range[1]+1])
    if len(ret[-1]) != len(ret[0]):
        ret = ret[:-1]
    return np.array(ret)

def get_stim_range(on_stim, merge_inter=20):
    d = np.diff(on_stim.astype(int))
    on = np.nonzero(d == 1)[0]
    off = np.nonzero(d == -1)[0]
    ret = []
    last_o, last_f = -1000, -1000
    for o, f in zip(on, off):
        if last_o >= 0:
            if o - last_f < merge_inter:
                ret.append([last_o, f])
            else:
                ret.append([last_o, last_f])
        last_o = o
        last_f = f
    if len(ret) == 0 or ret[-1][-1] != last_f:
        if len(on):
            ret.append([on[0], len(d)])
        else:
            ret.append([last_o, last_f])
    return ret

# input: array [false false true true true ....]
# return: [[3,3],[7,2]]   " (true position index, consecutive true numbers)
def get_all_sti_pos(l):
    num = 0
    res = []
    for i in range(len(l)):
        if l[i] == True:
            num += 1
        else:
            if num > 1:  # 如果有连续的
                res.append([i - num, num])
            num = 0
    if num > 1:
        res.append([i - num + 1, num])
    return res

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def load_pd_h5(pd_name):
    import h5py
    return h5py.File(pd_name, "r")['AI']['ai7'][:, 0]

def calc_cor_map(mat_name, img_rate):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    parent = os.path.dirname(mat_name)
    exp_name = os.path.basename(mat_name)[7:-4]
    print(exp_name)
    csv_name = os.path.join(parent, exp_name, "dFF.csv")
    if not os.path.exists(csv_name):
        csv_name = os.path.join(os.path.dirname(parent), exp_name, "dFF.csv")
    pd_name = os.path.join(parent, "PD-" + exp_name, "Episode001.h5")
    if not os.path.exists(pd_name):
        pd_name = os.path.join(os.path.dirname(parent), "PD-" + exp_name, "Episode001.h5")
    import h5py
    from scipy.io import loadmat
    pd_info = h5py.File(pd_name, "r")['AI']['ai7'][:, 0]  # (10801499, 1)
    stim_info = loadmat(mat_name)["frames"]  # (10, 19, 1000)
    # dFF = get_dFF(pd.read_csv(csv_name), (10, 40))  # 14500
    dFF = pd.read_csv(csv_name)
    # plt.plot(pd_info)
    # dFF.plot()
    frames = len(dFF)
    t = np.arange(frames) / img_rate
    pd_f = pd_info[(t * PD_RATE).astype(int)]
    row, col, count = stim_info.shape
    on_stim = pd_f > PD_THRESHOLD
    # plt.plot(pd_f)

    # dFF_on = dFF[on_stim]
    # stim_on = np.array(get_stim_on(on_stim, stim_info))
    fig, axs = plt.subplots(3, 4, figsize=(8, 5), sharex=True)
    plt.subplots_adjust(left=0.05, right=0.95, hspace=0.01, wspace=0.12)
    axs = axs.flatten()
    a = 0
    maps = []
    f_maps = []

    tr_range = (-int(img_rate / 2), int(img_rate / 2))
    for name in dFF.columns:
        on = np.diff(on_stim.astype(int))
        tr = get_stim_trigger(on, dFF[name], tr_range)
        pre, post = tr[:, :-tr_range[0]], tr[:, -tr_range[1]:]
        pre_a, post_a = np.mean(pre, 1), np.mean(post, 1)
        inc = post_a - pre_a#((post_a - pre_a) > 0).astype(int)-0.5
        map = np.zeros((row, col), dtype=float)
        map_sum = np.zeros((row, col), dtype=float)
        map_n = np.zeros((row, col))
        stim_count = min(len(inc), count)
        for p in range(stim_count):
            if post_a[p] > 0.1 and post_a[p] > 1.1 * pre_a[p]:
                map += stim_info[:, :, p] * post_a[p]
            for i in range(row):
                for j in range(col):
                    if stim_info[i, j, p] > 0:
                        map_sum[i, j] += post_a[p]
                        map_n[i, j] += 1
        maps.append(map / stim_count)
        f_maps.append(map_sum / map_n)

        ax = axs[a]
        ax.axis("off")
        im = ax.imshow(map, cmap="bwr")
        # im = ax.imshow(map_sum/map_n, cmap="bwr")
        plt.colorbar(im, ax=ax)
        ax.set_title(name)
        a += 1
    plt.tight_layout()
    plt.savefig(os.path.join(parent, "jpeg/_map"))

    # name = "3"
    # on = np.diff(on_stim.astype(int))
    # tr = get_stim_trigger(on, dFF[name], tr_range)
    # pre, post = tr[:, :-tr_range[0]], tr[:, -tr_range[1]:]
    # pre_a, post_a = np.mean(pre, 1), np.mean(post, 1)
    # inc = post_a - pre_a
    # stim_count = min(len(inc), count)
    # for i in range(row):
    #     for j in range(col):
    #         lines = []
    #         for p in range(stim_count):
    #             if stim_info[i, j, p] > 0 and post_a[p] > 0.1 and post_a[p] > 1.1 * pre_a[p]:
    #                 # print(pre_a[p], post_a[p])
    #                 lines.append(tr[p])
    #         plot_lines_with_err(lines, os.path.join(parent, "jpeg/%02d_%02d" % (i, j)))
