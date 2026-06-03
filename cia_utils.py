# -*- coding: utf-8 -*-
#-*- coding : utf-8-*-
# coding:unicode_escape

import os
import cv2
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import NoNorm
from glob import glob
from scipy.stats import circmean

from cia_consts import *
from cia_photodiode import get_stim_range, get_all_sti_pos, find_nearest
import imageio
from matplotlib.animation import FuncAnimation
COLORS = ["k", "r", "g", "b", "y", "c", "m", "gray", "pink", "springgreen", "deepskyblue", "yellow",]


def load_memmap(filename, mode='r'):
    file_to_load = filename
    filename = os.path.split(filename)[-1]
    fpart = filename.split('_')[1:-1]  # The filename encodes the structure of the map
    d1, d2, d3, T, order = int(fpart[-9]), int(fpart[-7]), int(fpart[-5]), int(fpart[-1]), fpart[-3]
    Yr = np.memmap(file_to_load, mode=mode, shape=(d1 * d2 * d3, T), dtype=np.float32, order=order)
    if d3 == 1:
        dims = (d1, d2)
    else:
        dims = (d1, d2, d3)
    return np.reshape(Yr.T, [T] + list(dims), order='F')

def calc_avg_frame(m, parent):
    avg_frame = np.mean(m, axis=0)
    eq = norm_img(avg_frame)
    cv2.imwrite(parent + "/i_avg.png", eq)
    std_frame = np.std(m, axis=0)
    eq2 = norm_img(std_frame)
    cv2.imwrite(parent + "/i_std.png", eq2)
    max_frame = np.max(m, axis=0)
    eq3 = norm_img(max_frame)
    cv2.imwrite(parent + "/i_max.png", eq3)
    eqc = np.transpose([eq3, eq2, eq], (1, 2, 0))
    cv2.imwrite(parent + "/i_all.png", eqc)

    # cluster_roi(m, parent)
    # show_cluster_roi(m, parent)
    # th2, res = cv2.threshold(eq3.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imwrite(parent + "/i_test.png", res)

def cluster_roi(m, parent):
    eq3 = norm_img(np.max(m, axis=0))*2  # NOTE: m: 1400*128*128
    th2, img = cv2.threshold(eq3.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    p_l = np.nonzero(img)
    n = len(p_l[0])
    cluster_m = np.zeros(img.shape, dtype=np.uint8)
    # for i in range(n):
    #     print(i)
    #     for j in range(n):
    #         cor_map[i, j] = cor(m[:, p_l[0][i], p_l[1][i]], m[:, p_l[0][j], p_l[1][j]])
    X = m[:, p_l[0], p_l[1]].T  # NOTE: n*1400

    from pyclustering.cluster.kmeans import kmeans
    from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
    from pyclustering.utils.metric import type_metric, distance_metric
    metric = distance_metric(type_metric.USER_DEFINED, func=lambda a, b: 1-cor(a, b))
    initial_centers = kmeans_plusplus_initializer(X, 10).initialize()
    kmeans_instance = kmeans(X, initial_centers, metric=metric)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    pred = np.zeros((X.shape[0]))
    cf = []
    for i, c in enumerate(clusters):
        pred[c] = i + 1
        cf.extend(c)

    plot_fast_cor_map(X, cf)
    # plt.show()
    plt.savefig(parent + "/i_cor.png")

    # from sklearn.cluster import KMeans, DBSCAN
    # estimator = KMeans(n_clusters=20, max_iter=500)
    # estimator = DBSCAN(eps=0.8, min_samples=50, metric=lambda a, b: 1-cor(a, b))

    # estimator.fit(X)
    # pred = estimator.labels_

    cluster_m[p_l[0], p_l[1]] = pred
    cv2.imwrite(parent + "/i_test.png", norm_img(cluster_m))

    for i in range(int(np.max(pred))):
        show_largest_for_roi(m, cluster_m, i)
        plt.savefig(parent + "/i_largest_%d.png" % i)
    # plt.imshow(cluster_m)

def plot_cor_map(X, cf):
    n = len(cf)
    cor_map = np.zeros((n, n))
    for i, cf1 in enumerate(cf):
        print(i, "/", n)
        for j, cf2 in enumerate(cf):
            cor_map[i][j] = cor(X[cf1], X[cf2])
    plt.imshow(cor_map, cmap="bwr")
    plt.colorbar()

def plot_fast_cor_map(X, cf):
    from scipy.stats import zscore
    Xz = zscore(X, axis=1)
    n = len(cf)
    cor_map = np.zeros((n, n))
    for i, cf1 in enumerate(cf):
        print(i, "/", n)
        for j, cf2 in enumerate(cf):
            cor_map[i][j] = np.dot(Xz[cf1], Xz[cf2])/n
    plt.imshow(cor_map, cmap="bwr")
    plt.colorbar()

def show_cluster_roi(m, parent):
    cluster_m = cv2.imread(parent + "/i_test.png")
    for k, i in enumerate(np.unique(cluster_m)):
        show_largest_for_roi(m, cluster_m, i)
        plt.savefig(parent + "/i_largest_%d.png" % k)

def show_largest_for_roi(m, cluster_m, label):
    p_l = np.nonzero(cluster_m == label)
    X = m[:, p_l[0], p_l[1]]
    n = X.shape[1]
    # if n < 1600:
    #     cor_map = np.zeros((n, n))
    #     for i in range(n):
    #         print(i, "/", n)
    #         for j in range(n):
    #             cor_map[i][j] = cor(X[:, i], X[:, j])
    #     plt.imshow(cor_map)
    #     plt.colorbar()
    #     plt.show()
    f = np.argmax(np.sum(X, axis=1))
    plt.figure(figsize=(6, 3))
    plt.title(str(f))
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(m[f])
    axs[0].scatter(p_l[1], p_l[0], alpha=1, color="r", s=1)
    axs[1].imshow(m[f])

def norm_img(img):
    pmax = np.percentile(img, 99)
    pmin = np.percentile(img, 1)
    r_norm = 255.0 * (img - pmin) / (pmax - pmin)
    np.clip(r_norm, 0, 255, r_norm)
    return r_norm.astype(np.uint8)

def load_roi(roi_file, shape):
    if roi_file.endswith(".zip"):
        return load_roi_zip(roi_file, shape)
    elif roi_file.endswith(".npy"):
        roi = np.load(roi_file, allow_pickle=True)
        names = np.arange(len(roi))
        xy = [r[:, 0, :] for r in roi]
        return names, roi_contours_to_points(xy, shape), xy

def load_roi_zip(roizip, shape):
    from read_roi import read_roi_zip# read_roi_file
    rois = read_roi_zip(roizip)

    xy = []
    names = []
    for d in rois.values():
        if d["type"] == "polygon":
            xy.append(np.array(tuple(zip(d["x"], d["y"]))))
            names.append(d["name"])
        elif d["type"] == "rectangle":
            l, t, w, h = d["left"], d["top"], d["width"], d["height"]
            xy.append(np.array([[l, t], [l + w, t], [l + w, t + h], [l, t + h]]))
            names.append(d["name"])

    return names, roi_contours_to_points(xy, shape), xy

def roi_contours_to_points(contours_xy, shape):
    xy = []
    for c in contours_xy:
        temp = np.zeros(shape)
        cv2.drawContours(temp, [c.astype(int)], 0, color=1, thickness=-1)
        xy.append(temp.nonzero())
        # plt.imshow(temp, cmap="Greys_r")
        # plt.show()
    return xy

def calc_all_roi_F(roizip, m, parent):
    print("Do calc F...")
    names, points, contours = load_roi(roizip, m[0].shape)
    ret = [[] for i in range(len(names))]
    for i, mi in enumerate(m):
        for j, xyj in enumerate(points):
            ret[j].append(float(mi[xyj].mean()))
    # plot_lines(ret, names)
    plot_rois(contours, m)
    plt.savefig(parent + "/roi.png")

    F = pd.DataFrame(np.array(ret).T, columns=names)
    F.to_csv(parent + "/F.csv", index=False)
    plot_lines([F[n] for n in names], names)
    plt.savefig(parent + "/F.png")

    zs_csv = parent + "/zscore.csv"
    zs = get_zscore(F)
    zs.to_csv(zs_csv, index=False)
    write_roi_dFF_video(zs, contours, m, zs_csv)

    dFF_csv = parent + "/dFF.csv"
    dFF = get_dFF(F)
    dFF.to_csv(dFF_csv, index=False)
    write_roi_dFF_video(dFF, contours, m, dFF_csv)
    print(dFF_csv)
    return dFF_csv, zs_csv

def write_roi_dFF_video(dFF, contours, m, name):
    return
    # fig = plt.figure(figsize=(6, 6))
    # ax = plt.gca()
    # ax.set_position([0, 0, 1, 1], which="both")
    # fig.canvas.draw()
    # w, h = fig.canvas.get_width_height()
    if os.path.exists(name + "_roi.avi"):
        return
    h, w = m[0].shape
    output_video = cv2.VideoWriter(name + "_roi.avi", cv2.VideoWriter_fourcc(*"DIVX"), 30, (w, h))
    cmap = plt.cm.get_cmap("Greys_r")
    dFFn = norm_img(np.array(dFF))
    for i, mi in enumerate(m):
        print(i)
        # ax.cla()
        # ax.imshow(mi, cmap="Greys_r")
        # for j, xys in enumerate(contours):
        #     ax.add_patch(plt.Polygon(xys, linewidth=0, color=cmap(dFFn.loc[i, j])))
        # write_fig_to_video(output_video, fig, w, h)
        img = np.zeros(mi.shape, dtype=np.uint8)#np.array(mi, dtype=np.uint8)
        for j, xys in enumerate(contours):
            cv2.drawContours(img, [contours[j].astype(int)], 0, color=cmap(dFFn[i, j])[0]*255, thickness=-1)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        output_video.write(img_bgr)
    output_video.release()

def write_zscore_video(m, name, pv_idx=None, contours=None):
    # n, h, w = m.shape
    cmap = plt.cm.get_cmap("plasma")
    mg = np.array([cv2.GaussianBlur(mi, (9, 9), 0) for mi in m])
    me = np.mean(mg, axis=0)
    m1 = mg - me
    m1[m1 < 0] = 0
    m1 = norm_img(m1)

    m2 = []
    for i, mi in enumerate(m1):
        img = mi.astype(np.uint8)#cv2.GaussianBlur(mi.astype(np.uint8), (3, 3), 0)
        if pv_idx is not None and i < len(pv_idx) and pv_idx[i] >= 0:
            # cv2.putText(img, str(pva[i]), (6, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 128), 1)
            cv2.drawContours(img, [contours[pv_idx[i]].astype(int)], 0, color=(255, 255, 255), thickness=1)
        m2.append(img)
    write_video(name, (cmap(m2)[:,:,:,:3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR, FICTRAC_RATE)

    # m_mean = np.mean(m2, axis=0)
    # m_std = np.std(m2, axis=0)
    # mz = []
    # for mi in m2:
    #     zs = (mi - m_mean) / m_std
    #     zs = np.nan_to_num(zs)
    #     zs[zs<0]=0
    #     mz.append(zs)
    # mz = norm_img(mz)
    # write_video(name + "_zscore.avi", (cmap(mz)[:,:,:,:3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR, FICTRAC_RATE)

# def write_fig_to_video(output_video, fig, w, h, desc=None, save_img_path=None):
#     fig.canvas.draw()
#     img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     img.shape = (h*2, w*2, 3)
#     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     desc and cv2.putText(img_bgr, desc, (6, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 128), 1)
#     output_video.write(img_bgr)
#     save_img_path and cv2.imwrite(save_img_path, img_bgr)
#     return img_bgr

def plot_lines(lines, names, ylim=None, xlim_r=1):
    n = len(lines)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(30, 12), dpi=300)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05, hspace=0)
    for i, r in enumerate(lines):
        ax = axes[n - i - 1]
        ax.plot(r, c=COLORS[i%12])
        ax.set_ylabel(names[i], rotation=0, fontsize=6)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim_r is not None:
            ax.set_xlim((0, xlim_r * len(r)))

def plot_lines_with_err(py_l, name, USE_SEM=False):
    py = np.nanmean(py_l, axis=0)
    pe = np.nanstd(py_l, axis=0)
    if USE_SEM:
        pe = pe / np.sqrt(np.count_nonzero(~np.isnan(py_l), axis=0))
    plt.figure()
    plt.plot(np.transpose(py_l), alpha=0.1)
    plt.plot(py, "--")
    plt.fill_between(range(len(py_l[0])), py - pe, py + pe, alpha=0.1)
    plt.savefig(name)
    plt.close()

def plot_lines_in_one(lines, names):
    plt.figure(figsize=(15, 10), dpi=300)
    for i, n in enumerate(lines):
        plt.plot(n, linewidth=1, c=COLORS[i%12], alpha=0.5)

def plot_pva(ax, dff, unwrap=False, c="k", offset=0):
    # max_idx = np.argmax(dff.T, axis=0) + offset
    pv_dir, pv_len = calc_pva(dff)
    if unwrap:
        pv_dir = unwrap_dir(pv_dir)
    pi = dff.shape[1]/2
    plot_angle(ax, (np.array(pv_dir)+np.pi)*pi/np.pi+offset, c, pi)

def plot_hot(df, save_name, img_rate, is_PB=False):
    plt.figure(figsize=(24, 2), dpi=300)
    plt.subplots_adjust(left=0.02, right=0.99)
    dff = df.to_numpy()
    plt.pcolor(dff.T, cmap="jet")
    if is_PB:
        plot_pva(plt.gca(), dff[:, :9])
        plot_pva(plt.gca(), dff[:, 9:], offset=9)
    else:
        plot_pva(plt.gca(), dff)

    frames = len(dff)
    seconds = int(frames / img_rate)
    labels = np.linspace(0, seconds, 9)
    plt.xticks(labels * img_rate, labels)
    # plt.colorbar()
    plt.savefig(save_name + ".png")

    plt.figure(figsize=(24, 2), dpi=300)
    plt.subplots_adjust(left=0.02, right=0.99)
    if is_PB:
        plot_pva(plt.gca(), dff[:, :9], unwrap=True)
        plot_pva(plt.gca(), dff[:, 9:], unwrap=True, c="gray", offset=9)
    else:
        plot_pva(plt.gca(), dff, unwrap=True)
    plt.xticks(labels * img_rate, labels)
    plt.xlim(0, seconds * img_rate)
    plt.savefig(save_name + "_unwrap.png")

def lim_dir(dir1, pi=np.pi):
    if dir1 > pi:
        dir1 -= 2*pi
    elif dir1 < -pi:
        dir1 += 2*pi
    return dir1

def lim_dir_l(dir_l, pi=np.pi):
    dir1 = dir_l.copy()
    dir1[dir1 > pi] -= 2*pi
    dir1[dir1 < -pi] += 2*pi
    return dir1

def unwrap_dir(v, pi=np.pi):
    ret = []
    li = v[0]
    offset = 0
    for i in v:
        i += offset
        d = i - li
        if d > pi:
            offset -= 2*pi
            i -= 2*pi
        elif d < -pi:
            offset += 2*pi
            i += 2*pi
        ret.append(i)
        li = i
    return ret

def unwrap_dir_win(v, rate, sec=10):
    win = int(sec*rate)
    ret = []
    for i in range(0, len(v), win):
        u = unwrap_dir(v[i:i + win])
        ret.extend([d-u[0] for d in u])
    return ret

def plot_angle(ax, al, c, pi=np.pi, xs=None):
    # al = unwrap_dir(al)  #[lim_dir(a-2) for a in al]
    if xs is None:
        xs = range(len(al))
    last = 0
    start = 0
    for i, a in enumerate(al):
        if abs(a - last) > pi:
            ax.plot(xs[start:i], al[start:i], c=c, lw=0.6)
            start = i
        last = a
    ax.plot(xs[start:len(al)], al[start:], c=c, lw=0.6)

def plot_scatter(ax, al, c, xs=None):
    if xs is None:
        xs = range(len(al))
    ax.scatter(xs, al, c=c, s=0.1)

def rotate_unit_vec(rad):
    return np.array([np.cos(rad), np.sin(rad)])

def calc_pva(f):
    # pv_len = np.max(f, axis=1)
    # pv_dir = (np.argmax(f, axis=1) + 0.5) / 8 * np.pi - np.pi

    vs = np.array([rotate_unit_vec(2*np.pi*r/f.shape[1]-np.pi) for r in range(f.shape[1])])
    pv = f.dot(vs)
    pv_len = np.sqrt(np.sum(pv ** 2, axis=1)) / f.shape[1]
    pv_dir = np.arctan2(pv[:, 1], pv[:, 0])
    return pv_dir, pv_len

def bump_amplitude(f):
    bump_a = np.max(f, axis=1) - np.min(f, axis=1)
    nor_bump_a = []
    max = np.mean(bump_a[np.argpartition(bump_a,-50)[-50:]])
    min = np.min(bump_a)
    for x in bump_a:
        x = float(x - min) / (max - min)
        nor_bump_a.append(x)
    return nor_bump_a

def bin_bump_speed(f,ft_speed,ft_vx,ft_vy,ft_vz): #for now, ft_speed-mm/s, ft_vx`ft_vy`ft_vz-rad/s
    nor_bump_a = bump_amplitude(f)
    all_speed = {'0-':[], '0~1':[] , '1~2':[],'2~3':[],'3~4':[],'4~5':[],'5~6':[],'6~7':[],'7~8':[],'8+':[]}
    forward_speed = {'0-':[], '0~1':[] , '1~2':[],'2~3':[],'3~4':[],'4~5':[],'5~6':[],'6~7':[],'7~8':[],'8~9':[],'9~10':[],'10+':[]}
    lateral_speed = {'-4-':[], '-4~-3':[] , '-3~-2':[],'-2~-1':[],'-1~0':[],'0~1':[],'1~2':[],'2~3':[],'3~4':[],'4+':[]}
    # rotation_speed = {}
    for i in range(len(ft_speed) - 10):
        mean_ft_speed = np.mean(ft_speed[i:i+10])
        mean_ft_vx = np.mean(ft_vx[i:i+10] * 8) # forward speed
        mean_ft_vy = np.mean(ft_vy[i:i+10] * 8) # lateral speed

        if mean_ft_speed < 0:
            all_speed['0-'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_speed > 0 and mean_ft_speed <= 1:
            all_speed['0~1'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_speed > 1 and mean_ft_speed <= 2:
            all_speed['1~2'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_speed > 2 and mean_ft_speed <= 3:
            all_speed['2~3'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_speed > 3 and mean_ft_speed <= 4:
            all_speed['3~4'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_speed > 4 and mean_ft_speed <= 5:
            all_speed['4~5'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_speed > 5 and mean_ft_speed <= 6:
            all_speed['5~6'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_speed > 6 and mean_ft_speed <= 7:
            all_speed['6~7'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_speed > 8:
            all_speed['8+'].append(np.mean(nor_bump_a[i:i+10]))

        if mean_ft_vx < 0:
            forward_speed['0-'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_vx > 0 and mean_ft_vx <= 1:
            forward_speed['0~1'].append(np.mean(nor_bump_a[i:i + 10]))
        elif mean_ft_vx > 1 and mean_ft_vx <= 2:
            forward_speed['1~2'].append(np.mean(nor_bump_a[i:i + 10]))
        elif mean_ft_vx > 2 and mean_ft_vx <= 3:
            forward_speed['2~3'].append(np.mean(nor_bump_a[i:i + 10]))
        elif mean_ft_vx > 3 and mean_ft_vx <= 4:
            forward_speed['3~4'].append(np.mean(nor_bump_a[i:i + 10]))
        elif mean_ft_vx > 4 and mean_ft_vx <= 5:
            forward_speed['4~5'].append(np.mean(nor_bump_a[i:i + 10]))
        elif mean_ft_vx > 5 and mean_ft_vx <= 6:
            forward_speed['5~6'].append(np.mean(nor_bump_a[i:i + 10]))
        elif mean_ft_vx > 6 and mean_ft_vx <= 7:
            forward_speed['6~7'].append(np.mean(nor_bump_a[i:i + 10]))
        elif mean_ft_vx > 7 and mean_ft_vx <= 8:
            forward_speed['7~8'].append(np.mean(nor_bump_a[i:i + 10]))
        elif mean_ft_vx > 8 and mean_ft_vx <= 9:
            forward_speed['8~9'].append(np.mean(nor_bump_a[i:i + 10]))
        elif mean_ft_vx > 10:
            forward_speed['10+'].append(np.mean(nor_bump_a[i:i + 10]))

        if mean_ft_vy < -4:
            lateral_speed['-4-'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_vy > -4 and mean_ft_vy < -3:
            lateral_speed['-4~-3'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_vy > -3 and mean_ft_vy < -2:
            lateral_speed['-3~-2'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_vy > -2 and mean_ft_vy < -1:
            lateral_speed['-2~-1'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_vy > -1 and mean_ft_vy < 0:
            lateral_speed['-1~0'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_vy > 0 and mean_ft_vy < 1:
            lateral_speed['0~1'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_vy > 1 and mean_ft_vy < 2:
            lateral_speed['1~2'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_vy > 2 and mean_ft_vy < 3:
            lateral_speed['2~3'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_vy > 3 and mean_ft_vy < 4:
            lateral_speed['3~4'].append(np.mean(nor_bump_a[i:i+10]))
        elif mean_ft_vy > 4:
            lateral_speed['4+'].append(np.mean(nor_bump_a[i:i+10]))

    return all_speed, forward_speed, lateral_speed

def unify_sample(f, ts, fps, n): # no interp
    idx = []
    inter = 1.0 / fps
    i = 0
    for j in range(n):
        t = j * inter
        if ts[i+1] < t:
            while i < len(ts) - 1 and ts[i+1] < t:
                i += 1
        if i+1 >= len(f):
            break
        if t < ts[i]:
            idx.append(i)
            continue
        if i >= len(ts) - 1:
            break
        if t - ts[i] > ts[i+1] - t:
            idx.append(i+1)
        else:
            idx.append(i)
    return f[idx], idx

def calc_offset(ft, pv):
    n = min(len(ft), len(pv))
    o = ft[:n] - pv[:n]
    # o[o < 0] += 2*np.pi
    o[o > np.pi] -= 2*np.pi
    o[o < -np.pi] += 2*np.pi
    return o

def plot_slide_cor(ax, ft, pv, rate):
    for win_sec in [10]:#4, 16, 64
        win = int(win_sec * rate)
        ax.plot(*slide_cor(ft, pv, win, rate))

def circular_std(o):
    o = o[~np.isnan(o)]
    # s = np.sin(o)
    # c = np.cos(o)
    # return np.sqrt(-2 * np.log(np.sqrt(np.sum(s) ** 2 + np.sum(c) ** 2) / np.linalg.norm(o)))
    from scipy.stats import circstd
    return circstd(o)

def format_time(t):
    if t < 60:
        return "%.2f" % t
    return "%d:%.2f" % (t//60, t%60)

def non_nan(s):
    return s[~np.isnan(s)]

def plot_dFF_MB_grating(dFF_name, exp_info, unify_rate=None, use_fictrac = False):
    parent = os.path.dirname(dFF_name)
    if use_fictrac:
        fictrac_name = get_ft_dat(parent)
        ft = pd.read_csv(fictrac_name, header=None).to_numpy()  # (15079frame, 23field) 50Hz
        bar_name = get_ft_bar(parent)
    pd_name = get_pd_h5(parent)

    dFF = pd.read_csv(dFF_name).to_numpy()[:, 1:]  # (2200frame, 16roi) 6.736Hz
    import h5py
    ni = h5py.File(pd_name, "r")
    pd_info_raw = ni['AI']["photodiode"][:, 0]  # (1640499frame,) 5000Hz
    frame_counter = ni['CI']["FrameCounter"][:, 0]
    frame_out = ni['DI']["FrameOut"][:, 0]
    img_rate = exp_info["frameRate"]  # only one z-step in this experiment
    if not unify_rate:
        unify_rate = img_rate
        use_img_rate = True
    seconds = len(pd_info_raw) / PD_RATE  # int(len(dFF) / img_rate)
    n = int(seconds * unify_rate)
    seconds = n / unify_rate

    frame_start = np.nonzero(np.diff(frame_counter) > 0)[0]
    volume_frame = 1
    frame_time = frame_start[int(exp_info["steps"] / 2)::volume_frame] / PD_RATE

    # smooth filter for PD data
    n = 100
    pd_info_raw = np.convolve(pd_info_raw, np.ones((n,)) / n, mode='same')
    temp = get_all_sti_pos(pd_info_raw > 0.02)
    sti_position_duration = []
    for cc in temp:
        if cc[1] > PD_RATE:
            sti_position_duration.append(cc)

    sti_ima_frame = []
    for cc in sti_position_duration:
        sti_ima_frame.append([find_nearest(frame_start,cc[0]),find_nearest(frame_start,cc[0]+cc[1])])

    fig, axs = plt.subplots(np.shape(dFF)[1]+1, 1, figsize=(20, 10), dpi=300, sharex=True)
    axs[0].set_title(parent.split('\\')[-2] + ' ' + parent.split('\\')[-1])
    ts = np.linspace(1,len(dFF),len(dFF)) / img_rate
    for i in range(np.shape(dFF)[1]):
        axs[i].plot(ts, dFF[:,i])
        axs[i].set_ylim( np.min(dFF[:,i]), np.max(dFF[:,i]) )
        for kk in sti_ima_frame:
            axs[i].add_patch(
                patches.Rectangle(
                    (kk[0] / img_rate, np.min(dFF[:,i])),
                    (kk[1]-kk[0]) / img_rate,
                    np.max(dFF[:,i]),
                    edgecolor=None,
                    facecolor='blue',
                    alpha = 0.3
                ))
        axs[i].set_ylabel('ROI' + str(i + 1) + ' _deltaF/F')
    axs[i].set_xlabel('t/s')
    plt.savefig(os.path.join(parent,'all_ROI.png'))
    plt.savefig(os.path.join(parent, 'all_ROI.pdf'))

    each_sti = []
    for kk in sti_ima_frame:
        each_sti.append((dFF[kk[0] - 100 : kk[1] + 100,:]).tolist())
        pass
    json.dump(each_sti, open(parent + "/each_sti.txt", "w"))




    ###########################################################################
    # m = load_memmap(glob(os.path.join(parent, "*.mmap"))[0])
    #
    # m2 = []
    # cmap = plt.cm.get_cmap("gray")
    # for i, mi in enumerate(m):
    #     img = mi.astype(np.uint8)#cv2.GaussianBlur(mi.astype(np.uint8), (3, 3), 0)
    #     m2.append(img)
    # write_video(parent + "/mc_sti.avi",(cmap(m2)[:,:,:,:3] * 255).astype(np.uint8),need_time=False,sti_img_frame=sti_ima_frame)




    # try:
    #     whole_sti_start_end = (dFF[sti_ima_frame[0][0] - 150 : sti_ima_frame[-1][-1] + 150]).tolist()
    #     whole_sti_pos = sti_ima_frame - sti_ima_frame[0][0] + 150
    # except:
    #     whole_sti_start_end = []
    #     whole_sti_pos = np.nan
    # json.dump( [whole_sti_start_end, whole_sti_pos.tolist()]
    #           , open(parent + "/whole_sti_start_end.txt", "w"))

    pass

def plot_dFF_MB_UVorLaser(dFF_name, exp_info, unify_rate=None, use_fictrac = False):
    parent = os.path.dirname(dFF_name)
    if use_fictrac:
        fictrac_name = get_fictrac_dat(parent)
        ft = pd.read_csv(fictrac_name, header=None).to_numpy()  # (15079frame, 23field) 50Hz
        bar_name = get_ft_bar(parent)
    pd_name = get_pd_h5(parent)

    dFF = pd.read_csv(dFF_name).to_numpy()[:, 1:]  # (2200frame, 16roi) 6.736Hz
    import h5py
    ni = h5py.File(pd_name, "r")
    pd_info_raw = ni['AI']["UVLED"][:, 0]  # the DAQ input of the matlab (UV or laser signal & fictrac trigger signal)
    frame_counter = ni['CI']["FrameCounter"][:, 0]
    frame_out = ni['DI']["FrameOut"][:, 0]
    img_rate = exp_info["frameRate"]  # only one z-step in this experiment
    if not unify_rate:
        unify_rate = img_rate
        use_img_rate = True
    seconds = len(pd_info_raw) / PD_RATE  # int(len(dFF) / img_rate)
    n = int(seconds * unify_rate)
    seconds = n / unify_rate

    frame_start = np.nonzero(np.diff(frame_counter) > 0)[0]

    if exp_info.get("zFastEnable"):
        img_rate = exp_info["frameRate"] / (exp_info["steps"] + exp_info["flybackFrames"])
        volume_frame = exp_info["steps"] + exp_info["flybackFrames"]
        frame_time = frame_start[int(exp_info["steps"] / 2)::volume_frame] / PD_RATE
        frame_start=frame_start[int(exp_info["steps"] / 2)::volume_frame]
    else:
        img_rate = exp_info["frameRate"]
        volume_frame = 1
        frame_time = frame_start / PD_RATE
    # volume_frame = 1
    # frame_time = frame_start[int(exp_info["steps"] / 2)::volume_frame] / PD_RATE

    #a = pd_info_raw > 0.04
    temp = get_all_sti_pos(pd_info_raw > 0.8)
    sti_position_duration = []
    for cc in temp:
        if cc[1] > PD_RATE/2:
            sti_position_duration.append(cc)

    sti_ima_frame = []
    for cc in sti_position_duration:
        sti_ima_frame.append([find_nearest(frame_start,cc[0]),find_nearest(frame_start,cc[0]+cc[1])])

    fig, axs = plt.subplots(np.shape(dFF)[1]+1, 1, figsize=(20, 10), dpi=300, sharex=True)
    axs[0].set_title(parent.split('\\')[-2] + ' ' + parent.split('\\')[-1])
    ts = np.linspace(1,len(dFF),len(dFF)) / img_rate
    for i in range(np.shape(dFF)[1]):
        axs[i].plot(ts, dFF[:,i])
        axs[i].set_ylim( np.min(dFF[:,i]), np.max(dFF[:,i]) )
        for kk in sti_ima_frame:
            axs[i].add_patch(
                patches.Rectangle(
                    (kk[0] / img_rate, np.min(dFF[:,i])),
                    (kk[1]-kk[0]) / img_rate,
                    np.max(dFF[:,i]),
                    edgecolor=None,
                    facecolor='blue',
                    alpha = 0.3
                ))
        axs[i].set_ylabel('ROI' + str(i + 1))
    axs[i].set_xlabel('t/s')
    plt.savefig(os.path.join(parent,'all_ROI.png'))
    # plt.savefig(os.path.join(parent, 'all_ROI.pdf'))

    each_sti = []
    for kk in sti_ima_frame:
        each_sti.append((dFF[kk[0] - int(img_rate * 2) : kk[1] + int(img_rate * 2),:]).tolist())
        pass
    json.dump(each_sti, open(parent + "/each_sti.txt", "w"))
    try:
        whole_sti_start_end = (dFF[sti_ima_frame[0][0] - 150 : sti_ima_frame[-1][-1] + 150]).tolist()
        whole_sti_pos = sti_ima_frame - sti_ima_frame[0][0] + 150
    except:
        whole_sti_start_end = whole_sti_pos = []
    json.dump( [whole_sti_start_end, whole_sti_pos.tolist()]
              , open(parent + "/whole_sti_start_end.txt", "w"))

    # ########################################################################### write video ######################################
    # m = load_memmap(glob(os.path.join(parent, "*.mmap"))[0])
    #
    # m2 = []
    # cmap = plt.cm.get_cmap("gray")
    # for i, mi in enumerate(m):
    #     img = mi.astype(np.uint8)#cv2.GaussianBlur(mi.astype(np.uint8), (3, 3), 0)
    #     m2.append(img)
    # write_video(parent + "/mc_sti.avi",(cmap(m2)[:,:,:,:3] * 255).astype(np.uint8),need_time=False,sti_img_frame=sti_ima_frame)


def plot_dFF_MB_fictrac(dFF_name, exp_info, unify_rate=None, use_fictrac = True):
    parent = os.path.dirname(dFF_name)
    if use_fictrac:
        fictrac_name = get_fictrac_dat(parent)
        ft = pd.read_csv(fictrac_name, header=None).to_numpy()  # (15079frame, 23field) 50Hz
        #bar_name = get_ft_bar(parent)
    pd_name = get_pd_h5(parent)
    dFF = pd.read_csv(dFF_name).to_numpy()[:, 1:]  # (2200frame, 16roi) 6.736Hz
    import h5py
    ni = h5py.File(pd_name, "r")
    pd_info_raw = ni['AI']["side_camera"][:, 0]  # the DAQ input of the matlab (UV or laser signal & fictrac trigger signal)
    frame_counter = ni['CI']["FrameCounter"][:, 0]
    frame_out = ni['DI']["FrameOut"][:, 0]

    frame_start = np.nonzero(np.diff(frame_counter) > 0)[0]
    if exp_info.get("zFastEnable"):
        img_rate = exp_info["frameRate"] / (exp_info["steps"] + exp_info["flybackFrames"])
        volume_frame = exp_info["steps"] + exp_info["flybackFrames"]
        frame_time = frame_start[int(exp_info["steps"] / 2)::volume_frame] / PD_RATE
        frame_start=frame_start[int(exp_info["steps"] / 2)::volume_frame]
    else:
        img_rate = exp_info["frameRate"]
        volume_frame = 1
        frame_time = frame_start / PD_RATE

    unify_rate = img_rate


      # "int(exp_info["steps"]/2)"  get the middle frame of each volumn

    #fictrac_start = np.nonzero(np.diff(pd_info_raw) > 1)[0]

    temp = get_all_sti_pos(pd_info_raw > 1)
    sti_position_duration = []
    for cc in temp:
        if cc[1] > 100:
            sti_position_duration.append(cc)

    fictrac_start = []
    for cc in sti_position_duration:
        fictrac_start.append(find_nearest(frame_start,cc[0]))

    fictrac_time = np.array(sti_position_duration)[:,0] / PD_RATE

    sti_ima_frame = [find_nearest(frame_start, sti_position_duration[0][0]), find_nearest(frame_start,sti_position_duration[-1][0])]

    dFF_unifyToFictrac = dFF[fictrac_start][0:-1]
    #Fictrac_data = ft[1:-2][:] # discard the first and last frame

    if len(dFF_unifyToFictrac) > len(ft):
        dFF_unifyToFictrac = dFF_unifyToFictrac[:len(ft)]
    else:
        ft = ft[:len(dFF_unifyToFictrac)]
    Fictrac_data = ft
    ft_speed = Fictrac_data[:, 18] * FICTRAC_RATE * BALL_RADIUS  # mm/s  all_speed
    ft_vy = -Fictrac_data[:, 5] * FICTRAC_RATE * BALL_RADIUS  # rad/s   lateral_speed
    ft_vx = Fictrac_data[:, 6] * FICTRAC_RATE * BALL_RADIUS  # rad/s   forward_speed
    ft_vz = -Fictrac_data[:, 7] * FICTRAC_RATE  # rad/s   rotation_speed


    for roi_index in range(np.shape(dFF)[1]):
        window = np.ones(int(10)) / float(10)
        dFF_unifyToFictrac_avg = np.convolve(dFF_unifyToFictrac[:,roi_index], window, 'same')
        ft_speed_avg = np.convolve(ft_speed, window, 'same')
        ft_vx_avg = np.convolve(ft_vx, window, 'same')
        ft_vy_avg = np.convolve(ft_vy, window, 'same')
        ft_vz_avg = np.convolve(ft_vz, window, 'same')
        ts = np.linspace(1, len(ft_speed_avg), len(ft_speed_avg)) / FICTRAC_RATE

        fig, axs = plt.subplots(4, 1, figsize=(20, 10), dpi=300, sharex=True)

        axs[0].set_title(os.path.basename(os.path.dirname(parent)) + os.path.basename(parent))
        axs[0].plot(ts, dFF_unifyToFictrac_avg, 'black')
        axs[0].set_ylabel('$\Delta$F/F')
        twin0 = axs[0].twinx()
        twin0.plot(ts, ft_speed_avg, 'red')
        twin0.set_ylabel('all speed mm/s', color='r', fontsize=8)

        axs[1].plot(ts, dFF_unifyToFictrac_avg, 'black')
        axs[1].set_ylabel('$\Delta$F/F')
        twin0 = axs[1].twinx()
        twin0.plot(ts, ft_vx_avg, 'g')
        twin0.set_ylabel('forward speed mm/s', color='g', fontsize=8)

        axs[2].plot(ts, dFF_unifyToFictrac_avg, 'black')
        axs[2].set_ylabel('$\Delta$F/F')
        twin0 = axs[2].twinx()
        twin0.plot(ts, ft_vy_avg, 'b')
        twin0.set_ylabel('lateral speed mm/s', color='b', fontsize=8)

        axs[3].plot(ts, dFF_unifyToFictrac_avg, 'black')
        axs[3].set_ylabel('$\Delta$F/F')
        twin0 = axs[3].twinx()
        twin0.plot(ts, ft_vz_avg, 'orange')
        twin0.set_ylabel('rotation speed rad/s', color='orange', fontsize=8)
        axs[3].set_xlabel('t/s')
        plt.savefig(os.path.join(os.path.dirname(dFF_name), dFF_name +str(roi_index+1)+ "_fictrac.png"))

    json.dump([dFF_unifyToFictrac.tolist(), ft_speed.tolist(), ft_vx.tolist(), ft_vy.tolist(), ft_vz.tolist()], open(parent + "/dFF_speed.txt", "w"))

    # seconds = len(pd_info_raw) / PD_RATE  # int(len(dFF) / img_rate)
    # n = int(seconds * unify_rate)
    # #dFF_info, dff_unify_idx = unify_sample(dFF, frame_time, unify_rate, n)
    # ft_info, ft_unify_idx = unify_sample(ft, fictrac_time - fictrac_time[0], unify_rate, n)

    pass


def plot_dFF_fictrac(dFF_name, exp_info, unify_rate=None):
    parent = os.path.dirname(dFF_name)
    fictrac_name = get_ft_dat(parent)
    bar_name = get_ft_bar(parent)
    pd_name = get_pd_h5(parent)
    ft = pd.read_csv(fictrac_name, header=None).to_numpy()  # (15079frame, 23field) 50Hz
    dFF = pd.read_csv(dFF_name).to_numpy()[:, 1:]  # (2200frame, 16roi) 6.736Hz
    import h5py
    ni = h5py.File(pd_name, "r")
    pd_info_raw = ni['AI']["photodiode"][:, 0]  # (1640499frame,) 5000Hz
    frame_counter = ni['CI']["FrameCounter"][:, 0]
    frame_out = ni['DI']["FrameOut"][:, 0]

    use_img_rate = False
    img_rate = exp_info["frameRate"]/(exp_info["steps"] + exp_info["flybackFrames"])
    if not unify_rate:
        unify_rate = img_rate
        use_img_rate = True

    seconds = len(pd_info_raw) / PD_RATE  #int(len(dFF) / img_rate)
    n = int(seconds * unify_rate)
    seconds = n / unify_rate
    frame_start = np.nonzero(np.diff(frame_counter) > 0)[0]
    volume_frame = exp_info["steps"] + exp_info["flybackFrames"]
    frame_time = frame_start[int(exp_info["steps"]/2)::volume_frame] / PD_RATE
    # frame_time = np.arange(frame_start[0]/PD_RATE, seconds, 1.0/img_rate)

    pd_info = down_sample(pd_info_raw[:int(seconds * PD_RATE)], n)
    pd_info_50 = down_sample(pd_info_raw[:int(seconds * PD_RATE)], int(seconds * 50))
    # frame_info = down_sample(frame_out[:int(seconds * PD_RATE)], n)/10.0
    dFF_info, dff_unify_idx = unify_sample(dFF, frame_time, unify_rate, n)
    ts = ft[:, -2]/1000
    if max(ts - ts[0]) <= 0:
        ts = np.arange(0, len(ts)) / FICTRAC_RATE
    ft_info, ft_unify_idx = unify_sample(ft, ts - ts[0], unify_rate, n)
    ft_bar_info = load_ft_bar(bar_name)
    if ft_bar_info is not None:
        ft_bar, ft_unify_idx = unify_sample(ft_bar_info, ts - ts[0], unify_rate, n)

    pd_thresh = 0.14
    ft_range = get_stim_range(pd_info_50 > pd_thresh)
    ft_range_t = np.array(ft_range[0]) / 50
    ft_range_u = (ft_range_t * unify_rate).astype(int)
    dff_range_t = [frame_time[0], frame_time[-1]]
    ft_unify_idx = np.concatenate([[0]*ft_range_u[0], ft_unify_idx]).astype(int)
    json.dump([ft_range_t.tolist(), dff_range_t, ft_unify_idx.tolist(), dff_unify_idx, ft_bar_info.tolist() if ft_bar_info is not None else []], open(dFF_name + "_time_info.txt", "w"))
    # exit(0)

    # ft_xs = np.arange(len(ft_info)) + ft_range_u[0]
    ft_heading = np.concatenate([[np.nan]*ft_range_u[0], ft_info[:, 16]])
    # ft_heading = lim_dir_l(ft_heading)
    if ft_bar_info is not None:
        ft_bar = np.concatenate([[np.nan]*ft_range_u[0], 2*np.pi - ft_bar])
        # ft_bar = lim_dir_l(ft_bar)
    else:
        ft_bar = ft_heading
    ft_speed = np.concatenate([[np.nan]*ft_range_u[0], ft_info[:, 18]]) * FICTRAC_RATE * BALL_RADIUS  # mm/s
    ft_vy = np.concatenate([[np.nan]*ft_range_u[0], -ft_info[:, 5]]) * FICTRAC_RATE  # rad/s
    ft_vx = np.concatenate([[np.nan]*ft_range_u[0], ft_info[:, 6]]) * FICTRAC_RATE
    ft_vz = np.concatenate([[np.nan]*ft_range_u[0], -ft_info[:, 7]]) * FICTRAC_RATE
    #########################################################################################################################################################
    all_speed, forward_speed, lateral_speed = bin_bump_speed(dFF_info,ft_speed,ft_vx,ft_vy,ft_vz)
    json.dump([all_speed, forward_speed, lateral_speed], open(parent + "/bin_bump_speed.txt", "w"))
    #return
    ###############################################################################################################################################################
    roi_n = dFF_info.shape[1]
    if use_img_rate:
        fig, axs = plt.subplots(roi_n, 1, figsize=(10, 30), dpi=300, sharex=True)
        pv_dir, pv_len = calc_pva(dFF_info)
        cor_s, cor_e = int(ft_range_u[0]+5*unify_rate), min(ft_range_u[1], len(ft_bar), len(pv_dir))

        ft_bar_cor = ft_bar[cor_s:cor_e]
        for i in range(roi_n):
            dffi = dFF_info[cor_s:cor_e, i]
            axs[i].scatter(ft_bar_cor, dffi, s=1)
        plt.savefig(parent + "/ft_pv_tune")
        #return

    fig, axs = plt.subplots(10, 1, figsize=(20, 10), dpi=300, sharex=True)
    plt.subplots_adjust(left=0.02, right=0.99)

    axs[0].pcolor(dFF_info.T, cmap="jet")
    is_pb = dFF_name.find("-PB-") > 0
    if is_pb:
        pv_dir, pv_len = calc_pva(dFF_info[:, :9])
        # pv_dir = lim_dir_l(np.array(pv_dir))
        pv_dir2, pv_len2 = calc_pva(dFF_info[:, 9:])
        # pv_dir2 = lim_dir_l(np.array(pv_dir2))
        # max_idx = np.argmax(dFF_info.T[:9], axis=0)
        # pv_dir = (max_idx + 0.5)*np.pi/4.5
        plot_angle(axs[0], (pv_dir+np.pi) * 4.5 / np.pi + 9, "k", 4.5)
    else:
        pv_dir, pv_len = calc_pva(dFF_info)
        # pv_dir = lim_dir_l(np.array(pv_dir))
        pv_dir += np.pi
        pv_dir2, pv_len2 = pv_dir, pv_len
        # max_idx = np.argmax(dFF_info.T, axis=0)
        # pv_dir = (max_idx + 0.5)*np.pi/8
        pi = roi_n / 2
        # plot_angle(axs[0], (np.array(pv_dir) + np.pi) * pi / np.pi, "k", pi)
        plot_angle(axs[0], pv_dir * pi / np.pi, "k", pi)

    # axs[1].plot(frame_info, alpha=0.2, c="k")
    axs[1].plot(pd_info, c="b")

    plot_angle(axs[2], pv_dir, c="k")
    # if is_pb:
    #     ft_bar = -ft_bar
    #     plot_angle(axs[2], pv_dir2, c="gray")
    plot_angle(axs[2], ft_bar, "b")  # bar
    # plot_angle(axs[2], ft_heading, "g")  # heading
    axs[2].set_ylim(0, 2*np.pi)
    # axs[2].set_ylim(-np.pi, np.pi)
    # axs[2].add_patch(plt.Rectangle((0, -np.pi), n, np.pi / 4, alpha=0.1, color="k", linewidth=0))
    # axs[2].add_patch(plt.Rectangle((0, 3*np.pi/4), n, np.pi / 4, alpha=0.1, color="k", linewidth=0))
    axs[2].add_patch(plt.Rectangle((0, 0), n, np.pi / 4, alpha=0.1, color="k", linewidth=0))
    axs[2].add_patch(plt.Rectangle((0, 7*np.pi/4), n, np.pi / 4, alpha=0.1, color="k", linewidth=0))

    offset = calc_offset(ft_bar, pv_dir)
    axs[3].plot(offset)

    # plot_angle(axs[4], lim_dir_l(pv_dir), c="k")
    ft_bar_correct = lim_dir_l(ft_bar - circmean(non_nan(offset)))
    ft_bar_correct[ft_bar_correct < 0] += 2 * np.pi
    plot_angle(axs[0], ft_bar_correct / (2*np.pi) * roi_n, "r")  # heading - offset
    # axs[4].set_ylim(-np.pi, np.pi)

    plot_scatter(axs[5], unwrap_dir_win(ft_bar, unify_rate), "b")
    plot_scatter(axs[5], unwrap_dir_win(pv_dir, unify_rate), "k")
    axs[5].set_ylim(-6, 6)

    axs[6].plot(unwrap_dir(ft_bar), c="b", lw=0.5)
    axs[6].plot(unwrap_dir(ft_heading), c="g", lw=0.5)
    axs[6].plot(unwrap_dir(smooth_angle(pv_dir, 10)), "k--", lw=0.5)
    axs[6].plot(unwrap_dir(pv_dir), c="k", lw=0.5)

    plot_slide_cor(axs[-3], ft_bar, pv_dir, unify_rate)
    axs[-3].set_ylim(-1, 1)

    axs[-2].plot(ft_speed, c="r")  # speed
    axs[-1].plot(zscore1(ft_speed), c="r", alpha=0.5)
    axs[-1].plot(zscore1(pv_len), c="k", alpha=0.5)

    # plt.show()
    ticks = np.arange(0, len(pd_info), 5*unify_rate)
    axs[-1].set_xticks(ticks)
    axs[-1].set_xticklabels(["%.1f"%l for l in ticks/unify_rate])

    cor_s, cor_e = int(ft_range_u[0]+5*unify_rate), min(ft_range_u[1], len(ft_bar), len(pv_dir))
    if cor_s > cor_e:
        print("error: ft range not found!")
        exit(0)
    ft, pv = ft_bar[cor_s:cor_e], pv_dir[cor_s:cor_e]
    cor_unwrap = cor(unwrap_dir(ft), unwrap_dir(pv))
    cor_cir = cir_cor(ft, pv)
    axs[0].set_title("%s fps: %.2f, speed: %.2f, dFF (%s, %s), fictrac (%s, %s), dFF~heading: %.2f (unwrap) %.2f (circ)" % (
        os.path.basename(parent), unify_rate, np.nanmean(ft_speed),
        format_time(dff_range_t[0]), format_time(dff_range_t[1]), format_time(ft_range_t[0]), format_time(ft_range_t[1]),
        cor_unwrap, cor_cir
    ))
    plt.savefig(os.path.join(os.path.dirname(dFF_name), dFF_name + "_fictrac.png"))
    json.dump({"cor_unwrap": cor_unwrap, "cor_cir": cor_cir}, open(parent + "/info.txt", "w"))
    ft_pv = [
        [cor_s/unify_rate, cor_e/unify_rate], ft.tolist(), pv.tolist(), pv_len[cor_s:cor_e].tolist(), ft_speed[cor_s:cor_e].tolist(),
        ft_vx[cor_s:cor_e].tolist(), ft_vy[cor_s:cor_e].tolist(), ft_vz[cor_s:cor_e].tolist()
    ]
    if pv_dir2 is not None:
        ft_pv.append(pv_dir2[cor_s:cor_e].tolist())
        ft_pv.append(pv_len2[cor_s:cor_e].tolist())
    json.dump(ft_pv, open(parent + "/ft_pv.txt", "w"))
    json.dump(offset.tolist(),open(parent + "/offset.txt", "w"))


    plt.figure()
    plt.hist(offset, bins=50)
    plt.title("%s fps: %.2f, mean: %.2f SD: %.2f cSD: %.2f" % (
        os.path.basename(parent), unify_rate,
        np.nanmean(offset), np.nanstd(offset), circular_std(offset)
    ))
    plt.savefig(os.path.join(os.path.dirname(dFF_name), dFF_name + "_offset.png"))

    # ft_win = window * FICTRAC_RATE
    # dFF_win = window * IMG_RATE
    # s = int(min(len(ft) / ft_win, len(dFF) / dFF_win))
    # cor_dir, cor_speed = [], []
    # for i in range(s):
    #     ft_s = ft[int(i * ft_win): int((i+1) * ft_win)]
    #     dFF_s = dFF[int(i * dFF_win): int((i+1) * dFF_win)]
    #     pv_dir, pv_len = calc_pva(dFF_s)
    #     # plt.figure("pv_dir");plt.plot(unwrap_dir(pv_dir))
    #     # plt.figure("heading");plt.plot(unwrap_dir(ft_s[:, 16]))
    #     # plt.figure("pv_len");plt.plot(pv_len)
    #     # plt.figure("speed");plt.plot(ft_s[:, 18])
    #     # plt.show()
    #     cor_dir.append(cor_equal_len(unwrap_dir(pv_dir), unwrap_dir(ft_s[:, 16])))
    #     cor_speed.append(cor_equal_len(pv_len, ft_s[:, 18]))
    # plt.plot(cor_dir)
    # plt.plot(cor_speed)
    # plt.show()

def plot_rois(rois, m):
    plt.figure(figsize=(3, 3), dpi=300)
    plt.axis("off")
    ax = plt.gca()
    ax.set_position([0, 0, 1, 1], which="both")
    ax.imshow(norm_img(np.std(m, axis=0)), cmap=plt.cm.gray)
    for i, xys in enumerate(rois):
        ax.add_patch(plt.Polygon(xys, alpha=0.6, fill=False, linewidth=1, color=COLORS[i%12]))
        x, y = xys.min(axis=0)/2 + xys.max(axis=0)/2
        ax.text(x, y, str(i), color="r")

def get_part(fname):
    if fname.find("-PB-") >= 0:
        return "PB"
    elif fname.find("-FB-") >= 0:
        return "FB"
    elif fname.find("MB") >= 0:
        return "manual draw"
    elif fname.find("57C10") >= 0:
        return "manual draw"
    elif fname.find("D161") >= 0:
        return "manual draw"
    elif fname.find("BU") >= 0:
        return "manual draw"
    elif fname.find("AOTU") >= 0:
        return "manual draw"
    elif fname.find("EB") >= 0:
        # return "manual draw"
        return "EB"
    else:
        return "manual draw"

def plot_fictrac(fname, parent, win=10, step=1, speed_min=0.5):
    df = pd.read_csv(fname, header=None).to_numpy().T
    heading = df[16]
    speed = df[18]
    inv = speed*FICTRAC_RATE < speed_min
    heading[inv] = np.nan
    stand = (np.count_nonzero(inv) / len(heading))
    print("stand still %.2f" % stand)
    vec_x, vec_y = np.cos(df[16]), np.sin(df[16])
    mx, my = [], []
    for i in range(0, len(heading) - win * FICTRAC_RATE, step * FICTRAC_RATE):
        mx.append(np.nanmean(vec_x[i: i+win * FICTRAC_RATE]))
        my.append(np.nanmean(vec_y[i: i+win * FICTRAC_RATE]))
    mean_vec = np.nanmean(np.sqrt(np.array(mx)**2+np.array(my)**2))
    plt.figure()
    plt.title("walk %.2f, mean %.2f" % (1-stand, mean_vec))
    plt.axis("equal")
    plt.hist2d(mx, my, bins=20, range=((-1, 1), (-1, 1)))
    plt.savefig(parent + "_mean_vec.png")
    plt.figure()
    plt.plot(speed*FICTRAC_RATE)
    plt.savefig(fname + "_speed.png")
    return

def get_pd_h5(parent):
    pp = os.path.dirname(parent)
    #if (not parent[-1].isdigit()) or parent[-1] == 'D' or parent[-1] == 'g' or parent[-1] == 'r' or parent[-1] == 'V' or parent[-6:] == 'MBON11':n
    if (not parent[-3:].isdigit()):
        ppn = os.path.basename(parent).rstrip("+") + '_000'
    else:
        ppn = os.path.basename(parent).rstrip("-").rstrip("+").replace("_0", "_0")
    return os.path.join(pp, "PD_" + ppn + "/Episode001.h5")

def get_fictrac_dat(parent):
    file_folder = os.path.basename(parent)
    temp = file_folder.split('_')
    if len(temp) == 5:
        data_folder = temp[2][0:2] + temp[3][0] + '-0'
    elif len(temp) == 6:
        data_folder = temp[2][0:2] + temp[3][0] + '-' + str(int(temp[-1]))

    if temp[0] == 'D161':
        data_folder = data_folder.replace('L','')
    if temp[0] ==  'VC102':
        # if len(temp) == 5:
        #     data_folder = temp[2][0:2] + temp[3] + '-0'
        # elif len(temp) == 6:
        #     data_folder = temp[2][0:2] + temp[3] + '-' + str(int(temp[-1]))
        if 'R' in data_folder:
            data_folder = data_folder.replace('R', '')
        elif 'L' in data_folder:
            data_folder = data_folder.replace('L', '')
    return glob(os.path.join(os.path.dirname(parent), data_folder)+ "/*.dat")[0]
    pass

def get_ft_dat(parent):
    pp = os.path.dirname(parent)
    ppn = os.path.basename(parent).rstrip("-").rstrip("+")
    #ppn = os.path.basename(parent)
    tt = ppn.split("-")
    if len(tt) > 2:
        return glob(os.path.join(pp, tt[2] + "-" + tt[-1]) + "/*.dat")[0]
    else:
        t3 = tt[0].split("_F")
        t4 = t3[1].split("_")
        t5 = t4[-1].split("-")
        if len(tt) == 2:
            n = tt[-1]
        else:
            n = t5[-1]
        if n == 'CLOSED':
            n = '000'
        return glob(os.path.join(pp, "F" + t4[0] + "-" + str(int(n))) + "/*.dat")[0]


def get_ft_bar(parent):
    return os.path.join(os.path.dirname(get_ft_dat(parent)), "bar_position.mat")

def write_fictrac_stim_dFF_video(parent):
    ft_name = get_ft_dat(parent)
    ft_folder = os.path.dirname(ft_name)
    ft_range_t, dff_range, ft_idx, dff_idx, ft_bar_info = json.load(open(parent + "/zscore.csv_time_info.txt", "r"))
    m = load_memmap(glob(os.path.join(parent, "*.mmap"))[0])
    m_color = m[dff_idx]

    c = json.load(open(parent + "/ft_pv.txt", "r")) #50Hz, fictrac start +5s
    t_range, heading, pv, pvl, speed, vx, vy, vx, pv2, pvl2 = c
    pva = np.concatenate([[np.nan] * int((ft_range_t[0]+5)*50), pv])  # after proc_csv
    names, points, contours = load_roi(parent + "/roi.npy", m[0].shape)
    roi_n = len(contours) - 1  # 16 or 18
    pv_contour_idx = ((pva*roi_n/(2*np.pi)) - 0.5).astype(int) + 1  # (0~16)->(0-15)+1
    write_zscore_video(m_color, parent + "/v_color.avi", pv_contour_idx, contours)
    # return



    if not ft_bar_info:
        df = pd.read_csv(ft_name, header=None).to_numpy().T
        heading = 2*np.pi - df[16]
    else:
        heading = ft_bar_info
    m_stim = []
    w, h = 320, 160
    barh = 5
    for d in heading:
        img = np.zeros((h, w), dtype=np.uint8)
        mid = int(d * w / (2*np.pi))
        img[:, mid - barh:mid + barh] = 255
        if mid < barh:
            img[:, 0 :mid + barh] = 255
            img[:, w-(barh-mid):w] = 255
        if mid + barh >= w:
            img[:, mid:w] = 255
            img[:,0:mid+barh-w] = 255
        m_stim.append(img)
    write_video(parent + "/v_stim.avi", np.array(m_stim)[ft_idx], fps=FICTRAC_RATE)
    #embed_video(ft_folder + "/fictrac-debug.avi", parent + "/v_stim.avi", parent + "/v_color.avi", ft_range_t[0], ft_range_t[0])
    embed_video(os.path.join(ft_folder,"fictrac-debug.avi"), os.path.join(parent,'v_stim.avi') , os.path.join(parent,'v_color.avi'), ft_range_t[0],
                ft_range_t[0])

def realtime_heading_PVA(frames, heading, pv,parent):
    filenames = []
    for i in range(0,len(heading),200):

        plt.figure(figsize=(30, 3))
        if i > 100:
            plt.plot(heading[i-100:i])
            plt.plot(pv[i-100:i])
        else:
            plt.plot(heading[0:i])
            plt.plot(pv[0:i])
        filename = f'{i}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter(os.path.join(parent,'heading_pva.avi'), fourcc, 10, (3000,300))
    for item in filenames:
        img = cv2.imread(item)
        video.write(img)
    video.release()

    for filename in set(filenames):
        os.remove(filename)
    pass


def plot_dFF(dff_csv, fr):
    dFF = pd.read_csv(dff_csv)
    names = dFF.columns
    print(names)

    plot_lines([dFF[n] for n in names], names)
    plt.savefig(dff_csv + ".png")
    plot_hot(dFF[dFF.keys()[1:]], dff_csv + "_hot", fr, is_PB=(dff_csv.find("-PB-") > 0))
    # plot_lines([auto_cor(dFF[n][200:]) for n in names], names, ylim=(-0.2, 0.5), xlim_r=0.505)
    # plt.savefig(parent + "/dFF_cor.png")

def get_dFF_by_baseline_range(F, baseline_range, bg_idx=0):
    names = F.columns
    bg = F[names[bg_idx]].copy()
    for g in names:
        F[g] = F[g] - bg
        baseline = np.mean(F[g][baseline_range[0]:baseline_range[1]])
        F[g] = (F[g] - baseline) / baseline
    return F

def get_dFF(F, bg_correction = True):
    names = F.columns
    bg = F[names[0]]
    for g in names[1:]:
        if np.mean(bg) < np.mean(F[g]) + 5:
            if bg_correction:
                f = F[g] - bg
            else:
                f = F[g] - np.mean(bg)
        else:
            f = F[g]
        f = f.mask(f < 0, np.nan)
        baseline = np.nanmean(f[f <= np.nanpercentile(f, 5)])  # NOTE: F as the mean of the lower 5% (MaimonG_Nat17)
        if baseline <= 1:
            baseline = 1
        print(g, "baseline", baseline)
        F[g] = (f - baseline) / baseline
    return F

def get_zscore(F):
    from scipy.stats import zscore
    zs = zscore(F, axis=0)
    return pd.DataFrame(zs, columns=F.columns)

def zscore1(xs):
    m, s = np.nanmean(xs), np.nanstd(xs)
    return (xs - m) / s

def cov1(xs, ys):
    n = len(xs)
    mx, my = np.mean(xs), np.mean(ys)
    return (xs - mx).dot(ys - my) / (n - 1)

def cor1(xs, ys):
    n = len(xs)
    return zscore1(xs).dot(zscore1(ys)) / (n - 1)

def cor2(xs, ys):
    return cov1(xs, ys) / np.std(xs) / np.std(ys)

def cir_cor(xs, ys): # circstat circ_corrcc
    mx, my = circmean(xs), circmean(ys)
    num = np.sin(xs - mx).dot(np.sin(ys - my))
    den = np.sqrt(np.sum(np.sin(xs - mx) ** 2) * np.sum(np.sin(ys - my) ** 2))
    return num / den

def auto_cor(xs):
    xm = xs - xs.mean()
    xn = np.sum(xm ** 2)
    return np.correlate(xm, xm, "same") / xn

def cor(x, y):
    # return np.dot(x, y)/np.count_nonzero(y>0.5)
    return np.corrcoef(x, y)[0][1]

def cor_equal_len(x, y):
    if len(x) > len(y):
        return cor(down_sample(x, len(y)), y)
    else:
        return cor(x, down_sample(y, len(x)))

def slide_cor(x, y, win, rate):
    n = min(len(x), len(y))
    rx, ry = [], []
    for i in range(0, n - win, int(rate)):
        c = cor(unwrap_dir(x[i:i + win]), unwrap_dir(y[i:i + win]))
        rx.append(i + win/2.0)
        ry.append(c)
    return rx, ry

def down_sample(x, n):
    x = np.array(x, dtype=np.float)
    index_arr = np.linspace(0, len(x) - 1, num=n, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int)
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor

    val1 = x[index_floor]
    val2 = x[index_ceil % len(x)]
    interp = val1 * (1.0 - index_rem) + val2 * index_rem
    assert (len(interp) == n)
    return interp

def view_img_seq(imgs, shifts_rig=None):
    global g_total_frame, g_frame, g_is_input_begin, g_input_int
    g_total_frame, h, w = imgs.shape
    g_frame = 0
    g_is_input_begin = False
    g_input_int = 0

    def plot_one_frame(f):
        global g_frame
        g_ax.cla()
        g_ax.imshow(imgs[f].astype(int), cmap=plt.cm.gray, norm=NoNorm())
        # g_ax.set_xlabel("%02d:%02.2f" % (t_sec / 60, t_sec % 60))
        g_ax.set_xlabel(str(f))
        if shifts_rig is not None:
            g_ax.set_title(str(shifts_rig[g_frame]))
        g_ax.grid(True)
        g_ax.set_xticks(np.linspace(0, imgs[f].shape[1], 13))
        g_frame = f
    def on_slider(val):
        plot_one_frame(int(val))
    def onkey(event):
        print(event.key)
        global g_frame, g_is_input_begin, g_input_int
        if event.key == "left":
            g_frame -= 1
        elif event.key == "right":
            g_frame += 1
        elif event.key == "enter":
            g_frame = g_input_int
            g_input_int = 0
        elif event.key in list([*"1234567890"]):
            g_input_int = int(event.key) + g_input_int * 10
            print("input: %d" % g_input_int)
            return
        else:
            g_input_int = 0
            return
        if g_frame >= g_total_frame:
            g_frame = g_total_frame - 1
        if g_frame < 0:
            g_frame = 0
        g_slider.set_val(g_frame)
        event.canvas.draw()

    from matplotlib.widgets import Slider
    fig, g_ax = plt.subplots(figsize=(w/25, h/25))
    plt.subplots_adjust(top=0.95, bottom=0.1)
    g_slider = Slider(plt.axes([0.1, 0.03, 0.8, 0.03]), "", valmin=0, valmax=g_total_frame - 1, valfmt="%d", valinit=0)
    g_slider.on_changed(on_slider)
    plot_one_frame(0)
    fig.canvas.mpl_connect('key_press_event', onkey)

def load_exp_xml(xml):
    if not os.path.exists(xml):
        return None
    def get_value(ss, key):
        p = ss.find(key + "=")
        return ss[p:].split("\"")[1] if p >= 0 else None
    s = open(xml, "r").readlines()
    ret = {}
    for ss in s:
        if ss.find("ThorZPiezo") >= 0:
            ret["steps"] = int(get_value(ss, "steps"))
            ret["stepSizeUM"] = float(get_value(ss, "stepSizeUM"))
            ret["startPos"] = float(get_value(ss, "startPos"))
        elif ss.find("<Streaming") >= 0:
            ret["frames"] = int(get_value(ss, "frames"))
            ret["zFastEnable"] = int(get_value(ss, "zFastEnable"))
            ret["flybackFrames"] = int(get_value(ss, "flybackFrames"))
        elif ss.find("<LSM") >= 0:
            ret["name"] = get_value(ss, "name")
            ret["pixelX"] = int(get_value(ss, "pixelX"))
            ret["pixelY"] = int(get_value(ss, "pixelY"))
            ret["pixelSizeUM"] = float(get_value(ss, "pixelSizeUM"))
            ret["frameRate"] = float(get_value(ss, "frameRate")) #167.569
    return ret

def load_ft_bar(bar_name):
    if not os.path.exists(bar_name):
        return None
    import scipy.io as sio
    bar_pos = sio.loadmat(bar_name)["bar_position"][0]
    p = np.min(np.nonzero(np.isnan(bar_pos)))
    return (bar_pos[:p] - 64) / 1648 * 2 * np.pi  # 64~1712

def write_video(path, m, cvt=cv2.COLOR_GRAY2BGR, fps=30, need_time=True, sti_img_frame = None):
    h, w = m[0].shape[:2]
    output_video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"DIVX"), fps, (w, h))
    for i, mi in enumerate(m):
        #img_bgr = cv2.cvtColor(mi, cvt)
        img_bgr = mi
        if need_time:
            cv2.putText(img_bgr, format_time(i/fps), (6, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
        if sti_img_frame is not None:
            for j in range(len(sti_img_frame)):
                if i>sti_img_frame[j][0] and i < sti_img_frame[j][1]:
                    img = np.ones((20, 20,3), dtype=np.uint8) * 255
                    img_bgr[0:20, 0:20, :] = img
                pass
        output_video.write(img_bgr)
    output_video.release()

def embed_video(v1, v2, v3, t2, t3, offset2=(0, 320), offset3=(160, 320)):
    cap1 = cv2.VideoCapture(v1)
    cap2 = cv2.VideoCapture(v2)
    cap3 = cv2.VideoCapture(v3)
    n1 = int(cap3.get(cv2.CAP_PROP_FRAME_COUNT))
    cap2.set(cv2.CAP_PROP_POS_FRAMES, t2 * FICTRAC_RATE)
    cap3.set(cv2.CAP_PROP_POS_FRAMES, t3 * FICTRAC_RATE)
    w, h = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(v2 + "_combine.avi", cv2.VideoWriter_fourcc(*"DIVX"), FICTRAC_RATE, (w, h))
    for seq in range(0, n1):
        ret, img1 = cap1.read()
        if not ret:
            break
        ret, img2 = cap2.read()
        if not ret:
            break
        ret, img3 = cap3.read()
        img3 = resize_image(img3,320,320)
        if not ret:
            break
        h, w, c = img2.shape
        img1[offset2[0]:offset2[0]+h, offset2[1]:offset2[1]+w, :] = img2
        h, w, c = img3.shape
        img1[offset3[0]:offset3[0]+h, offset3[1]:offset3[1]+w, :] = img3
        output_video.write(img1)
    output_video.release()

def resize_image(image, width, height,COLOUR=[0,0,0]):
    h, w, layers = image.shape
    if h > height:
        ratio = height/h
        image = cv2.resize(image,(int(image.shape[1]*ratio),int(image.shape[0]*ratio)))
    h, w, layers = image.shape
    if w > width:
        ratio = width/w
        image = cv2.resize(image,(int(image.shape[1]*ratio),int(image.shape[0]*ratio)))
    h, w, layers = image.shape
    if h < height and w < width:
        hless = height/h
        wless = width/w
        if(hless < wless):
            image = cv2.resize(image, (int(image.shape[1] * hless), int(image.shape[0] * hless)))
        else:
            image = cv2.resize(image, (int(image.shape[1] * wless), int(image.shape[0] * wless)))
    h, w, layers = image.shape
    if h < height:
        df = height - h
        df /= 2
        image = cv2.copyMakeBorder(image, int(df), int(df), 0, 0, cv2.BORDER_CONSTANT, value=COLOUR)
    if w < width:
        df = width - w
        df /= 2
        image = cv2.copyMakeBorder(image, 0, 0, int(df), int(df), cv2.BORDER_CONSTANT, value=COLOUR)
    image = cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
    return image


def merge_z_slices(fname, n, method=None):
    import tifffile
    with tifffile.TiffFile(fname) as tffl:
        input_arr = tffl.asarray()  # 25200*128*128
        # write_video(fname + ".avi", norm_img(input_arr))
        # plt.figure()
        # a = input_arr[0].flatten()
        # plt.hist(a[a>0.01], bins=50)
        # plt.savefig(fname + "_hist.png")
        chs = []
        tn = int(len(input_arr) / n)
        #fname_ch_list = []
        for i in range(n):
            ch0 = input_arr[i:tn*n:n]
            # avg_frame = np.mean(ch0, axis=0)
            # w = 255.0 / np.max(avg_frame)
            # print(w)
            eq = ch0 #* w*w
            # eq[eq > 0] = 255
            chs.append(eq)
            #tifffile.imwrite(fname + "_ch%d.tif" % i, eq)
            #fname_ch_list.append(fname + "_ch%d.tif" % i)
            # write_video(fname + "_ch%d.avi"%i, norm_img(eq))
            cv2.imwrite(fname + "_avg%d.png" % i, norm_img(np.mean(eq, axis=0)))
        volu_arr = input_arr.reshape((np.shape(ch0)[0],np.shape(ch0)[1],np.shape(ch0)[2],n))
        tifffile.imwrite(fname + "_ch_all.tif", volu_arr)
        if method == "max":
            avg_tif = np.max(chs, axis=0)
        else:
            avg_tif = np.mean(chs, axis=0)
        tifffile.imwrite(fname + "_avg.tif", avg_tif)
        return fname + "_avg.tif", fname + "_ch_all.tif"

def merge_info(date_folder):
    info_l = []
    for f in glob(date_folder + "/*/info.txt"):
        c = json.load(open(f, "r"))
        print(f, c)
        exp_name = os.path.basename(os.path.dirname(f))
        t = exp_name.split("-")
        pair = "-".join(t[1:-1])
        trail = t[-1]
        stim = t[-2]
        part = t[-3]
        fly = t[2]
        info_l.append([pair, fly, trail, part, stim, c["cor_unwrap"], c["cor_cir"]])
    df = pd.DataFrame.from_records(info_l, columns=["pair", "fly", "trail", "part", "stim", "cor_unwrap", "cor_cir"])
    import seaborn as sns
    for x in ["cor_unwrap", "cor_cir"]:
        plt.figure()
        sns.set_theme(style="darkgrid")
        sns.catplot(x=x, y="stim", hue="fly", col="part", data=df)
        plt.xlim(-1, 1)
        plt.tight_layout()
        plt.savefig(date_folder + "/" + x)

def smooth_angle(s, win):
    hw = int(win/2)
    s1 = np.concatenate([[s[0]] * hw, s, [s[-1]] * hw])
    ret = []
    for i in range(len(s)):
        m = circmean(s1[i:i+win])
        if m > np.pi:
            m -= np.pi * 2
        ret.append(m)
    return ret

def proc_ft_pv(fname):
    c = json.load(open(fname, "r"))
    t_range, heading, pv, pvl, speed, vx, vy, vx, pv2, pvl2 = c
    n = len(heading)
    ts = np.linspace(t_range[0], t_range[1], n)
    # plot_lines(c[:7], "heading pv pvl speed vx vy vz".split())
    pvs = smooth_angle(pv, 10)
    pvs2 = smooth_angle(pv2, 10)

    fig, axs = plt.subplots(3, 1, figsize=(8, 4), sharex=True, dpi=300)
    # plt.subplots_adjust(left=0.02, right=0.99)
    axs[0].plot(ts, unwrap_dir(np.array(heading)), c="m", lw=0.5)
    axs[0].plot(ts, unwrap_dir(pvs), "k--", lw=0.5)
    axs[0].plot(ts, unwrap_dir(pv), c="k", lw=0.5)
    # axs[0].plot(ts, unwrap_dir(pvs2), c="gray", lw=0.5)
    print(cor(unwrap_dir(np.array(heading)), unwrap_dir(pvs)))
    # axs[0].plot(ts, unwrap_dir(pv2), c="gray", lw=0.5)
    # axs[1].plot(ts, -np.array(heading), c="m", lw=0.5)
    axs[1].plot(ts, pv, c="k", lw=0.5)
    # axs[1].plot(ts, pvs, c="g", lw=0.5)
    axs[1].plot(ts, pv2, c="gray", lw=0.5)
    axs[2].plot(ts, zscore1(speed), c="m", lw=0.5)
    axs[2].plot(ts, zscore1(pvl), c="k", lw=0.5)
    axs[2].plot(ts, zscore1(pvl2), c="gray", lw=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig(fname + ".png")
"""
// frame_count
ss << _cnt << ", ";
// rel_vec_cam[3] | error
ss << _dr_cam[0] << ", " << _dr_cam[1] << ", " << _dr_cam[2] << ", " << _err << ", ";
// rel_vec_world[3]
ss << _dr_lab[0] << ", " << _dr_lab[1] << ", " << _dr_lab[2] << ", ";
// abs_vec_cam[3]
ss << _r_cam[0] << ", " << _r_cam[1] << ", " << _r_cam[2] << ", ";
// abs_vec_world[3]
ss << _r_lab[0] << ", " << _r_lab[1] << ", " << _r_lab[2] << ", ";
// integrated xpos | integrated ypos | integrated heading
ss << _posx << ", " << _posy << ", " << _heading << ", ";
// direction (radians) | speed (radians/frame)
ss << _step_dir << ", " << _step_mag << ", ";
// integrated x movement | integrated y movement (mouse output equivalent)
ss << _intx << ", " << _inty << ", ";
// timestamp | sequence number
ss << _ts << ", " << _seq << std::endl;
"""

if __name__ == '__main__':
    ft, pv = json.load(open(r"D:\exp_2p\data\EPG\210823\CX1001-7F-F6-PB-reverseCLOSED-7+\ft_pv.txt", "r"))
    print(cor(ft, pv))
    print(cor(unwrap_dir(ft), unwrap_dir(pv)))
    print(cir_cor(ft, pv))
