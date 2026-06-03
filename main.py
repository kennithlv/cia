import time

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps
# from caiman.source_extraction import cnmf

import caiman as cm
from caiman.motion_correction import MotionCorrect
# from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
# from caiman.summary_images import local_correlations_movie_offline
import glob
import sys
import os
import pandas as pd
import gc
from _3D_registration import load_stim_txt, get_exp_info, SYNC_RATE, correct_ft_lost_idx, interpolation_lost_frame
from cia_utils import *
from cia_roi_template import ROIModifyUI
from joblib import Parallel, delayed

from plot_utils import align_sti_length, get_min_sti_length

USE_AUTO_ROI = False
img_rate, exp_info, parent = None, None, None


def main_init(fname):
    print(fname)
    global img_rate, exp_info, parent
    parent = os.path.dirname(fname)
    exp_info = load_exp_xml(os.path.join(parent, "Experiment.xml"))
    print(exp_info)
    try:
        img_rate = exp_info["frameRate"]
        if exp_info.get("zFastEnable"):
            img_rate /= exp_info["steps"] + exp_info["flybackFrames"]
    except:
        pass



def proc_tif(fname):
    from cia_caiman import cm_motion_correction

    if (exp_info is not None) and exp_info.get("zFastEnable") and fname.find("tif_ch") < 0:
        fname, fname_ch_all = merge_z_slices(fname, exp_info["steps"], method="max")
        fname = cm_motion_correction([fname_ch_all], parent, save_avi=False, is3D=True)

    else:
        # fname = merge_z_slices(fname, 1)
        fname = cm_motion_correction([fname], parent, save_avi=True)
    # return view_mc([fname], glob.glob(parent + "/*.mmap")[0])

    return fname


def proc_csv(fname):
    plot_dFF_MB_grating(fname, exp_info)
    # plot_dFF(fname, img_rate)
    # plot_dFF_fictrac(fname, exp_info)
    # plot_dFF_fictrac(fname, exp_info, 50)
    # write_fictrac_stim_dFF_video(parent)


last_roi_info = {}


def proc_mmap(fname, only_roi=False, only_roi_change = False):
    # return calc_avg_frame(load_memmap(fname), parent)
    global last_roi_info
    plt.ioff()
    map = load_memmap(fname)

    if exp_info and exp_info.get("zFastEnable"):
        map_shape = np.shape(map)
        map = np.reshape(map,(map_shape[0],map_shape[3],map_shape[1],map_shape[2]))   ############ fix me!   why ???###############
        map = np.mean(map,1)  # mean z slices

    # write_zscore_video(map, fname)
    # return
    if USE_AUTO_ROI:
        calc_avg_frame(map, parent)
        from cia_caiman import cm_detect_roi
        cm_detect_roi(fname)
        dff_csv = parent + "/dFF.csv"
        plot_dFF_MB_UVorLaser(dff_csv, exp_info)

    else:
        roi_file = parent + "/roi.npy"

        # roi_file = parent + "/RoiSet.zip"
        calc_avg_frame(map, parent)
        if only_roi_change:
            plt.close("all")
            part = get_part(fname)
            temp = last_roi_info.get(part)
            if temp is None:
                ROIModifyUI(roi_file, parent + "/i_std.png").show()
            else:
                ROIModifyUI(temp, parent + "/i_std.png").show()
            last_roi_info = {part: roi_file}
            # dff_csv, zs_csv = calc_all_roi_F(roi_file, map, parent)
            # plot_dFF_MB_UVorLaser(dff_csv, exp_info)
            return
        if not os.path.exists(roi_file):
            plt.close("all")
            part = get_part(fname)
            temp = last_roi_info.get(part)
            if temp is None:
                temp = r"D:\LQT\LQTgithub\code_2p\roi_templates/%s.npy" % part
            ROIModifyUI(temp, parent + "/i_std.png").show()
            last_roi_info = {part: roi_file}
        if not only_roi:
            dff_csv, zs_csv = calc_all_roi_F(roi_file, map, parent)
            if exp_info:
                plot_trial_F(parent, parent + "/F.csv",save_sti_startframe=True)
                plot_trial(parent, dff_csv,save_sti_startframe=True)


def plot_trial_F(trial_folder,roi_f,save_sti_startframe = False):

    roi_f = pd.read_csv(roi_f)
    fob_dir = os.path.dirname(trial_folder)

    pd_name = get_pd_h5(trial_folder)
    ni = h5py.File(pd_name, "r")
    #sti_signal = ni['AI']["UVLED"][:, 0]  # (1640499frame,) 5000Hz
    #frame_counter = ni['CI']["FrameCounter"][:, 0]
    frame_out = ni['DI']["FrameOut"][:, 0]
    camera_signal = ni['AI']["side_camera"][:, 0]

    frame_start = np.nonzero(np.diff(frame_out.astype(int)) > 0)[0]
    exp_info = get_exp_info(os.path.join(trial_folder, "Image_scan_1_region_0_0.tif"))
    if exp_info.get("zFastEnable"):
        img_rate = exp_info["frameRate"] / (exp_info["steps"] + exp_info["flybackFrames"])
        volume_frame = exp_info["steps"] + exp_info["flybackFrames"]
        frame_time = frame_start[int(exp_info["steps"] / 2)::volume_frame] / PD_RATE
        frame_start = frame_start[int(exp_info["steps"] / 2)::volume_frame]
    camera_start = np.nonzero(np.diff(camera_signal) > 1)[0]



    dat = glob.glob(fob_dir + "/*.dat")[0]
    stim_config_d, stim_df = load_stim_txt(os.path.join(fob_dir,'stim.txt'))
    FICTRAC_RATE = 50

    FT_df = pd.read_csv(dat, header=None, index_col=0)
    # if len(FT_df) < 20*50:
    #     return None

    FT_vs = -FT_df[5] * FICTRAC_RATE * BALL_RADIUS
    FT_vf = FT_df[6] * FICTRAC_RATE * BALL_RADIUS
    FT_av = -FT_df[7] * FICTRAC_RATE
    FT_v = FT_df[18] * FICTRAC_RATE * BALL_RADIUS
    FT_x = FT_df[14]
    FT_y = FT_df[15]
    FT_heading = FT_df[16]
    FT_ts = FT_df[21]

    print(fob_dir, "v:", FT_v.mean(), "av:", np.abs(FT_av).mean())
    # mean_v = FT_v.mean()
    # if mean_v < 2 or mean_v > 50 or FT_v[-200:].mean() > 50:
    #     return None, None
    stim_df = stim_df.join(pd.DataFrame(
        {"vs": FT_vs, "vf": FT_vf, "av": FT_av, "v": FT_v, "x": FT_x, "y": FT_y, "heading": FT_heading, "ts": FT_ts}))
    ######################### repair lost frame #####################################################################
    idx = stim_df.index.tolist()  # NOTE: cnt not match FT_frame (but match .dat:ft_ts)
    ft_ts1 = stim_df["ts"].to_numpy()  # NOTE: camera ts
    cnt0 = idx[0]
    ft_frame = camera_start[cnt0:]
    ft_ts2 = (ft_frame - ft_frame[0]) / SYNC_RATE
    ft_ts1 = ft_ts1 - ft_ts1[0]
    lost_frame = len(ft_ts2) - len(ft_ts1)
    idx_re = np.array(correct_ft_lost_idx(ft_ts1, ft_ts2)) + cnt0
    lost_frame_index = np.diff(idx_re - idx)
    idx_interp = interpolation_lost_frame(lost_frame_index,idx)
    stim_df = stim_df.loc[idx_interp]
    ######################## end of repair #########################################################################

    stim_df.to_csv(os.path.join(trial_folder, 'stim_F.csv'))

    ######################## sample 2p imaging frame rate to camera frame ##########################################
    image_idx = []
    frame_start_fictrac_idx = stim_df.index.values
    for i in range(len(frame_start_fictrac_idx)):
        image_idx.append(find_nearest(frame_start, camera_start[frame_start_fictrac_idx[i]]))

    stim_df_downsample = stim_df.reset_index()
    stim_df_downsample = pd.concat(([stim_df_downsample.reset_index(drop=True),
                                     roi_f.iloc[image_idx].reset_index(drop=True)]), axis=1)


    if os.path.exists(os.path.join(fob_dir,'trigger_data.mat')):
        import scipy.io as scio
        trigger_data = scio.loadmat(os.path.join(fob_dir, 'trigger_data.mat'))
        trigger_data = np.array(trigger_data['outputdata1']).astype('float')
        trigger_data_downsample = trigger_data[np.nonzero(np.diff(trigger_data[:, 0]) > 1)[0], :]
        if np.shape(trigger_data_downsample)[1] > 2:
            trigger_data_downsample_df = pd.DataFrame(data=trigger_data_downsample[:,1:3],columns=['stepper1','stepper2'])
        else:
            trigger_data_downsample_df = pd.DataFrame(data=trigger_data_downsample[:,1],columns=['Dop_on'])
        stim_df_downsample = pd.concat(([stim_df_downsample.reset_index(drop=True),
                                         trigger_data_downsample_df.iloc[cnt0:,:].reset_index(drop=True)]), axis=1)
    stim_df_downsample.to_csv(os.path.join(trial_folder, 'stim_F_downsample.csv'))
    ################################################################################################################

    ########################################### save sti start frame if needed #####################################
    if save_sti_startframe == True:
        sti_frame_start_index = np.where(stim_df_downsample['is_wait']<1)[0][0]
        sti_frame_start = find_nearest(frame_start, camera_start[stim_df_downsample['cnt'][sti_frame_start_index].astype('int8')])
        np.savetxt(os.path.join(trial_folder, 'sti_frame_start.csv'),np.atleast_1d( sti_frame_start))
        # with open(os.path.join(trial_folder, 'sti_start.csv'), 'w', newline='') as f:
        #     import csv
        #     writer = csv.writer(f)
        #     writer.writerow(sti_frame_start)
    ########################################## end save sti start ##################################################

    ##################################### plot roi #################################################################
    fig, axs = plt.subplots(np.shape(roi_f)[1], 1, figsize=(20, 10), dpi=300, sharex=True)
    sti_start = np.where(stim_df_downsample['is_wait']<1)[0][0]
    sti_end = np.where(stim_df_downsample['is_wait']<1)[0][-1]
    #axs[0].set_title(parent.split('\\')[-2] + ' ' + parent.split('\\')[-1])
    ts = np.linspace(1,len(stim_df_downsample),len(stim_df_downsample)) / 50
    for i in range(np.shape(roi_f)[1]):
        axs[i].plot(ts, np.array(stim_df_downsample.iloc[:,-1-i]))
        axs[i].set_ylim( np.min(stim_df_downsample.iloc[:,-1-i]), np.max(stim_df_downsample.iloc[:,-1-i]) )

        axs[i].add_patch(
                patches.Rectangle(
                    (sti_start / 50, np.min(stim_df_downsample.iloc[:,-1-i])),
                    (sti_end-sti_start) / 50,
                    np.max(stim_df_downsample.iloc[:,-1-i]),
                    edgecolor=None,
                    facecolor='blue',
                    alpha = 0.3
                ))
        axs[i].set_ylabel('ROI' + str(np.shape(roi_f)[1]-i-1))
    axs[i].set_xlabel('t/s')
    plt.savefig(os.path.join(parent,'all_ROI.png'))
    plt.savefig(os.path.join(parent, 'all_ROI.pdf'))
    ################################################################################################################
    ################################### plot V & compartment #####################################################
    for i in range(np.shape(roi_f)[1]-1):
        fig, axs = plt.subplots(4, 1, figsize=(20, 10), dpi=300, sharex=True)
        axs[0].plot(ts,np.array(stim_df_downsample['v']), c = 'k')
        axs[0].set_ylabel('v')
        # axs[0].axvline(x = 6, color='black', linestyle='--', linewidth=3)
        # axs[0].axvline(x=9, color='black', linestyle='--', linewidth=1)
        # axs[0].axvline(x=12, color='black', linestyle='--', linewidth=3)
        axs_2 = axs[0].twinx()
        axs_2.plot(ts,np.array(stim_df_downsample.iloc[:,-1-i]), c = 'b')
        axs_2.set_ylabel('ROI' + str(np.shape(roi_f)[1]-i-1))

        axs[1].plot(ts,np.array(stim_df_downsample['vf']), c = 'k')
        axs[1].set_ylabel('vf')
        # axs[1].axvline(x = 6, color='black', linestyle='--', linewidth=3)
        # axs[1].axvline(x=9, color='black', linestyle='--', linewidth=1)
        # axs[1].axvline(x=12, color='black', linestyle='--', linewidth=3)
        axs_2 = axs[1].twinx()
        axs_2.plot(ts,np.array(stim_df_downsample.iloc[:,-1-i]), c = 'b')
        axs_2.set_ylabel('ROI' + str(np.shape(roi_f)[1]-i-1))

        axs[2].plot(ts,np.array(stim_df_downsample['vs']), c = 'k')
        axs[2].set_ylabel('vs')
        # axs[2].axvline(x = 6, color='black', linestyle='--', linewidth=3)
        # axs[2].axvline(x=9, color='black', linestyle='--', linewidth=1)
        # axs[2].axvline(x=12, color='black', linestyle='--', linewidth=3)
        axs_2 = axs[2].twinx()
        axs_2.plot(ts,np.array(stim_df_downsample.iloc[:,-1-i]), c = 'b')
        axs_2.set_ylabel('ROI' + str(np.shape(roi_f)[1]-i-1))

        axs[3].plot(ts,np.array(stim_df_downsample['av']), c = 'k')
        axs[3].set_ylabel('av')
        # axs[3].axvline(x = 6, color='black', linestyle='--', linewidth=3)
        # axs[3].axvline(x=9, color='black', linestyle='--', linewidth=1)
        # axs[3].axvline(x=12, color='black', linestyle='--', linewidth=3)
        axs_2 = axs[3].twinx()
        axs_2.plot(ts,np.array(stim_df_downsample.iloc[:,-1-i]), c = 'b')
        axs_2.set_ylabel('ROI' + str(np.shape(roi_f)[1]-i-1))

        plt.savefig(os.path.join(parent, 'ROI' + str(np.shape(roi_f)[1]-i-1) +'.png'))
        # plt.savefig(os.path.join(parent, 'all_ROI.pdf'))


def plot_trial(trial_folder,roi_f,save_sti_startframe = False):

    roi_f = pd.read_csv(roi_f)
    fob_dir = os.path.dirname(trial_folder)

    pd_name = get_pd_h5(trial_folder)
    ni = h5py.File(pd_name, "r")
    #sti_signal = ni['AI']["UVLED"][:, 0]  # (1640499frame,) 5000Hz
    #frame_counter = ni['CI']["FrameCounter"][:, 0]
    frame_out = ni['DI']["FrameOut"][:, 0]
    camera_signal = ni['AI']["side_camera"][:, 0]

    frame_start = np.nonzero(np.diff(frame_out.astype(int)) > 0)[0]
    exp_info = get_exp_info(os.path.join(trial_folder, "Image_scan_1_region_0_0.tif"))
    if exp_info.get("zFastEnable"):
        img_rate = exp_info["frameRate"] / (exp_info["steps"] + exp_info["flybackFrames"])
        volume_frame = exp_info["steps"] + exp_info["flybackFrames"]
        frame_time = frame_start[int(exp_info["steps"] / 2)::volume_frame] / PD_RATE
        frame_start = frame_start[int(exp_info["steps"] / 2)::volume_frame]
    camera_start = np.nonzero(np.diff(camera_signal) > 1)[0]



    dat = glob.glob(fob_dir + "/*.dat")[0]
    stim_config_d, stim_df = load_stim_txt(os.path.join(fob_dir,'stim.txt'))
    FICTRAC_RATE = 50

    FT_df = pd.read_csv(dat, header=None, index_col=0)
    # if len(FT_df) < 20*50:
    #     return None

    FT_vs = -FT_df[5] * FICTRAC_RATE * BALL_RADIUS
    FT_vf = FT_df[6] * FICTRAC_RATE * BALL_RADIUS
    FT_av = -FT_df[7] * FICTRAC_RATE
    FT_v = FT_df[18] * FICTRAC_RATE * BALL_RADIUS
    FT_x = FT_df[14]
    FT_y = FT_df[15]
    FT_heading = FT_df[16]
    FT_ts = FT_df[21]

    print(fob_dir, "v:", FT_v.mean(), "av:", np.abs(FT_av).mean())
    # mean_v = FT_v.mean()
    # if mean_v < 2 or mean_v > 50 or FT_v[-200:].mean() > 50:
    #     return None, None
    stim_df = stim_df.join(pd.DataFrame(
        {"vs": FT_vs, "vf": FT_vf, "av": FT_av, "v": FT_v, "x": FT_x, "y": FT_y, "heading": FT_heading, "ts": FT_ts}))
    ######################### repair lost frame #####################################################################
    idx = stim_df.index.tolist()  # NOTE: cnt not match FT_frame (but match .dat:ft_ts)
    ft_ts1 = stim_df["ts"].to_numpy()  # NOTE: camera ts
    cnt0 = idx[0]
    ft_frame = camera_start[cnt0:]
    ft_ts2 = (ft_frame - ft_frame[0]) / SYNC_RATE
    ft_ts1 = ft_ts1 - ft_ts1[0]
    lost_frame = len(ft_ts2) - len(ft_ts1)
    idx_re = np.array(correct_ft_lost_idx(ft_ts1, ft_ts2)) + cnt0
    lost_frame_index = np.diff(idx_re - idx)
    idx_interp = interpolation_lost_frame(lost_frame_index,idx)
    stim_df = stim_df.loc[idx_interp]
    ######################## end of repair #########################################################################

    stim_df.to_csv(os.path.join(trial_folder, 'stim_df.csv'))

    ######################## sample 2p imaging frame rate to camera frame ##########################################
    image_idx = []
    frame_start_fictrac_idx = stim_df.index.values
    for i in range(len(frame_start_fictrac_idx)):
        image_idx.append(find_nearest(frame_start, camera_start[frame_start_fictrac_idx[i]]))
    pd.DataFrame(image_idx).to_csv(os.path.join(trial_folder, 'image_idx.csv'))


    stim_df_downsample = stim_df.reset_index()
    stim_df_downsample = pd.concat(([stim_df_downsample.reset_index(drop=True),
                                     roi_f.iloc[image_idx].reset_index(drop=True)]), axis=1)


    if os.path.exists(os.path.join(fob_dir,'trigger_data.mat')):
        import scipy.io as scio
        trigger_data = scio.loadmat(os.path.join(fob_dir, 'trigger_data.mat'))
        trigger_data = np.array(trigger_data['outputdata1']).astype('float')
        trigger_data_downsample = trigger_data[np.nonzero(np.diff(trigger_data[:, 0]) > 1)[0], :]
        if np.shape(trigger_data_downsample)[1] > 2:
            trigger_data_downsample_df = pd.DataFrame(data=trigger_data_downsample[:,1:3],columns=['stepper1','stepper2'])
        else:
            trigger_data_downsample_df = pd.DataFrame(data=trigger_data_downsample[:,1],columns=['Dop_on'])
        stim_df_downsample = pd.concat(([stim_df_downsample.reset_index(drop=True),
                                         trigger_data_downsample_df.iloc[cnt0:,:].reset_index(drop=True)]), axis=1)
    stim_df_downsample.to_csv(os.path.join(trial_folder, 'stim_df_downsample.csv'))
    ################################################################################################################

    ############################### save video #####################################################################


    ############################### end video saving ###############################################################

    ########################################### save sti start frame if needed #####################################
    if save_sti_startframe == True:
        sti_frame_start_index = np.where(stim_df_downsample['is_wait']<1)[0][0]
        sti_frame_start = find_nearest(frame_start, camera_start[stim_df_downsample['cnt'][sti_frame_start_index].astype('int8')])
        np.savetxt(os.path.join(trial_folder, 'sti_frame_start.csv'),np.atleast_1d( sti_frame_start))
        # with open(os.path.join(trial_folder, 'sti_start.csv'), 'w', newline='') as f:
        #     import csv
        #     writer = csv.writer(f)
        #     writer.writerow(sti_frame_start)
    ########################################## end save sti start ##################################################

    ##################################### plot roi #################################################################
    fig, axs = plt.subplots(np.shape(roi_f)[1], 1, figsize=(20, 10), dpi=300, sharex=True)
    sti_start = np.where(stim_df_downsample['is_wait']<1)[0][0]
    sti_end = np.where(stim_df_downsample['is_wait']<1)[0][-1]
    #axs[0].set_title(parent.split('\\')[-2] + ' ' + parent.split('\\')[-1])
    ts = np.linspace(1,len(stim_df_downsample),len(stim_df_downsample)) / 50
    for i in range(np.shape(roi_f)[1]):
        axs[i].plot(ts, np.array(stim_df_downsample.iloc[:,-1-i]))
        axs[i].set_ylim( np.min(stim_df_downsample.iloc[:,-1-i]), np.max(stim_df_downsample.iloc[:,-1-i]) )

        axs[i].add_patch(
                patches.Rectangle(
                    (sti_start / 50, np.min(stim_df_downsample.iloc[:,-1-i])),
                    (sti_end-sti_start) / 50,
                    np.max(stim_df_downsample.iloc[:,-1-i]),
                    edgecolor=None,
                    facecolor='blue',
                    alpha = 0.3
                ))
        axs[i].set_ylabel('ROI' + str(np.shape(roi_f)[1]-i-1))
    axs[i].set_xlabel('t/s')
    plt.savefig(os.path.join(parent,'all_ROI.png'))
    plt.savefig(os.path.join(parent, 'all_ROI.pdf'))
    ################################################################################################################
    ################################### plot V & compartment #####################################################
    for i in range(np.shape(roi_f)[1]-1):
        fig, axs = plt.subplots(4, 1, figsize=(20, 10), dpi=300, sharex=True)
        axs[0].plot(ts,np.array(stim_df_downsample['v']), c = 'k')
        axs[0].set_ylabel('v')
        # axs[0].axvline(x = 6, color='black', linestyle='--', linewidth=3)
        # axs[0].axvline(x=9, color='black', linestyle='--', linewidth=1)
        # axs[0].axvline(x=12, color='black', linestyle='--', linewidth=3)
        axs_2 = axs[0].twinx()
        axs_2.plot(ts,np.array(stim_df_downsample.iloc[:,-1-i]), c = 'b')
        axs_2.set_ylabel('ROI' + str(np.shape(roi_f)[1]-i-1))

        axs[1].plot(ts,np.array(stim_df_downsample['vf']), c = 'k')
        axs[1].set_ylabel('vf')
        # axs[1].axvline(x = 6, color='black', linestyle='--', linewidth=3)
        # axs[1].axvline(x=9, color='black', linestyle='--', linewidth=1)
        # axs[1].axvline(x=12, color='black', linestyle='--', linewidth=3)
        axs_2 = axs[1].twinx()
        axs_2.plot(ts,np.array(stim_df_downsample.iloc[:,-1-i]), c = 'b')
        axs_2.set_ylabel('ROI' + str(np.shape(roi_f)[1]-i-1))

        axs[2].plot(ts,np.array(stim_df_downsample['vs']), c = 'k')
        axs[2].set_ylabel('vs')
        # axs[2].axvline(x = 6, color='black', linestyle='--', linewidth=3)
        # axs[2].axvline(x=9, color='black', linestyle='--', linewidth=1)
        # axs[2].axvline(x=12, color='black', linestyle='--', linewidth=3)
        axs_2 = axs[2].twinx()
        axs_2.plot(ts,np.array(stim_df_downsample.iloc[:,-1-i]), c = 'b')
        axs_2.set_ylabel('ROI' + str(np.shape(roi_f)[1]-i-1))

        axs[3].plot(ts,np.array(stim_df_downsample['av']), c = 'k')
        axs[3].set_ylabel('av')
        # axs[3].axvline(x = 6, color='black', linestyle='--', linewidth=3)
        # axs[3].axvline(x=9, color='black', linestyle='--', linewidth=1)
        # axs[3].axvline(x=12, color='black', linestyle='--', linewidth=3)
        axs_2 = axs[3].twinx()
        axs_2.plot(ts,np.array(stim_df_downsample.iloc[:,-1-i]), c = 'b')
        axs_2.set_ylabel('ROI' + str(np.shape(roi_f)[1]-i-1))

        plt.savefig(os.path.join(parent, 'ROI' + str(np.shape(roi_f)[1]-i-1) +'.png'))
        # plt.savefig(os.path.join(parent, 'all_ROI.pdf'))


def main(fname):
    parent = os.path.dirname(fname)
    if fname.endswith(".dat"):
        return plot_fictrac(fname, parent)
    if fname.endswith("time_info.txt"):
        return write_fictrac_stim_dFF_video(parent)
    if fname.endswith("ft_pv.txt"):
        return proc_ft_pv(fname)

    main_init(fname)
    if fname.endswith(".tif") or fname.endswith(".avi"):  # TIF->MC
        fname = proc_tif(fname)
        proc_mmap(fname)
    elif fname.endswith(".mmap"):  # MC+ROI->F
        proc_mmap(fname)
    elif fname.endswith("i_std.png"):
        ROIModifyUI(r"D:\exp_2p\code_2p\roi_templates/%s.npy" % get_part(fname), fname).show()
    elif fname.endswith(".hdf5"):
        from cia_caiman import cm_view_components
        plt.ion()
        cm_view_components(fname)
        plt.ioff()
        plt.show()
    elif fname.endswith(".csv"):
        proc_csv(fname)
    # elif fname.endswith(".mat"):
    #     calc_cor_map(fname)
    else:
        print("invalid file!")
        return


def multi_processing(trial_folder):
    import glob
    #trial_folder = os.path.join(root, f)
    if os.path.isdir(trial_folder):
        # if f.startswith("F"):
        #     main(glob.glob(os.path.join(root, f, "*.dat"))[0])
        #if (f.startswith("MB") or f.startswith("57C10") or f.startswith("D161") or f.startswith("VC")) and not f.endswith("_Z"):
            mmap_l = glob.glob(os.path.join(trial_folder, "*.mmap"))
            zs_csv = os.path.join(trial_folder, "zscore.csv")
            color_avi = os.path.join(trial_folder, "v_color.avi")
            roi_npy = os.path.join(trial_folder, "roi.npy")
            ft_pv_txt = os.path.join(trial_folder, "ft_pv.txt")
            tif = os.path.join(trial_folder, "Image_scan_1_region_0_0.tif")
            if not os.path.exists(tif):
                try:
                    tif = glob.glob(os.path.join(trial_folder, "*.tiff"))[0]
                except:
                    try:
                        tif = glob.glob(os.path.join(trial_folder, "*.tif"))[0]
                    except:
                        return
            else:
                main_init(tif)
            cluster = glob.glob(os.path.join(trial_folder, "cluster"))
            roi_png = os.path.join(trial_folder, "all_roi.png")

            if (len(mmap_l) == 0):
                proc_tif(tif)
            # else:
            #     if not os.path.exists(roi_png):
            #         proc_mmap(mmap_l[0])

            plt.clf()
            plt.close("all")
    pass


# Return the list of subdirectories that start with 'MB'
def get_MB_file_folder(file_folder,need_MB_start = True):# "E:\LQT_2data\TH+DDC\221208_virgin_f_mantis\221208-M5-FT_DdcTH"
    folder_list = []
    for root,dirs,files in os.walk(file_folder):
        for dir in dirs:
            for _,exp_folders,_ in os.walk(os.path.join(root,dir)):
                for exp_folder in exp_folders:
                    if need_MB_start == True:
                        if exp_folder.startswith('MB'):
                            folder_list.append(os.path.join(root,dir,exp_folder))
                    else:
                        folder_list.append(os.path.join(root, dir, exp_folder))
    return folder_list


if __name__ == '__main__':


    multi_process = True
    for i in range(1,len(sys.argv)):
        folder_list = get_MB_file_folder(sys.argv[i], need_MB_start=False)
        import glob

        # root = sys.argv[1]  # r"\\192.168.1.63\lqt\LQTdata\EPG\210824"#r"D:\exp_2p\data\EPG\210823"
        # for f in os.listdir(root):
        #     trial_folder = os.path.join(root, f)

        for trial_folder in folder_list:
            if os.path.isdir(trial_folder):
                # if f.startswith("F"):
                #     main(glob.glob(os.path.join(root, f, "*.dat"))[0])
                # if (f.startswith("MB") or f.startswith("57C10") or f.startswith("D161") or f.startswith("VC")) and not f.endswith("_Z"):
                    mmap_l = glob.glob(os.path.join(trial_folder, "*.mmap"))
                    zs_csv = os.path.join(trial_folder, "zscore.csv")
                    color_avi = os.path.join(trial_folder, "v_color.avi")
                    roi_npy = os.path.join(trial_folder, "roi.npy")
                    ft_pv_txt = os.path.join(trial_folder, "ft_pv.txt")
                    tif = os.path.join(trial_folder, "Image_scan_1_region_0_0.tif")
                    if not os.path.exists(tif):
                        try:
                            tif = glob.glob(os.path.join(trial_folder, "*.tiff"))[0]
                        except:
                            try:
                                tif = glob.glob(os.path.join(trial_folder, "*.tif"))[0]
                            except:
                                continue
                        parent = trial_folder
                    else:
                        main_init(tif)
                    offset = os.path.join(trial_folder, "offset.txt")
                    #roi_png = os.path.join(trial_folder, "all_roi.png")
                    roi_png = os.path.join(trial_folder, "roi.png")

                    if len(mmap_l) == 0:
                        if multi_process:
                            break
                        proc_tif(tif)
                    else:
                        # proc_mmap(mmap_l[0])
                        if not os.path.exists(roi_npy):
                            proc_mmap(mmap_l[0], True)
                        else:
                            if not os.path.exists(roi_png):
                                # if not time.ctime(os.path.getmtime(zs_csv))[0:10] == time.ctime(time.time())[0:10]: # not modified today
                                proc_mmap(mmap_l[0],only_roi_change=False,only_roi=False)
                                pass
                                # if not os.path.exists(color_avi):
                                # proc_csv(zs_csv)
                            else:
                                proc_mmap(mmap_l[0])
                                pass
                    plt.clf()
                    plt.close("all")
                    gc.collect()

        if multi_process:
            Parallel(n_jobs=15, backend='multiprocessing', verbose=1) \
                (delayed(multi_processing)(trials) for trials in folder_list)

    # else:
    #     main(sys.argv[1])
