# -*- coding: utf-8 -*-

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction.cnmf import cnmf as cnmf

def cm_view_components(hdf5):
    est = cnmf.load_CNMF(hdf5).estimates
    est.plot_contours()#img=est.Cn)
    # plt.savefig(os.path.dirname(hdf5) + "/ROI.png")
    # NOTE: spatial components
    plt.figure(); plt.imshow(np.reshape(est.A[:, 0].toarray(), (128, 128), order='F'))
    # NOTE: temporal components
    plt.figure(); plt.plot(est.C[0])
    plt.figure(); plt.plot(est.F_dff[0])
    # est.view_components()

def cm_view_mc(fnames, mc_mmap):
    shifts_rig = np.load(os.path.join(os.path.dirname(mc_mmap), "mc_shifts.npy"))
    for i, j in enumerate(shifts_rig):
        print("%d %s" % (i, j))

    m_orig = cm.load_movie_chain(fnames)
    m_els = cm.load(mc_mmap)
    from cia_utils import view_img_seq
    view_img_seq(np.concatenate([m_orig, m_els], axis=2))

    # ds_ratio = 0.2
    # moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio),
    #                               m_els.resize(1, 1, ds_ratio)], axis=2)
    # moviehandle.play(fr=20, q_max=99.5, magnification=2)


def cm_motion_correction(fnames, parent, fps=5, save_avi=True, use_cuda = False, is3D = False):
    # dxy = (2., 2.)  # spatial resolution in x and y in (um per pixel)
    # max_shift_um = (12., 12.)  # maximum shift in um
    # patch_motion_um = (100., 100.)
    dxy = (1., 1.)  # spatial resolution in x and y in (um per pixel)
    max_shift_um = (12., 12.)  # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
    max_shifts = [int(a / b) for a, b in zip(max_shift_um, dxy)]
    strides = tuple([int(a / b) for a, b in zip(patch_motion_um, dxy)])
    opts = params.CNMFParams(params_dict={'fnames': fnames, 'fr': fps, 'decay_time': 0.4, 'dxy': dxy, 'pw_rigid': False,
                                          'max_shifts': max_shifts, 'strides': strides, 'overlaps': (24, 24),
                                          'max_deviation_rigid': 3, 'border_nan': 'copy', 'niter_rig': 3, 'use_cuda': use_cuda, 'is3D': is3D })
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=True)
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    m = cm.load(mc.mmap_file, is3D=is3D)
    if save_avi:
        m.save(parent + "/mc.avi")
        np.save(parent + "/mc_shifts.npy", mc.shifts_rig)
    return mc.mmap_file[0]

def cm_detect_roi(mmap_F, K=12, gSig=16, merge_thr=0.9, method_init ="corr_pnr"):
    # https://caiman.readthedocs.io/en/master/Getting_Started.html#basic-structure

    mmap_C = cm.save_memmap([mmap_F], base_name='memmap_', order='C', border_to_0=12)  # exclude borders
    Yr, dims, T = cm.load_memmap(mmap_C)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    Cn = images.mean(axis=0)
    dxy = (1., 1.)  # spatial resolution in x and y in (um per pixel)
    max_shift_um = (12., 12.)  # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
    max_shifts = [int(a / b) for a, b in zip(max_shift_um, dxy)]
    strides = tuple([int(a / b) for a, b in zip(patch_motion_um, dxy)])

    fnames = glob.glob(os.path.dirname(mmap_F) + "/Image_scan_*.tif")
    opts = params.CNMFParams(params_dict={'fnames': fnames, 'fr': 6.736, 'decay_time': 0.4, 'dxy': dxy, 'pw_rigid': False,
                                          'max_shifts': max_shifts, 'strides': strides, 'overlaps': (24, 24),
                                          'max_deviation_rigid': 3, 'border_nan': 'copy', 'niter_rig': 1})

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=True)

    # NOTE: use patch rf=15
    # opts.change_params(params_dict={'fnames': fnames, 'p': 0, 'fr': 6.736, 'nb': 2, 'rf': 15, 'K': 4, 'gSig': [4, 4], 'stride': 6,
    #     'method_init': 'greedy_roi', 'rolling_sum': True, 'merge_thr': 0.85, 'n_processes': n_processes, 'only_init': True, 'ssub': 2, 'tsub': 2})
    # NOTE: not use patch rf=None K=10 (p = 0 turns deconvolution off) method_init=(greedy_roi|corr_pnr)
    opts.change_params(params_dict={'fnames': fnames, 'p': 0, 'fr': 6.736, 'nb': 1, 'rf': None, 'K': K, 'gSig': [gSig, gSig], 'stride': 6, "update_background_components": False,
        'method_init': method_init, 'min_pnr': 10, 'rolling_sum': True, 'merge_thr': merge_thr, 'n_processes': n_processes, 'only_init': True, 'ssub': 1, 'tsub': 1})
    # opts.change_params(params_dict={'dims': dims,
    #                                 'method_init': 'corr_pnr',  # use this for 1 photon
    #                                 'K': K,
    #                                 'gSig': [3, 3],
    #                                 'gSiz': (13, 13),
    #                                 'merge_thr': .7,
    #                                 'p': 0,
    #                                 'tsub': 2,
    #                                 'ssub': 2,
    #                                 'rf': 40,
    #                                 'stride': 20,
    #                                 'only_init': True,  # set it to True to run CNMF-E
    #                                 'nb': 0,
    #                                 'nb_patch': 0,
    #                                 'low_rank_background': None,
    #                                 'update_background_components': True,
    #                                 # sometimes setting to False improve the results
    #                                 'min_corr': .8,
    #                                 'min_pnr': 10,
    #                                 'normalize_init': False,  # just leave as is
    #                                 'center_psf': True,  # leave as is for 1 photon
    #                                 'ssub_B': 2,
    #                                 'ring_size_factor': 1.4,
    #                                 'del_duplicates': True,  # whether to remove duplicates from initialization
    #                                 'border_pix': 12})
    # c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=True)
    # opts = params.CNMFParams(params_dict={'decay_time': 0.4, 'dxy': dxy, 'pw_rigid': False,
    #                                       'max_shifts': max_shifts, 'strides': strides, 'overlaps': (24, 24),
    #                                       'max_deviation_rigid': 3, 'border_nan': 'copy', 'niter_rig': 1,
    #
    #                                       'fnames': None, 'p': 1, 'fr': 30, 'nb': 2, 'rf': 15, 'K': 4, 'gSig': [4, 4],
    #                                       'stride': 6, 'method_init': 'greedy_roi', 'rolling_sum': True, 'merge_thr': 0.85,
    #                                       'n_processes': n_processes,
    #                                       'only_init': True, 'ssub': 2, 'tsub': 2
    #                                       })
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)
    Cns = cm.summary_images.local_correlations_movie_offline(mmap_F, remove_baseline=True, window=1000, stride=1000,
                                                             winSize_baseline=100, quantil_min_baseline=10, dview=dview)
    Cn = Cns.max(axis=0)
    Cn[np.isnan(Cn)] = 0
    # cnm.estimates.plot_contours(img=Cn)
    cnm.estimates.Cn = Cn
    # return
    cnm2 = cnm
    if method_init == "greedy_roi":
        cnm2 = cnm.refit(images, dview=dview)
    cnm2.estimates.plot_contours(img=Cn)
    plt.savefig(os.path.dirname(mmap_F) + "/auto_ROI_%s.png" % method_init)

    # cnm2.params.set('quality', {'decay_time': 0.4, 'min_SNR': 2, 'rval_thr': 0.85, 'use_cnn': False, 'min_cnn_thr': 0.99,
    #                             'cnn_lowest': 0.1})
    # cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    # cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
    # cnm2.estimates.select_components(use_object=True)

    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
    # from caiman.source_extraction.cnmf.utilities import extract_DF_F
    # cnm2.estimates.F_dff = extract_DF_F(Yr, cnm2.estimates.A, cnm2.estimates.C, cnm2.bl,
    #                      quantileMin=8, frames_window=200, dview=dview)

    cnm2.estimates.view_components(img=Cn)
    # cnm2.estimates.Cn = Cn
    cnm2.save(os.path.dirname(mmap_F) + "cnmf.hdf5")
    pd.DataFrame(cnm2.estimates.F_dff.T, columns=range(1, cnm2.estimates.F_dff.shape[0]+1)).to_csv(os.path.dirname(mmap_F) + "/dFF.csv", index=False)

    # plt.figure()
    # i = 6
    # d = cnm2.estimates.F_dff[i]
    # a = [np.sum(img * np.reshape(cnm2.estimates.A[:, i], (128, 128))) for img in images]
    # b = [np.sum(img * np.reshape((cnm2.estimates.A[:, i] > 0).astype(int), (128, 128))) for img in images]
    # plt.plot((a - np.mean(a)) / np.std(a), label="a")
    # plt.plot((b - np.mean(b)) / np.std(b), label="b")
    # plt.plot((d - np.mean(d)) / np.std(d), label="dff")
    # plt.legend()

    #  cnm2 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    #  cnm2.fit_file(motion_correct=True)
    cm.stop_server(dview=dview)
