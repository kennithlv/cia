import sys
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from cia_utils import *

img_rate = None


def load_each_sti(file):
    each_sti = json.load(open(file, "r"))
    return each_sti
    pass


def load_whole_sti(file):
    whole_sti, sti_img_fame = json.load(open(file, "r"))
    return whole_sti, sti_img_fame


def get_min_sti_length(each_sti):
    mini_len = len(each_sti[0])
    for i in range(len(each_sti)):
        if len(each_sti[i]) < mini_len:
            mini_len = len(each_sti[i])
    return mini_len


def align_sti_length(each_sti, mini_len):
    for i in range(len(each_sti)):
        each_sti[i] = each_sti[i][:mini_len]
    return each_sti


def plot_mean_sti(parent, each_sti):
    global img_rate

    for i in range(len(each_sti)):  # minuse the start average
        a = np.average(each_sti[i][0:int(2 * img_rate)], axis=0)
        each_sti[i] = each_sti[i] - a
        pass
    dFF = np.sum([each_sti[i] for i in range(len(each_sti))], axis=0).tolist()
    dFF = np.divide(dFF, len(each_sti))
    # img_rate = 49.466

    fig, axs = plt.subplots(np.shape(dFF)[1] + 1, 1, figsize=(20, 10), dpi=300, sharex=True)
    axs[0].set_title(parent.split('\\')[-2] + ' ' + parent.split('\\')[-1] + ' mean')
    ts = np.linspace(1, len(dFF), len(dFF)) / img_rate
    for i in range(np.shape(dFF)[1]):
        for trial in range(len(each_sti)):
            axs[i].plot(ts, np.array(each_sti[trial])[:, i], color='grey', alpha=0.3)
            pass
        axs[i].plot(ts, dFF[:, i])
        axs[i].set_ylim(np.min(dFF[:, i]), np.max(dFF[:, i]))
        axs[i].add_patch(
            patches.Rectangle(
                (2, np.min(dFF[:, i])),
                2,
                np.max(dFF[:, i]),
                edgecolor=None,
                facecolor='blue',
                alpha=0.3
            ))
        axs[i].set_ylabel('ROI' + str(i + 1) + ' _deltaF/F')
    axs[i].set_xlabel('t/s')
    plt.savefig(os.path.join(parent, 'all_ROI_mean.png'))
    plt.savefig(os.path.join(parent, 'all_ROI_mean.pdf'))
    plt.close("all")
    return dFF


def plot_mean_whole(parent, each_sti, sti_img_frame):
    for i in range(len(each_sti)):  # minuse the start average
        a = np.average(each_sti[i][0:100], axis=0)
        each_sti[i] = each_sti[i] - a
        pass
    dFF = np.sum([each_sti[i] for i in range(len(each_sti))], axis=0).tolist()
    dFF = np.divide(dFF, len(each_sti))
    img_rate = 49.447

    fig, axs = plt.subplots(np.shape(dFF)[1] + 1, 1, figsize=(20, 10), dpi=300, sharex=True)
    axs[0].set_title(parent.split('\\')[-2] + ' ' + parent.split('\\')[-1].split('-')[0] + ' mean')
    ts = np.linspace(1, len(dFF), len(dFF)) / img_rate
    for i in range(np.shape(dFF)[1]):
        for trial in range(len(each_sti)):
            axs[i].plot(ts, np.array(each_sti[trial])[:, i], color='grey', alpha=0.3)
            pass
        axs[i].plot(ts, dFF[:, i])
        axs[i].set_ylim(np.min(dFF[:, i]), np.max(dFF[:, i]))
        for kk in sti_img_frame:
            axs[i].add_patch(
                patches.Rectangle(
                    (kk[0] / img_rate, np.min(dFF[:, i])),
                    (kk[1] - kk[0]) / img_rate,
                    np.max(dFF[:, i]),
                    edgecolor=None,
                    facecolor='blue',
                    alpha=0.3
                ))
        axs[i].set_ylabel('ROI' + str(i + 1) + ' _deltaF/F')
    axs[i].set_xlabel('t/s')

    plt.savefig(os.path.join(os.path.dirname(parent), parent.split('\\')[-1].split('-')[0] + '_mean.png'))
    plt.savefig(os.path.join(os.path.dirname(parent), parent.split('\\')[-1].split('-')[0] + '_mean.pdf'))
    plt.close("all")


def plot_each_trial_same_fly(fname, each_sti):
    global img_rate

    for i in range(len(each_sti)):  # minuse the start average
        a = np.average(each_sti[i][0:int(2 * img_rate)], axis=0)
        each_sti[i] = each_sti[i] - a
        pass
    each_sti = align_sti_length(each_sti, get_min_sti_length(each_sti))
    dFF = np.sum([each_sti[i] for i in range(len(each_sti))], axis=0).tolist()
    dFF = np.divide(dFF, len(each_sti))
    # img_rate = 49.466
    fig, axs = plt.subplots(np.shape(dFF)[1] + 1, 1, figsize=(20, 10), dpi=300, sharex=True)
    axs[0].set_title(fname.split('\\')[-2] + ' ' + fname.split('\\')[-1].strip('.txt') + ' mean')
    ts = np.linspace(1, len(dFF), len(dFF)) / img_rate
    for i in range(np.shape(dFF)[1]):
        cmap = plt.get_cmap('jet', len(each_sti))
        for trial in range(len(each_sti)):
            axs[i].plot(ts, np.array(each_sti[trial])[:, i], color='grey', alpha=0.3)
            pass
        axs[i].plot(ts, dFF[:, i])
        axs[i].set_ylim(np.min(dFF[:, i]), np.max(dFF[:, i]))
        axs[i].add_patch(
            patches.Rectangle(
                (2, np.min(dFF[:, i])),
                2,
                np.max(dFF[:, i]),
                edgecolor=None,
                facecolor='blue',
                alpha=0.3
            ))
        axs[i].set_ylabel('ROI' + str(i + 1) + ' _deltaF/F')
    axs[i].set_xlabel('t/s')
    plt.savefig(os.path.join(os.path.dirname(fname), fname.split('\\')[-1].strip('.txt') + '_mean.png'))
    plt.savefig(os.path.join(os.path.dirname(fname), fname.split('\\')[-1].strip('.txt') + '_mean.pdf'))
    pass

def pixel_sort_combine(sys_arg):
    for i in range(len(sys_arg)):
        if not i == 0:
            parent = sys_arg[i]
            temp = np.load(os.path.join(parent,'pixel_order','bin4','pixel_sti_sum_avg.npy'))
            if i == 1:
                pixel_sti_sum_avg = temp
            else:
                min_length_0 = np.minimum(np.size(temp, 0), np.size(pixel_sti_sum_avg, 0))
                min_length_1 = np.minimum(np.size(temp, 1), np.size(pixel_sti_sum_avg, 1))
                pixel_sti_sum_avg = pixel_sti_sum_avg[0:min_length_0, 0:min_length_1] + temp[0:min_length_0, 0:min_length_1]
            pass
    pixel_sti_sum_avg = pixel_sti_sum_avg / (len(sys_arg)-1)

    path = os.path.dirname(parent)
    np.save(os.path.join(path, os.path.basename(sys_arg[1]) + ' pixel_sti_sum_avg.npy'), pixel_sti_sum_avg)

    window = np.ones(int(10)) / float(10)
    for i in range(len(pixel_sti_sum_avg)):
        pixel_sti_sum_avg[i] = np.convolve(pixel_sti_sum_avg[i], window, 'same')  # smmoth the pixel signal
    #pixel_sti_sum_avg = (pixel_sti_sum_avg.T / pixel_sti_sum_avg.max(axis=1)).T  # normalise the pixel
    pixel_sti_sum_avg = np.squeeze(
        TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(pixel_sti_sum_avg)  # normalise the pixel signal (z-score)
                                    )

    # order according to the max position of the signal in each pixel
    index_max = np.argmax(pixel_sti_sum_avg[:, round(img_rate * 2):round(img_rate * 4)], axis=1)
    array_sort = pixel_sti_sum_avg[np.argsort(index_max)]

    plt.matshow(array_sort)
    plt.axvline(x=2*img_rate, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=4*img_rate, color='black', linestyle='--', linewidth=1)
    plt.axis('off')
    plt.title('pixel ordered fig all')
    plt.savefig(os.path.join(path, os.path.basename(sys_arg[1]) + ' pixel_order_all' + '.png'))
    plt.savefig(os.path.join(path, os.path.basename(sys_arg[1]) + ' pixel_order_all' + '.pdf'))
    pass

if __name__ == '__main__':
    # a = load_each_sti(r"D:\LQT\LQTdata\DAN\211102\MBr1pedc_G7F_F1f_Lmb_laser.txt")
    # plot_each_trial_same_fly(r"D:\LQT\LQTdata\DAN\211102\MBr1pedc_G7F_F1f_Lmb_laser.txt",a)

    #global img_rate
    parent = sys.argv[1]
    exp_info = load_exp_xml(os.path.join(parent, "Experiment.xml"))
    print(exp_info)
    img_rate = exp_info["frameRate"]
    if exp_info.get("zFastEnable"):
        img_rate /= exp_info["steps"] + exp_info["flybackFrames"]

    fly_trial_mean = []
    fly_trial = []

    #pixel_sort_combine(sys.argv)
    # # here to mean the same sti in one trial
    for i in range(len(sys.argv)):
        if not i == 0:
            parent = sys.argv[i]
            each_sti = load_each_sti(os.path.join(sys.argv[i], 'each_sti.txt'))
            each_sti = align_sti_length(each_sti, get_min_sti_length(each_sti))
            fly_trial_mean.append(plot_mean_sti(sys.argv[i], each_sti).tolist())
            fly_trial = np.array(each_sti).tolist() + fly_trial
    fname = os.path.dirname(parent) + '\\' + os.path.basename(sys.argv[1]) + ".txt"
    fname_each = os.path.dirname(parent) + '\\' + os.path.basename(sys.argv[1]) + "each.txt"
    json.dump(fly_trial_mean, open(fname, "w"))
    json.dump(fly_trial, open(fname_each, "w"))
    plot_each_trial_same_fly(fname, load_each_sti(fname))
    pass
