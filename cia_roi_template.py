# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from EasyROI import EasyROI
roi_helper = EasyROI(verbose=True)


SCALE_SPEED = 0.01
ROTATE_SPEED = 1


def contours_bbox(contours):
    if isinstance(contours, list):
        c = np.concatenate(contours)
    else:
        c = contours
    return c[:, 0, 0].min(), c[:, 0, 1].min(), c[:, 0, 0].max(), c[:, 0, 1].max()

def contour_center(c):
    return (c[:, 0, 0].min() + c[:, 0, 0].max())/2, (c[:, 0, 1].min() + c[:, 0, 1].max())/2

def plot_contours(ax, contours, flags, texts=None):
    for i, c in enumerate(contours):
        xs, ys = c[:, 0, 0], c[:, 0, 1]
        ax.plot(np.concatenate([xs, xs[:1]]), np.concatenate([ys, ys[:1]]), linestyle="--" if flags[i] else "-", alpha=0.6)
        if texts is not None and texts[i]:
            x, y = contour_center(c)
            ax.text(x, y, texts[i])

class ROITemplateUI(object):
    def __init__(self, file, threshold=245):
        self.file = file
        if file.endswith(".png"):
            gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            _, self.contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.RETR_CCOMP)  #cv2.RETR_EXTERNAL
        else:
            self.contours = np.load(file, allow_pickle=True)
        self.flags = np.ones((len(self.contours), ), np.bool).tolist()

        fig, self.ax = plt.subplots(figsize=(8, 6), num="ROI Template")
        plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.15)
        fig.canvas.mpl_connect('key_press_event', self.onkey)
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        fig.canvas.mpl_connect("close_event", self.onclose)
        self.refresh()

    def refresh(self):
        self.ax.cla()
        texts = []
        i = 0
        for f in self.flags:
            if f:
                texts.append(str(i))
                i += 1
            else:
                texts.append(None)
        plot_contours(self.ax, self.contours, self.flags, texts)
        self.ax.invert_yaxis()
        print(self.flags)
        plt.draw()

    def save_result(self):
        if self.file.endswith(".png"):
            sel = []
            for i, c in enumerate(self.contours):
                if self.flags[i]:
                    sel.append(c)
            np.save(self.file.replace(".png", ".npy"), sel)

    def onclick(self, event):
        if event.xdata and event.ydata:
            if event.button == 3:
                for i, c in enumerate(self.contours):
                    if self.flags[i] and cv2.pointPolygonTest(c, (event.xdata, event.ydata), False) == 1:
                        self.contours.pop(i)
                        self.contours.insert(0, c)
                        self.flags.pop(i)
                        self.flags.insert(0, True)
                        break
            else:
                for i, c in enumerate(self.contours):
                    if cv2.pointPolygonTest(c, (event.xdata, event.ydata), False) == 1:
                        self.flags[i] = not self.flags[i]
            self.refresh()

    def onkey(self, event):
        if event.key == "enter":
            self.save_result()
        if event.key == "z":
            for i, c in enumerate(self.contours):
                self.flags[i] = False
            self.refresh()

    def show(self):
        plt.show()

    def onclose(self, event):
        self.save_result()


class ROIModifyUI(object):
    def __init__(self, template, bg_file):


        self.file = bg_file
        self.template = template
        self.bg_img = cv2.imread(bg_file)
        if template and os.path.exists(template):
            use_template = template.find("template") > 0
            self.contours = np.load(template, allow_pickle=True)
            self.contours = [c.astype(float) for c in self.contours]
        else:
            use_template = False
            if os.path.exists(os.path.join(os.path.dirname(bg_file), "roi.npy")):
                result = np.load(os.path.join(os.path.dirname(bg_file), "roi.npy"), allow_pickle=True)
            else:
                result = self.manual_draw()
            self.contours = result
        self.ins_points = []
        self.flags = np.ones((len(self.contours)*2, ), np.bool)
        self.is_press_left = False
        self.is_press_right = False
        self.is_ctrl = False
        self.is_shift = False

        self.fig, self.ax = plt.subplots(figsize=(8, 6), num="Modify ROI")
        plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.15)
        self.id_onkeypress =self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.id_onkeyrelease = self.fig.canvas.mpl_connect('key_release_event', self.onkeyrelease)
        self.id_onpress =self.fig.canvas.mpl_connect('button_press_event', self.onpress)
        self.id_onmove =self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.id_onrelease =self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.id_onscroll=self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.id_onclose =self.fig.canvas.mpl_connect("close_event", self.onclose)
        if use_template:
            self.scale_contours_to_center()
        self.refresh()

    def manual_draw(self):
        from roipoly import RoiPoly, MultiRoi
        #fig_manual = plt.figure(figsize=(8, 6))
        img = cv2.imread(self.file)
        n = int(input("Enter a roi number: "))
        polygon_roi = roi_helper.draw_polygon(img, n)  # quantity=3 specifies number of polygons to draw

        #frame_temp = roi_helper.visualize_roi(img, polygon_roi)

        #plt.imshow(img)
        #multiroi_named = MultiRoi(fig=fig_manual)
        #multiroi_named = RoiPoly(color='r')  # draw new ROI in red color
        contours = []
        '''
        for key in multiroi_named.rois:
            contours.append(np.array(
                list(zip(multiroi_named.rois[key].x, multiroi_named.rois[key].y))
                                    ).reshape(len(multiroi_named.rois[key].x),1,2)
                            )
        '''
        for key in polygon_roi['roi']:
            contours.append(np.array(
                list(polygon_roi['roi'][key]['vertices'])
            ).reshape(len(polygon_roi['roi'][key]['vertices']),1,2)
            )

        # contours.append(np.array(
        #         list(zip(multiroi_named.x, multiroi_named.y))
        #                             ).reshape(len(multiroi_named.x),1,2)
        #                     )
        return contours


    def scale_contours_to_center(self):
        if not len(self.contours):
            return
        l, t, r, b = contours_bbox(self.contours)
        cx, cy = (l + r) / 2, (t + b) / 2
        ix, iy = self.bg_img.shape[1] / 2, self.bg_img.shape[0] / 2
        sx, sy = ix - cx, iy - cy
        sc = self.bg_img.shape[1] / (r - l) * 0.5
        for c in self.contours:
            c[:, 0, 0] = (c[:, 0, 0] - cx) * sc + ix
            c[:, 0, 1] = (c[:, 0, 1] - cy) * sc + iy

    def scale_contours(self, contours, stepx, stepy):
        if not len(contours):
            return
        l, t, r, b = contours_bbox(contours)
        cx, cy = (l + r) / 2, (t + b) / 2
        for c in contours:
            c[:, 0, 0] = (c[:, 0, 0] - cx) * (1 + stepx * SCALE_SPEED) + cx
            c[:, 0, 1] = (c[:, 0, 1] - cy) * (1 + stepy * SCALE_SPEED) + cy

    def move_contours(self, contours, dx, dy):
        for c in contours:
            c[:, 0, 0] += int(dx)
            c[:, 0, 1] += int(dy)

    def rotate_contours(self, contours, angle):
        if not len(contours):
            return
        angle = np.deg2rad(angle) * ROTATE_SPEED
        l, t, r, b = contours_bbox(contours)
        cx, cy = (l + r) / 2, (t + b) / 2
        for c in contours:
            x1, y1 = c[:, 0, 0] - cx, c[:, 0, 1] - cy
            c[:, 0, 0] = x1 * np.cos(angle) - y1 * (np.sin(angle)) + cx
            c[:, 0, 1] = x1 * np.sin(angle) + y1 * (np.cos(angle)) + cy

    def get_sel_contours(self):
        ret = []
        for i, c in enumerate(self.contours):
            if self.flags[i]:
                ret.append(c)
        return ret

    def refresh(self):
        self.ax.cla()
        self.ax.imshow(self.bg_img)
        self.ax.set_xlim(0, self.bg_img.shape[1])
        self.ax.set_ylim(self.bg_img.shape[0], 0)
        plot_contours(self.ax, self.contours, self.flags, [str(i) for i in range(len(self.contours))])
        if len(self.ins_points):
            self.ax.plot([x for x, y in self.ins_points], [y for x, y in self.ins_points], ".-", c="r", lw=1)
        self.show()

    def save_result(self):
        np.save(os.path.join(os.path.dirname(self.file), "roi.npy"), self.contours)
        self.save_csv_result()

    def save_csv_result(self):
        import pandas as pd
        res = []
        for i, c in enumerate(self.contours):
            res.extend([[i, p[0][0], p[0][1]] for p in c])
        pd.DataFrame(res, columns=["id", "x", "y"]).to_csv(os.path.join(os.path.dirname(self.file), "roi.csv"), index=False)

    def onpress(self, event):
        if event.xdata and event.ydata:
            self.last_move = None
            if event.button == 1:
                self.is_press_left = True
            elif event.button == 3:
                self.is_press_right = True

    def onmove(self, event):
        if event.xdata and event.ydata:
            if self.is_press_left:  # NOTE: move
                if self.last_move is not None:
                    self.move_contours(self.get_sel_contours(), event.xdata - self.last_move[0], event.ydata - self.last_move[1])
                self.last_move = event.xdata, event.ydata
                self.refresh()
            elif self.is_press_right:  # NOTE: rotate by x
                if self.last_move is not None:
                    self.rotate_contours(self.get_sel_contours(), event.x - self.last_move[0])
                self.last_move = event.x, event.y
                self.refresh()

    def onrelease(self, event):
        if self.last_move is None:
            if self.is_press_left:
                for i, c in enumerate(self.contours):
                    if cv2.pointPolygonTest(c.astype(int), (event.xdata, event.ydata), False) == 1:
                        if self.is_shift:
                            self.flags[i] = not self.flags[i]
                        else:
                            for j in range(len(self.contours)):
                                self.flags[j] = False
                            self.flags[i] = True
                        self.refresh()
                        break
            elif self.is_press_right:  # NOTE: insert point
                self.ins_points.append([event.xdata, event.ydata])
                self.refresh()

        if event.button == 1:
            self.is_press_left = False
        elif event.button == 3:
            self.is_press_right = False

    def onscroll(self, event):
        print(event.step)
        if self.is_ctrl:
            self.scale_contours(self.get_sel_contours(), 0, event.step)
        elif self.is_shift:
            self.scale_contours(self.get_sel_contours(), event.step, 0)
        else:
            self.scale_contours(self.get_sel_contours(), event.step, event.step)
        self.refresh()

    def confirm_insert_contour(self):
        if not len(self.ins_points):
            return
        contour = np.array([[p] for p in self.ins_points])
        self.contours.insert(0, contour)
        self.ins_points = []
        self.refresh()

    def onkeypress(self, event):
        print(event.key)
        if event.key == "enter":
            self.confirm_insert_contour()
            self.save_result()
        elif event.key == "control":
            self.is_ctrl = True
        elif event.key == "shift":
            self.is_shift = True
        elif event.key == "ctrl+z":
            # if len(self.ins_points):
            #     self.ins_points.pop()
            #     self.refresh()
            if self.older_contours:
                self.contours = self.older_contours
                self.refresh()
        elif event.key == "ctrl+a":
            for i in range(len(self.contours)):
                self.flags[i] = True
            self.refresh()
        elif event.key == "A":
            for i in range(len(self.contours)):
                self.flags[i] = False
            self.refresh()
        elif event.key == "delete":
            self.older_contours = self.contours
            i = np.where(self.flags == True)[0]
            del self.contours[i[0]]
            self.flags = np.ones((len(self.contours)*2,), np.bool)
            self.refresh()
            if len(self.contours) == 0:
                self.contours = self.manual_draw()
                self.flags = np.ones((len(self.contours) * 2,), np.bool)
                # plt.close(self.fig)
                # ROIModifyUI(self.template, self.file)
                self.refresh()
        elif event.key == "ctrl+equal":
            i = np.where(self.flags == True)[0]

            self.contours[i[0]],self.contours[i[0]+1] = self.contours[i[0]+1],self.contours[i[0]]
            self.flags[i[0]],self.flags[i[0]+1] = self.flags[i[0]+1],self.flags[i[0]]

            self.refresh()

    def onkeyrelease(self, event):
        if event.key == "control":
            self.is_ctrl = False
        elif event.key == "shift":
            self.is_shift = False

    def show(self):
        plt.show(block= True)

    def onclose(self, event):
        self.save_result()

if __name__ == '__main__':
    # D:\exp_2p\code_2p\roi_templates\EB.npy D:\exp_2p\EPG\210618\CX1001-7F-F4-EB-DARK-1\i_std.png
    # 0 D:\exp_2p\EPG\210618\CX1001-7F-F4-EB-DARK-1\i_std.png
    plt.ioff()
    if len(sys.argv) > 2:
        ROIModifyUI(sys.argv[1], sys.argv[2]).show()
    else:
        ROITemplateUI(sys.argv[1], 100).show()  #PB 100 EB 245
