import cv2
import numpy as np
from nms import py_cpu_nms
from time import sleep


class Detector(object):

    def __init__(self, name='my_video', frame_num=10, k_size=7):

        self.name = name

        self.nms_threshold = 0.3

        self.time = 1/frame_num

        self.es = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size))

    def catch_video(self, video_index=0, k_size=7,
                    iterations=3, threshold=20, bias_num=1,
                    min_area=360, show_test=True, 
                    enhance=True, logical='or'):

        # video_index：摄像头索引或者视频路径
        # k_size：中值滤波的滤波器大小
        # iteration：腐蚀+膨胀的次数
        # threshold：二值化阙值
        # bias_num：计算帧差图时的帧数差
        # min_area：目标的最小面积
        # show_test：是否显示二值化图片

        logical = logical.lower()

        if not bias_num > 0:
            raise Exception('bias_num must > 0')

        if isinstance(video_index, str):
            is_camera = False
        else:
            is_camera = True

        cap = cv2.VideoCapture(video_index)  # 创建摄像头识别类

        if not cap.isOpened():
            # 如果没有检测到摄像头，报错
            raise Exception('Check if the camera is on.')

        frame_num = 0

        previous = []

        while cap.isOpened():

            catch, frame = cap.read()  # 读取每一帧图片

            if not catch:

                raise Exception('Unexpected Error.')

            if frame_num < bias_num:

                value = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                previous = [value]*bias_num

                frame_num += 1

            raw = frame.copy()

            gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.absdiff(gray1, previous[0])

            gray1 = cv2.medianBlur(gray1, k_size)

            _, mask1 = cv2.threshold(
                gray1, threshold, 255, cv2.THRESH_BINARY)

            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.absdiff(gray2, previous[1])

            gray2 = cv2.medianBlur(gray2, k_size)

            _, mask2 = cv2.threshold(
                gray2, threshold, 255, cv2.THRESH_BINARY)

            if logical == 'or':
                mask = (np.logical_or(mask1, mask2) + 0)
            elif logical == 'and':
                mask = (np.logical_and(mask1, mask2) + 0)
            else:
                raise Exception('Logical must be \'OR\' or \'AND\'')
            mask = (mask * 255).astype(np.uint8)

            if enhance:

                mask = cv2.dilate(mask, self.es, iterations)
                mask = cv2.erode(mask, self.es, iterations)

            _, cnts, _ = cv2.findContours(
                mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bounds = self.nms_cnts(cnts, mask, min_area)

            for b in bounds:

                x, y, w, h = b

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if not is_camera:

                sleep(self.time)

            cv2.imshow(self.name, frame)  # 在window上显示图片
            if show_test:
                cv2.imshow(self.name+'_frame', mask)  # 边界

            value = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            previous = self.pop(previous, value)

            cv2.waitKey(10)

            if cv2.getWindowProperty(self.name, cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

            if show_test and cv2.getWindowProperty(self.name+'_frame', cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

        # 释放摄像头
        cap.release()
        cv2.destroyAllWindows()

    def nms_cnts(self, cnts, mask, min_area):

        bounds = [cv2.boundingRect(
            c) for c in cnts if cv2.contourArea(c) > min_area]

        if len(bounds) == 0:
            return []

        scores = [self.calculate(b, mask) for b in bounds]

        bounds = np.array(bounds)

        scores = np.expand_dims(np.array(scores), axis=-1)

        keep = py_cpu_nms(np.hstack([bounds, scores]), self.nms_threshold)

        return bounds[keep]

    def calculate(self, bound, mask):

        x, y, w, h = bound

        area = mask[y:y+h, x:x+w]

        pos = area > 0 + 0

        score = np.sum(pos)/(w*h)

        return score

    def pop(self, l, value):

        l.pop(0)
        l.append(value)

        return l

    def process_image(self, image):
        pass


if __name__ == "__main__":

    detector = Detector()

    detector.catch_video('./test.avi', bias_num=2, iterations=3,
                         k_size=5, show_test=True, enhance=False,
                         logical='OR')