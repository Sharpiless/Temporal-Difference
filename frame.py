import cv2
import numpy as np
from nms import py_cpu_nms
from time import sleep


class Detector(object):

    def __init__(self, name='my_video', frame_num=10, k_size=7, color=(0, 255, 0)):

        self.name = name

        self.color = color

        self.nms_threshold = 0.3

        self.time = 1/frame_num  # 频率

        self.es = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size))

    def catch_video(self, video_index=0, method='T', k_size=7, dialta=0.95,
                    iterations=3, threshold=20, bias_num=1, logical='or',
                    min_area=360, show_test=True, nms=True):

        # video_index：摄像头索引（数字）或者视频路径（字符路径）
        # k_size：中值滤波的滤波器大小
        # dialta：背景权重
        # iteration：腐蚀+膨胀的次数，0表示不进行腐蚀和膨胀操作
        # threshold：二值化阙值
        # bias_num：计算帧差图时的帧数差
        # min_area：目标的最小面积
        # show_test：是否显示二值化图片
        # nms：是否进行非极大值抑制

        method = method.lower()

        if not bias_num > 0:
            raise Exception('bias_num must > 0')

        if isinstance(video_index, str):
            self.is_camera = False
            # 如果是视频，则需要调整帧率
        else:
            self.is_camera = True

        cap = cv2.VideoCapture(video_index)  # 创建摄像头识别类

        if not cap.isOpened():
            # 如果没有检测到摄像头，报错
            raise Exception('Check if the camera is on.')

        self.frame_num = 0

        if method == 'tr':

            self.background = []

        if method == 'g':

            self.mog = cv2.createBackgroundSubtractorMOG2()

        while cap.isOpened():

            if method == 't':

                mask, frame = self.temporal_difference(
                    cap, k_size, dialta, iterations, threshold,
                    bias_num, min_area, show_test, nms)

            elif method == 'b':

                mask, frame = self.weights_bk(
                    cap, k_size, dialta, iterations, threshold,
                    bias_num, min_area, show_test, nms)

            elif method == 'tr':

                mask, frame = self.tri_temporal_difference(
                    cap, k_size, dialta, iterations, threshold,
                    bias_num, min_area, show_test, nms, logical)

            elif method == 'g':

                mask, frame = self.gaussian_bk(cap, k_size, iterations)

            else:

                raise Exception(
                    'method must be \'T\' or \'Tr\' or \'B\' or \'G\'')

            _, cnts, _ = cv2.findContours(
                mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bounds = self.nms_cnts(cnts, mask, min_area, nms=nms)

            for b in bounds:

                x, y, w, h = b

                thickness = (w*h)//min_area

                thickness = thickness if thickness <= 3 else 3
                thickness = thickness if thickness >= 1 else 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), self.color, thickness)

            if not self.is_camera:

                sleep(self.time)

            # 在window上显示背景
            # cv2.imshow(self.name+'_bk', background)
            cv2.imshow(self.name, frame)  # 在window上显示图片
            if show_test:
                cv2.imshow(self.name+'_frame', mask)  # 边界

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

    def gaussian_bk(self, cap, k_size=7, iterations=3):

        catch, frame = cap.read()  # 读取每一帧图片

        if not catch:

            if self.is_camera:

                raise Exception('Unexpected Error.')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.mog.apply(gray)

        mask = cv2.medianBlur(mask, k_size)

        mask = cv2.dilate(mask, self.es, iterations)
        mask = cv2.erode(mask, self.es, iterations)

        return mask, frame

    def weights_bk(self, cap, k_size=7, dialta=0.95,
                   iterations=3, threshold=20, bias_num=1,
                   min_area=360, show_test=True, nms=True):

        catch, frame = cap.read()  # 读取每一帧图片

        if not catch:

            if self.is_camera:

                raise Exception('Unexpected Error.')

        if self.frame_num < bias_num:

            value = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.background = value

            self.frame_num += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw = cv2.GaussianBlur(gray, (3, 3), -1e-5)

        gray = np.abs(gray-self.background).astype('uint8')

        self.background = self.background*dialta + raw*(1-dialta)  # 这里有问题
        # self.background = self.background.astype('uint8')

        gray = cv2.medianBlur(gray, k_size)

        _, mask = cv2.threshold(
            gray, threshold, 255, cv2.THRESH_BINARY)

        # mask = cv2.medianBlur(mask, k_size)

        mask = cv2.dilate(mask, self.es, iterations)
        mask = cv2.erode(mask, self.es, iterations)

        return mask, frame

    def temporal_difference(self, cap, k_size=7, dialta=0.95,
                            iterations=3, threshold=20, bias_num=1,
                            min_area=360, show_test=True, nms=True):

        catch, frame = cap.read()  # 读取每一帧图片

        if not catch:

            if self.is_camera:

                raise Exception('Unexpected Error.')

        if self.frame_num < bias_num:

            value = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.background.append(value)

            self.frame_num += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw = gray.copy()

        gray = cv2.absdiff(gray, self.background[0])

        gray = cv2.medianBlur(gray, k_size)

        _, mask = cv2.threshold(
            gray, threshold, 255, cv2.THRESH_BINARY)

        # mask = cv2.medianBlur(mask, k_size)

        mask = cv2.dilate(mask, self.es, iterations)
        mask = cv2.erode(mask, self.es, iterations)

        self.background = self.pop(self.background, raw)

        return mask, frame

    def tri_temporal_difference(self, cap, k_size=7, dialta=0.95,
                                iterations=3, threshold=20, bias_num=1,
                                min_area=360, show_test=True, nms=True,
                                logical='or'):

        catch, frame = cap.read()  # 读取每一帧图片

        if not catch:

            if self.is_camera:

                raise Exception('Unexpected Error.')

        if self.frame_num < bias_num:

            value = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.background = [value]*bias_num

            self.frame_num = bias_num

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray1 = cv2.absdiff(gray, self.background[0])
        gray1 = cv2.medianBlur(gray1, k_size)

        _, mask1 = cv2.threshold(
            gray1, threshold, 255, cv2.THRESH_BINARY)

        gray2 = cv2.absdiff(gray, self.background[1])
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

        mask = cv2.dilate(mask, self.es, iterations)
        mask = cv2.erode(mask, self.es, iterations)

        self.background = self.pop(self.background, gray)

        return mask, frame

    def nms_cnts(self, cnts, mask, min_area, nms=True):
        # 对检测到的边界框使用非极大值抑制
        bounds = [cv2.boundingRect(
            c) for c in cnts if cv2.contourArea(c) > min_area]

        if len(bounds) == 0:
            return []

        if not nms:
            return bounds

        scores = [self.calculate(b, mask) for b in bounds]

        bounds = np.array(bounds)

        scores = np.expand_dims(np.array(scores), axis=-1)

        keep = py_cpu_nms(np.hstack([bounds, scores]), self.nms_threshold)

        return bounds[keep]

    def calculate(self, bound, mask):

        x, y, w, h = bound

        area = mask[y:y+h, x:x+w]

        pos = area > 0
        pos = pos.astype(np.float)
        # 得分应与检测框大小也有关系

        score = np.sum(pos)/(w*h)

        return score

    def pop(self, l, value):

        l.pop(0)
        l.append(value)

        return l

    def propose_gaussian(self, num):

        nums = []

        for _ in range(num):

            nums.append(np.random.normal())

        nums = sorted(nums)

        s2b = nums[::2]
        b2s = nums[1::2]

        s2b.extend(b2s[::-1])

        return np.array(s2b) - np.min(s2b)


if __name__ == "__main__":

    detector = Detector(name='test')

    detector.catch_video('./test.avi', method='g', bias_num=2, iterations=2,
                         k_size=5, show_test=True, min_area=240,
                         nms=False, threshold=30, dialta=0.9)
