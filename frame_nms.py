import cv2
import numpy as np
from nms import py_cpu_nms
from time import sleep


class Detector(object):

    def __init__(self, name='my_video', min_area=1500, frame_num=10, iterations=3, k_size=7):

        self.name = name

        self.threshold = 20

        self.nms_threshold = 0.3

        self.time = 1/frame_num

        self.iterations = iterations

        self.min_area = min_area

        self.es = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size))

    def catch_video(self, video_index=0, k_size=7):

        if isinstance(video_index, str):

            is_camera = False

        else:

            is_camera = True

        cap = cv2.VideoCapture(video_index)  # 创建摄像头识别类

        if not cap.isOpened():

            # 如果没有检测到摄像头，报错
            raise Exception('Check if the camera is on.')

        frame_num = 0

        while cap.isOpened():

            catch, frame = cap.read()  # 读取每一帧图片

            if not catch:

                raise Exception('Error.')

            if not frame_num:

                previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            raw = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.absdiff(gray, previous)

            median = cv2.medianBlur(gray, k_size)

            ret, mask = cv2.threshold(
                gray, self.threshold, 255, cv2.THRESH_BINARY)

            mask = cv2.dilate(mask, self.es, self.iterations)
            mask = cv2.erode(mask, self.es, self.iterations)

            _, cnts, _ = cv2.findContours(
                mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bounds = self.nms_cnts(cnts, mask)

            for b in bounds:

                x, y, w, h = b

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if not is_camera:

                sleep(self.time)

            cv2.imshow(self.name, frame)  # 在window上显示图片
            cv2.imshow(self.name+'_frame', mask)  # 边界

            previous = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            frame_num += 1

            key = cv2.waitKey(10)

            if key & 0xFF == ord('q'):
                # 按q退出
                break

            if cv2.getWindowProperty(self.name, cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

            if cv2.getWindowProperty(self.name+'_frame', cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

        # 释放摄像头
        cap.release()
        cv2.destroyAllWindows()

    def nms_cnts(self, cnts, mask):

        bounds = [cv2.boundingRect(
            c) for c in cnts if cv2.contourArea(c) > self.min_area]

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


if __name__ == "__main__":

    detector = Detector(min_area=360)

    detector.catch_video('./test.avi')
