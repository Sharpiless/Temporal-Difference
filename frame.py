import cv2
import numpy as np


class Detector(object):

    def __init__(self, name='my_video', min_area=1500):

        self.name = name

        self.threshold = 10

        self.min_area = min_area

        self.es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))

    def catch_video(self, video_index=0):

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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.absdiff(gray, previous)

            median = cv2.medianBlur(gray, 5)

            ret, mask = cv2.threshold(
                gray, self.threshold, 255, cv2.THRESH_BINARY)

            mask = cv2.erode(mask, self.es, iterations=1)
            mask = cv2.dilate(mask, self.es, iterations=1)

            _, cnts, _ = cv2.findContours(
                mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

            for c in cnts:

                if cv2.contourArea(c) < self.min_area:
                    continue

                x, y, w, h = cv2.boundingRect(c)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow(self.name, frame)  # 在window上显示图片
            cv2.imshow(self.name+'_frame', mask)  # 边界

            previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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


if __name__ == "__main__":

    detector = Detector()

    detector.catch_video()
