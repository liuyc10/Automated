import csv
from datetime import datetime
from plc import PLC

import cv2 as cv
import numpy as np
from scipy.spatial.distance import pdist

import calculation_tools


class PlcCurtainTrack(PLC):

    def __init__(self, create_csv=True, save_pic=True, **kwargs):
        super(PlcCurtainTrack, self).__init__(**kwargs)
        self.create_csv = create_csv
        self.save_pic = save_pic

        self.track_coordinate = dict()
        self.track_status = dict()
        self.coordinate_tolerance = 0
        self.track_count = 0
        self.prev_frame = None

        self.f = None
        self.writer = None
        self.set_done = False

    def start_test(self, x, s):
        self.set_done = True
        if self.create_csv:
            self.writer.writeheader()

    def initial_setup(self):

        centers = self.raw_data()
        self.track_coordinate = dict()
        self.track_status = dict()
        num = 0
        for center in centers:
            self.track_coordinate[num] = center
            self.track_status[num] = 1
            num += 1
        self.track_count = len(centers)
        if len(centers) < 2:
            self.coordinate_tolerance = 100
        else:
            distances = pdist(centers, 'euclidean')
            self.coordinate_tolerance = min(min(distances) / 2, 100)
        self.set_done = True

        return self.track_coordinate, self.track_status

    def current_status(self):

        centers = self.raw_data()
        track_status_new = dict()

        for center in centers:
            distances = calculation_tools.distance_P2Ps(center, self.track_coordinate.values())
            min_index = np.argmin(distances)
            if distances[min_index] < self.coordinate_tolerance:
                track_status_new[min_index] = 1

        diff = set(self.track_status) - set(track_status_new)

        if diff:
            for key in diff:
                track_status_new[key] = 0

        for key in self.track_status.keys():
            if self.track_status[key] != track_status_new[key]:
                self.track_status[key] = track_status_new[key]

        return self.track_status

    def run(self):

        if self.create_csv:
            timestamp = self.time(False)
            self.f = open('./data/log_{}.csv'.format(timestamp), mode='w', newline='')

        try:
            while self.v.isOpened():
                rat, self.current_frame = self.v.read()
                if self.current_frame is None:
                    break

                save_img = False
                self.timestamp = self.time()

                if self.skip_frame_no > 0:
                    self.skip_frame_no -= 1
                    self.add_TimeStamp()
                    cv.imshow('frame', self.current_frame)
                    continue

                circles = self.searching()

                centers = np.asarray(
                    [calculation_tools.cal_central_point(circle.reshape(circle.shape[0], 2)) for circle in circles])

                if not self.set_done:
                    self.track_coordinate = dict()
                    self.track_status = dict()
                    num = 0
                    for center in centers:
                        self.track_coordinate[num] = center
                        self.track_status[num] = 1
                        num += 1
                    self.track_count = len(centers)
                    if len(centers) < 2:
                        self.coordinate_tolerance = 40
                    else:
                        distances = pdist(centers, 'euclidean')
                        self.coordinate_tolerance = min(distances) / 2
                    if self.create_csv:
                        heads = ['Time Stamp']
                        heads.extend(self.track_status.keys())
                        self.writer = csv.DictWriter(self.f, heads)
                else:
                    lights_status_new = dict()
                    # if len(centers) > len(lights_status):
                    #    continue

                    for center in centers:
                        distances = calculation_tools.distance_P2Ps(center, self.track_coordinate.values())
                        min_index = np.argmin(distances)
                        if distances[min_index] < self.coordinate_tolerance:
                            lights_status_new[min_index] = 1

                    diff = set(self.track_status) - set(lights_status_new)

                    if diff:
                        for key in diff:
                            lights_status_new[key] = 0

                    for key in self.track_status.keys():
                        if self.track_status[key] != lights_status_new[key]:
                            self.track_status[key] = lights_status_new[key]
                            print('{} Track No. {} status has been changed to {}'.format(
                                datetime.now(),
                                key,
                                "OFF" if self.track_status[key] == 0 else 'ON'))
                            save_img = True
                if self.save_pic:
                    if save_img:
                        self.add_TimeStamp()
                        cv.imwrite('./data/{}.jpg'.format(self.timestamp), self.current_frame)
                        cv.imwrite('./data/{}_p.jpg'.format(self.prev_frame[1]), self.prev_frame[0])

                        if self.create_csv:
                            record = {'Time Stamp': self.timestamp}
                            for key, value in self.track_status.items():
                                record[key] = value
                            self.writer.writerow(record)
                            self.skip_frame_no = 15

                    self.prev_frame = (self.current_frame, self.timestamp)

                cv.imshow('frame', self.current_frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            if self.create_csv:
                self.f.close()
            self.v.release()
