import numpy as np


class CoordinateQueue(object):

    def __init__(self, size=3):
        self.size = size
        self.__list = []

    def push(self, element):
        self.__list.append(element)
        if len(self.__list) > self.size:
            self.__list.pop(0)

    def __len__(self):
        return len(self.__list)

    def get_list(self):
        return np.array(self.__list)

    def average(self):
        npl = np.asarray(self.__list)
        avg = np.sum(npl, axis=0)/len(self.__list)
        return avg

