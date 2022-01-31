import numpy as np


class ExperienceBuffer:

    def __init__(self, size, n_elements=1):
        self.size = size
        self.elements = tuple([[] for __ in range(0, n_elements)])

    def insert(self, *args):
        for i in range(0, len(args)):
            self.elements[i].append(args[i])
            if len(self.elements[i]) > self.size:
                self.elements[i].pop(0)

    def get(self):
        return tuple([np.concatenate(self.elements[i], axis=0) for i in range(0, len(self.elements))])



class SVGD_ExperienceBuffer(ExperienceBuffer):

    def insert(self, *args):
        for i in range(0, len(args)):
            self.elements[i].append(args[i])
            np_el = np.concatenate(self.elements[i], axis=0)
            while np_el.size > self.size:
                if self.elements[i][0].size > np_el.size - self.size:
                    self.elements[i][0] = self.elements[i][0][:self.size]
                else:
                    self.elements[i].pop(0)
                np_el = np.concatenate(self.elements[i], axis=0)

    def remove(self, n: int):
        for element in self.elements:
            del element[n]

    def remove_multiple(self, ns: np.ndarray):
        assert len(self.elements[0]) == ns.shape[0]
        for i in range(0, len(self.elements)):
            for j in range(0, ns.shape[0]):
                self.elements[i][j] = self.elements[i][j][ns[j]]


class SVGD_ExperienceBuffer2(ExperienceBuffer):
    def __init__(self, size, n_elements=1):
        self.size = size
        self.elements = list((None for _ in range(0, n_elements)))

    def get(self):
        return tuple(self.elements)

    def insert(self, *args):
        for i in range(0, len(args)):
            if self.elements[i] is None:
                self.elements[i] = args[i]
            else:
                self.elements[i] = np.concatenate((self.elements[i], args[i]), axis=0)
        assert self.elements[0].shape[0] <= self.size

    def keep_specific_indices(self, ns: np.ndarray):
        for i in range(0, len(self.elements)):
            self.elements[i] = self.elements[i][ns]

    def get_specific(self, i: int):
        return self.elements[i]