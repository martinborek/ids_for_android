#!/usr/bin/env python3

# Author: Martin Borek (mborekcz@gmail.com)
# Date: Mar/2017

import errors
import logging

# logging.basicConfig(level=logging.INFO)


class FeatureVector:
    def __init__(self, feature_dict, system_calls_num, normalise=False):
        self.feature_dict = feature_dict
        self.system_calls_num = system_calls_num

        if not isinstance(normalise, bool):
            raise errors.ArgError("'normalise' has to be boolean")
        self.normalise = normalise

    def get_normalised(self):
        return [(k, (self.feature_dict[k] / self.system_calls_num)) for k in
                sorted(self.feature_dict, key=self.feature_dict.get, reverse=True)]

    def get_original(self):
        return [(k, self.feature_dict[k]) for k in
                sorted(self.feature_dict, key=self.feature_dict.get, reverse=True)]

    def __len__(self):
        return len(self.feature_dict)

    def __repr__(self):
        return "FeatureVector(system_calls_num=%r, normalise=%r)" % (self.system_calls_num, self.normalise)

    def __str__(self):
        if self.normalise:
            vector = self.get_normalised()
        else:
            vector = self.get_original()

        string = ""
        for call in vector:
            string += "{}: {}\n".format(*call)
        return string


class Histogram(FeatureVector):
    def __repr__(self):
        return "Histogram(system_calls_num=%r, normalise=%r)" % (self.system_calls_num, self.normalise)


class Ngram(FeatureVector):
    def __repr__(self):
        return "Histogram(system_calls_num=%r, normalise=%r)" % (self.system_calls_num, self.normalise)


class CoOccurrenceMatrix(FeatureVector):
    def __repr__(self):
        return "CoOccurrenceMatrix(system_calls_num=%r, normalise=%r)" % (self.system_calls_num, self.normalise)


class LogParser:
    def __init__(self, input_log, time_included=False):
        self._input_log = input_log
        self.system_calls_num = 0
        self._s_call_list = []

        # Position of system call names in the log
        if time_included:
            call_position = 2
        else:
            call_position = 1

        # Parse system call names
        logging.info("Parsing system call names...")
        for line in self._input_log:
            part = line.split(maxsplit=call_position)  # No need to split after the system call name
            part_s_call = part[call_position]

            if part_s_call[0] in ('<', '+', '-'):
                continue  # Not a new system call
            else:
                pos = part_s_call.find('(')
                s_call = part_s_call[:pos]
                self._s_call_list.append(s_call)
                self.system_calls_num += 1

    def histogram(self, normalise=False):

        histogram_dict = {}
        for call in self._s_call_list:
            histogram_dict[call] = histogram_dict.get(call, 0) + 1

        histogram = Histogram(histogram_dict, self.system_calls_num, normalise)

        return histogram

    def ngram(self, n, normalise=False):

        ngram_dict = {}
        for i in range(0, len(self._s_call_list) - n + 1):
            identifier = tuple(self._s_call_list[i:i + n])
            ngram_dict[identifier] = ngram_dict.get(identifier, 0) + 1

        ngram = Ngram(ngram_dict, self.system_calls_num, normalise)

        return ngram

    def co_occurrence_matrix(self, offset, normalise=False):

        com_dict = {}

        for i in range(0, len(self._s_call_list)):
            for j in range(i, min(i+offset, len(self._s_call_list))):
                identifier = (self._s_call_list[i], self._s_call_list[j])
                com_dict[identifier] = com_dict.get(identifier, 0) + 1

        com = CoOccurrenceMatrix(com_dict, self.system_calls_num, normalise)

        return com
