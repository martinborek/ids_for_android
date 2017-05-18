#!/usr/bin/env python3

# Author: Martin Borek (mborekcz@gmail.com)
# Date: Mar/2017

import errors
import logging
from itertools import product

# logging.basicConfig(level=logging.INFO)


class FeatureVector:
    def __init__(self, feature_dict, system_calls_num, normalise=False, sort_alphabetically=False):
        self.feature_dict = feature_dict
        self.system_calls_num = system_calls_num
        self.sort_alphabetically = sort_alphabetically

        if not isinstance(normalise, bool):
            raise errors.ArgError("'normalise' has to be boolean")
        self.normalise = normalise

    def get_normalised(self):
        if self.sort_alphabetically:
            sorted_feature_list = sorted(self.feature_dict)
        else:
            sorted_feature_list = sorted(self.feature_dict, key=self.feature_dict.get, reverse=True)

        return [(k, (self.feature_dict[k] / self.system_calls_num)) for k in sorted_feature_list]

    def get_original(self):
        if self.sort_alphabetically:
            sorted_feature_list = sorted(self.feature_dict)
        else:
            sorted_feature_list = sorted(self.feature_dict, key=self.feature_dict.get, reverse=True)

        return [(k, self.feature_dict[k]) for k in sorted_feature_list]

    def get_values(self):
        # The returned values have to be always in the same order. For this purpose, keys are sorted alphabetically
        # These values can be used as a feature vector for training a model (e.g. in SVM)

        if self.normalise:
            return [str(self.feature_dict[k] / self.system_calls_num) for k in sorted(self.feature_dict)]
        else:
            return [str(self.feature_dict[k]) for k in sorted(self.feature_dict)]

    def get_csv_values(self):
        # The returned values have to be always in the same order. For this purpose, keys are sorted alphabetically
        # These values can be used as a feature vector for training a model (e.g. in SVM)

        if self.normalise:
            csv_string = ','.join([str(self.feature_dict[k] / self.system_calls_num) for k in sorted(self.feature_dict)])
        else:
            csv_string = ','.join([str(self.feature_dict[k]) for k in sorted(self.feature_dict)])

        return csv_string

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
    def __str__(self):
        return "\n".join(self._s_call_list)

    def __init__(self, input_log, column_pos=0):
        # column_pos specifies at which column in the log the system call name is
        self._input_log = input_log
        self.system_calls_num = 0
        self._s_call_list = []

        # Parse system call names
        logging.info("Parsing system call names...")
        for line in self._input_log:
            part = line.split(maxsplit=column_pos)  # No need to split after the system call name
            part_s_call = part[column_pos]

            if part_s_call[0] in ('<', '+', '-'):
                continue  # Not a new system call
            else:
                pos = part_s_call.find('(')
                s_call = part_s_call[:pos]
                self._s_call_list.append(s_call)
                self.system_calls_num += 1

    def histogram(self, normalise=False, syscalls_list=None):

        syscalls_selected = (syscalls_list is not None)
        if syscalls_selected:
            # Specified system calls should be included even if they did not occur in the log file
            histogram_dict = {syscall: 0 for syscall in syscalls_list}
        else:
            histogram_dict = {}

        for call in self._s_call_list:

            if syscalls_selected and call not in histogram_dict:
                # Only specified system calls should be included
                continue

            histogram_dict[call] = histogram_dict.get(call, 0) + 1

        histogram = Histogram(histogram_dict, self.system_calls_num, normalise, syscalls_selected)

        return histogram

    def ngram(self, n, normalise=False, syscalls_list=None):
        syscalls_selected = (syscalls_list is not None)
        if syscalls_selected:
            # Specified system calls should be included even if they did not occur in the log file.
            # Create all n combinations of system calls from the list
            ngram_dict = {combination: 0 for combination in product(syscalls_list, repeat=n)}
        else:
            ngram_dict = {}

        for i in range(0, len(self._s_call_list) - n + 1):
            identifier = tuple(self._s_call_list[i:i + n])

            if syscalls_selected and identifier not in ngram_dict:
                # Only specified system calls should be included
                continue

            ngram_dict[identifier] = ngram_dict.get(identifier, 0) + 1

        ngram = Ngram(ngram_dict, self.system_calls_num, normalise, syscalls_selected)

        return ngram

    def co_occurrence_matrix(self, offset, normalise=False, syscalls_list=None):
        syscalls_selected = (syscalls_list is not None)
        if syscalls_selected:
            # Specified system calls should be included even if they did not occur in the log file.
            # Create all pairs of system calls from the list
            com_dict = {(syscall_one, syscall_two): 0 for syscall_one in syscalls_list for syscall_two in
                        syscalls_list}
        else:
            com_dict = {}

        for i in range(0, len(self._s_call_list)):
            for j in range(i, min(i+offset, len(self._s_call_list))):
                identifier = (self._s_call_list[i], self._s_call_list[j])

                if syscalls_selected and identifier not in com_dict:
                    # Only specified system calls should be included
                    continue

                com_dict[identifier] = com_dict.get(identifier, 0) + 1

        com = CoOccurrenceMatrix(com_dict, self.system_calls_num, normalise, syscalls_selected)

        return com
