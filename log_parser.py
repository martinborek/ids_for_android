#!/usr/bin/env python3

# Author: Martin Borek (mborekcz@gmail.com)
# Date: Mar/2017


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
        print("Parsing system call names...")
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

    def histogram(self, normalise=True):

        histogram_dict = {}
        for call in self._s_call_list:
            histogram_dict[call] = histogram_dict.get(call, 0) + 1

        if normalise:
            histogram = [(k, (histogram_dict[k] / self.system_calls_num)) for k in
                         sorted(histogram_dict, key=histogram_dict.get, reverse=True)]
        else:
            histogram = [(k, histogram_dict[k]) for k in sorted(histogram_dict, key=histogram_dict.get, reverse=True)]

        return histogram

    def ngram(self, n, normalise=True):

        ngram_dict = {}
        for i in range(0, len(self._s_call_list) - n + 1):
            identifier = tuple(self._s_call_list[i:i + n])
            ngram_dict[identifier] = ngram_dict.get(identifier, 0) + 1

        ngram_num = len(ngram_dict)

        if normalise:
            ngram = [(k, (ngram_dict[k] / ngram_num)) for k in sorted(ngram_dict, key=ngram_dict.get, reverse=True)]
        else:
            ngram = [(k, ngram_dict[k]) for k in sorted(ngram_dict, key=ngram_dict.get, reverse=True)]

        return ngram

    def co_occurrence_matrix(self, offset, normalise=True):

        com_dict = {}

        return

