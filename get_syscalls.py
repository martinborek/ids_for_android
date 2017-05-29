#!/usr/bin/env python3

# Author: Martin Borek (mborekcz@gmail.com)
# Date: May/2017

# Get a list of all system calls that appear in all files in the specified directory
# As input, give directory with FILTERED calls (meaning, just syscall names, no arguments, no return values, no time)

import argparse
import os


class Params:
    def __init__(self):
        self.input_directory = None
        self.num_occurrences = False

    def get_args(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("input_directory", help="Input directory")
        arg_parser.add_argument("--n", action="store_true",
                                help="Show number of occurrences")
        args = arg_parser.parse_args()

        self.input_directory = args.input_directory
        self.num_occurrences = args.n


class SysCalls:
    def __init__(self, show_occurrences):
        self.calls_dictionary = dict()
        self.show_occurrences = show_occurrences

    def _process_file(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:

                # Each call is on a separate line
                call = line[:-1]
                self.calls_dictionary[call] = self.calls_dictionary.get(call, 0) + 1

    def process_directory(self, directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                self._process_file(file_path)

        return self.calls_dictionary

    def __str__(self):

        sorted_calls = [(k, self.calls_dictionary[k]) for k in sorted(self.calls_dictionary,
                                                                      key=self.calls_dictionary.get, reverse=True)]

        string = ""
        for call in sorted_calls:
            if self.show_occurrences:
                string += "{}: {}\n".format(*call)
            else:
                string += "{}\n".format(call[0])

        return string


def main():
    params = Params()
    params.get_args()

    syscalls = SysCalls(params.num_occurrences)
    syscalls.process_directory(params.input_directory)

    print(syscalls, end='')


main()
