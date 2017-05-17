#!/usr/bin/env python3

# Author: Martin Borek (mborekcz@gmail.com)
# Date: Mar/2017

import argparse
import sys
from log_parser import LogParser
import errors
import logging

import json


class Params:
    def __init__(self):
        self.input_log = None
        self.syscalls_list = None #  Work only with these syscalls when processing logs
        self.output = sys.stdout
        self.time_included = False
        self.normalise = False
        self.json = None
        self.histogram = None
        self.ngram = None
        self.co_occurrence_matrix = None

    def __del__(self):
        self.cleanup()

    def __repr__(self):
        return "Params(input_log=%r, syscalls_list=%r, output=%r, time_included=%r, normalise=%r, json=%r," \
               "histogram=%r, ngram=%r, co_occurrence_matrix=%r)" % (self.input_log, self.syscalls_list,
                                                                     self.output_log, self.time_included,
                                                                     self.normalise, self.json, self.histogram,
                                                                     self.ngram, self.co_occurrence_matrix)

    def _open_input(self, filename):

        try:
            self.input_log = open(filename, "r", encoding="utf-8")
        except ValueError:
            raise errors.InputError("Input file encoding is not supported.")
        except:
            raise errors.InputError("Input file couldn't be opened.")

    def _open_output(self, filename):

        try:
            self.output = open(filename, "w", encoding="utf-8")
        except ValueError:
            raise errors.InputError("Output file encoding is not supported.")
        except:
            raise errors.InputError("Output file couldn't be opened.")

    def cleanup(self):
        if self.input_log is not None and not self.input_log.closed:
            self.input_log.close()

        if self.output is not sys.stdout and not self.output.closed:
            self.output.close()

    # Load list of system calls from a file
    def _read_syscalls(self, filename):
        try:
            with open(filename, 'r') as file:
                self.syscalls_list = [line[:-1] for line in file]
        except:
            raise errors.InputError("Syscalls file couldn't be opened.")

    def get_args(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("input", help="Input file")
        arg_parser.add_argument("--output", help="Output file; if not specified, standard output is used.")
        arg_parser.add_argument("--syscalls", help="Input file with system calls to include in the processing; each"
                                                   "system call is on a new line and system calls should not repeat")
        arg_parser.add_argument("--timeincluded", action="store_true",
                                help="Is time information included in the input log?")
        arg_parser.add_argument("--normalise", action="store_true",
                                help="If selected, result is normalised")
        arg_parser.add_argument("--json", action="store_true",
                                help="If selected, output is in JSON format.")
        mode_group = arg_parser.add_mutually_exclusive_group()
        mode_group.add_argument("--histogram", help="Histogram", action="store_true")
        mode_group.add_argument("--ngram", help="Ngram and its length", type=int)
        mode_group.add_argument("--comatrix", help="Co-occurrence matrix and its offset", type=int)
        args = arg_parser.parse_args()

        self.time_included = args.timeincluded
        self.normalise = args.normalise
        self.json = args.json
        self.histogram = args.histogram

        if args.syscalls is not None:
            self._read_syscalls(args.syscalls)

        if args.ngram is not None:
            if args.ngram < 2:
                raise errors.ParamError("--ngram needs to have value greater than 1.")
        self.ngram = args.ngram

        if args.comatrix is not None:
            if args.comatrix < 2:
                raise errors.ParamError("--comatrix needs to have an offset greater than 1.")
        self.co_occurrence_matrix = args.comatrix

        self._open_input(args.input)
        if args.output is not None:
            self._open_output(args.output)


def main():
    """MAIN PROGRAM"""

    logging.info("Script starting...")
    params = Params()
    logging.info("Parsing parameters...")
    try:
        params.get_args()

    except (errors.InputError, errors.ParamError) as e:
        sys.stderr.write(str(e) + '\n')
        exit(1)

    logging.info("Parsing system calls log file...")

    # Position of system call names in the log
    if params.time_included:
        column_pos = 1
    else:
        column_pos = 0

    parser = LogParser(params.input_log, column_pos)

    print("Calls analysed: {}".format(parser.system_calls_num)) # TODO

    if params.ngram is not None:
        logging.info("Getting Ngram...")

        ngram = parser.ngram(params.ngram, params.normalise, params.syscalls_list)
        print("Unique sequences: {}".format(len(ngram)))

        print(ngram, file=params.output)

    elif params.co_occurrence_matrix is not None:
        logging.info("Getting Co-occurrence matrix...")

        co_occurrence_matrix = parser.co_occurrence_matrix(params.co_occurrence_matrix, params.normalise,
                                                           params.syscalls_list)
        print(co_occurrence_matrix, file=params.output)

    else:
        if not params.histogram:
            sys.stderr.write("No option selected, using histogram.\n")

        logging.info("Getting Histogram...")
        histogram = parser.histogram(params.normalise, params.syscalls_list)

        print(histogram, file=params.output)

    params.cleanup()

main()
