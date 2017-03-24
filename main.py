#!/usr/bin/env python3

# Author: Martin Borek (mborekcz@gmail.com)
# Date: Mar/2017

import argparse
import sys
from log_parser import LogParser
import errors

import json


class Params:
    def __init__(self):
        self.input_log = None
        self.output = sys.stdout
        self.time_included = False
        self.normalise = False
        self.json = None
        self.histogram = None
        self.ngram = None
        self.co_occurrence_matrix = None

    def __del__(self):
        self.cleanup()

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

    def get_args(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("input", help="Input file")
        arg_parser.add_argument("--output", help="Output file; if not specified, standard output is used.")
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

    print("Script starting..")
    params = Params()
    print("Parsing arguments...")
    try:
        params.get_args()

    except errors.InputError as e:
        sys.stderr.write("Input error: " + e.value + '\n')
        exit(1)

    except errors.ParamError as e:
        sys.stderr.write("Wrong parameters: " + e.value + '\n')
        exit(1)

    print("Arguments parsed")

    parser = LogParser(params.input_log, params.time_included)
    print("Calls analysed: {}".format(parser.system_calls_num))

    if params.ngram is not None:
        print("Getting ngram...")
        ngram = parser.ngram(params.ngram)
        print("Ngram({}):".format(params.ngram))
        print("Unique sequences: {}".format(len(ngram)))
        for call in ngram:
            print("{}: {}".format(*call), file=params.output)

    elif params.co_occurrence_matrix is not None:
        print("Getting co_occurrence matrix...")
        # TODO

    else:
        if not params.histogram:
            print("No option selected, using histogram.")

        print("Getting histogram...")
        histogram = parser.histogram(params.normalise)
        print("Histogram:")
        for call in histogram:
            print("{}: {}".format(*call), file=params.output)

    params.cleanup()

main()
