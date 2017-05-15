#!/usr/bin/env python3

# Author: Martin Borek (mborekcz@gmail.com)
# Date: May/2017

import argparse
import sys
from log_parser import LogParser
import errors
import logging

import json


class Params:
    def __init__(self):
        self.input_log = None
        self.output = sys.stdout
        self.time_included = False

    def __del__(self):
        self.cleanup()

    def __repr__(self):
        return "Params(input_log=%r, output=%r, time_included=%r)" % (self.input_log, self.output_log, self.time_included)

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
        args = arg_parser.parse_args()

        self.time_included = args.timeincluded

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
    parser = LogParser(params.input_log, params.time_included)

    print("Calls analysed: {}".format(parser.system_calls_num)) # TODO

    print(parser, file=params.output)

    params.cleanup()

main()
