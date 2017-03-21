#!/usr/bin/env python3

# Author: Martin Borek (mborekcz@gmail.com)
# Date: 21/Mar/2017

import argparse

class Params:
    def __init__(self):
        self.mode = None
        self.time_include = False
        self.input_file = None
        self.output_file = None

    def _print_help(self):
        print("""Strace Log Parser:
Params:
    --help Help
    --input=filename Input file
    --output=filename Output file
""")

    def get_args(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("input", help="Input file")
        arg_parser.add_argument("output", help="Output file")


def main():






main()