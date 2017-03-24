#!/usr/bin/env python3

# Author: Martin Borek (mborekcz@gmail.com)
# Date: Mar/2017


class InputError(Exception):
    # TODO: test if necessary
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ParamError(Exception):
    # TODO: test if necessary
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
