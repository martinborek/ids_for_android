#!/usr/bin/env python3

# Author: Martin Borek (mborekcz@gmail.com)
# Date: Mar/2017


class CustomError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class InputError(CustomError):
    def __str__(self):
        return "Input error: {}".format(self.value)


class ParamError(CustomError):
    def __str__(self):
        return "Wrong parameters: {}".format(self.value)


class ArgError(CustomError):
    def __str__(self):
        return "Wrong arguments: {}".format(self.value)
