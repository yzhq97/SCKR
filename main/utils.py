import argparse
import re

def is_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

def parse_device_str(str):
    outstr = ""
    prefix = str[:3]
    postfix = str[3:]
    if prefix == 'cpu': outstr += '/device:CPU:'
    elif prefix == 'gpu': outstr += '/device:GPU:'
    else: raise Exception("invalid device name")
    if is_int(postfix) and int(postfix) >= 0:
        outstr += postfix
    else: raise Exception("invalid device number")
    return outstr

def parse_name(str):
    pattern = re.compile(r"(\S+)_(\d+)_(\d+).(\S+)")
    match = pattern.match(str)
    start, end = match.regs[1]
    series = str[start:end]
    start, end = match.regs[2]
    number = str[start:end]
    start, end = match.regs[3]
    label = str[start:end]
    return series, number, label