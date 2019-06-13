"""
Script to show off the noise rejection part of Network Noise Rejection. Data taken from the original Network Noise Rejection repo: https://github.com/mdhumphries/NetworkNoiseRejection
"""
import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import __init__ as nnr # network noise rejection
import numpy as np
import datetime as dt
