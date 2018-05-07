"""Simple usage example of ask_first.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2018
"""

import argparse
import logging

import ask_first

parser = argparse.ArgumentParser()
parser.add_argument('first_path', help='Path to FIRST data')
parser.add_argument('-v', help='Verbose', action='store_true')
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG if args.v else logging.WARNING,
    format='%(asctime)s-%(name)s-%(levelname)s: %(message)s')

paths = ask_first.read_paths(args.first_path)
# im = get_image((162.5302917, 30.6770889), 3 / 60, paths)
