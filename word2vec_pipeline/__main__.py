#! /usr/bin/env python
"""
Usage: w2v INPUT [-o OUTPUT|-t] [--condense] [--nocopy] [--verbose] [--watch=<kn>]

-h --help     Show this help
-o, --output  FILE specify output file [default: INPUT.html]
-t, --term    Output just the slides to stdout
--watch=<kn>  Continuously rebuild on changes to input every n seconds [default: once]
--condense    Don't pretty-print the output [default: False]
--nocopy      Don't copy the static files: css, js, etc [default: False]
--verbose     Print more text [default: False]
"""

import os
import time
from docopt import docopt


def main():
    args = docopt(__doc__)
    f_md = args["INPUT"]

    if not os.path.exists(f_md):
        raise IOError("{} not found".format(f_md))

    if args["OUTPUT"] is None:
        f_base = os.path.basename(f_md)
        args["OUTPUT"] = '.'.join(f_base.split('.')[:-1]) + '.html'

    if args["--watch"] == 'once':
        build(args)
        exit()

    while True:
        build(args)
        time.sleep(float(args["--watch"]))


if __name__ == "__main__":
    main()
