import pprint

import numpy as np


class FormatPrinter(pprint.PrettyPrinter):
    def __init__(self, formats):
        super(FormatPrinter, self).__init__(width=120)
        self.formats = formats

    def format(self, obj, ctx, maxlvl, lvl):
        if type(obj) in self.formats:
            return self.formats[type(obj)] % obj, 1, 0
        return pprint.PrettyPrinter.format(self, obj, ctx, maxlvl, lvl)


pfmt = FormatPrinter({float: "%.3g", np.float32: "%.3g", np.float64: "%.3g"})
