import numpy as np


def one_hot_to_map(onehot: np.ndarray):
    to_print = ""
    for col in onehot.transpose((1, 0, 2)):
        for cell in col:
            if cell[0] == 1:
                to_print += ">"
            elif cell[1] == 1:
                to_print += "V"
            elif cell[2] == 1:
                to_print += "<"
            elif cell[3] == 1:
                to_print += "^"
            elif cell[4] == 1:
                to_print += "W"
            elif cell[5] == 1:
                to_print += "G"
            else:
                to_print += " "
        to_print += "\n"
    return to_print
