import h5py as h5
import numpy as np

filelist = [
    f"wd/forward/Ele_FixedAngle/EleEscan_{i}_{j}.h5"
    for i in range(1, 9)
    for j in range(1, 11)
]


def data_generator():
    for fn in filelist:
        f = h5.File(fn, "r")
        caloimgs = f["ECAL"]
        for img in caloimgs:
            yield np.swapaxes(img, 0, 2)


data_gen = data_generator()