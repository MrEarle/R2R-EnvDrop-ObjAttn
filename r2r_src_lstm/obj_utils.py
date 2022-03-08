import numpy as np


def get_obj_coords(ix, elevation, heading, WIDTH=640, HEIGHT=480):
    h = ix % 12
    e = 2 - (ix // 12)
    min_h, max_h = (h - 1) * np.radians(30), (h + 1) * np.radians(30)
    min_e, max_e = -(np.radians(30) * e), np.radians(60) - (np.radians(30) * e)

    if min_h < -np.pi:
        min_h += 2 * np.pi
    if max_h > np.pi:
        max_h -= 2 * np.pi

    # return min_h, max_h, min_e, max_e

    if min_h < max_h and (min_h < heading < max_h):
        x = int(((heading - min_h) / (max_h - min_h)) * WIDTH)

    elif min_h > max_h and (heading > min_h or heading < max_h):
        _max_h = max_h + 2 * np.pi

        _heading = heading
        if heading < max_h:
            _heading += 2 * np.pi

        x = int(((_heading - min_h) / (_max_h - min_h)) * WIDTH)

        if x > WIDTH or x < 0:
            return None, None

    else:
        return None, None

    if min_e < elevation < max_e:
        y = (elevation - min_e) / (max_e - min_e)
        y = int((1 - y) * HEIGHT)

    else:
        return None, None

    return x, y
