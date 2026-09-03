import numpy as np
import cv2


def scaleContourDim(contours, scale):
    return [np.array(cont * scale, dtype='int32') for cont in contours]


def scaleHolesDim(contours, scale):
    return [[np.array(hole * scale, dtype='int32') for hole in holes]
            for holes in contours]


def filter_contours(contours, hierarchy, filter_params):
    """
        Filter contours by: area.
    """
    filtered = []

    # find indices of foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    all_holes = []

    # loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # actual contour
        cont = contours[cont_idx]
        # indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        # take contour area (includes holes)
        a = cv2.contourArea(cont)
        # calculate the contour area of each hole
        hole_areas = [
            cv2.contourArea(contours[hole_idx]) for hole_idx in holes
        ]
        # actual area of foreground contour region
        a = a - np.array(hole_areas).sum()
        if a == 0:
            continue
        if tuple((filter_params['a_t'], )) < tuple((a, )):
            filtered.append(cont_idx)
            all_holes.append(holes)

    foreground_contours = [contours[cont_idx] for cont_idx in filtered]

    hole_contours = []

    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids]
        unfiltered_holes = sorted(unfiltered_holes,
                                  key=cv2.contourArea,
                                  reverse=True)
        # take max_n_holes largest holes by area
        unfiltered_holes = unfiltered_holes[:filter_params['max_n_holes']]
        filtered_holes = []

        # filter these holes
        for hole in unfiltered_holes:
            if cv2.contourArea(hole) > filter_params['a_h']:
                filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours
