# wsi_core/annotations.py
from __future__ import annotations
import numpy as np
from xml.dom import minidom
from typing import List

Array = np.ndarray


def load_tumor_xml(xml_path: str) -> List[Array]:

    def _to_contour(coord_list) -> Array:
        return np.array([[[
            int(float(c.attributes['X'].value)),
            int(float(c.attributes['Y'].value))
        ]] for c in coord_list],
                        dtype=np.int32)

    xmldoc = minidom.parse(xml_path)
    coords = [
        anno.getElementsByTagName('Coordinate')
        for anno in xmldoc.getElementsByTagName('Annotation')
    ]
    contours = [_to_contour(cl) for cl in coords]
    contours = sorted(contours, key=lambda c: cv_np_area(c), reverse=True)
    return contours


def load_tumor_txt(txt_path: str) -> List[Array]:
    with open(txt_path, "r") as f:
        annot = eval(f.read())  # mirrors original CLAM IO
    all_cnts: List[Array] = []
    for group in annot:
        cgs = group['coordinates']
        if group['type'] == 'Polygon':
            for contour in cgs:
                cnt = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
                all_cnts.append(cnt)
        else:
            for sgmt_group in cgs:
                flat = []
                for seg in sgmt_group:
                    flat.extend(seg)
                cnt = np.array(flat, dtype=np.int32).reshape(-1, 1, 2)
                all_cnts.append(cnt)
    all_cnts = sorted(all_cnts, key=lambda c: cv_np_area(c), reverse=True)
    return all_cnts


# small helper to avoid importing cv2 here
def cv_np_area(contour: Array) -> float:
    import cv2
    return cv2.contourArea(contour)
