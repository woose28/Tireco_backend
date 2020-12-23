from pytesseract import *
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import re
import pandas as pd
from typing import List, Tuple


class Tireco:
    boundaries: List[Tuple] = [
        ([103, 119, 225], [133, 149, 255]),
        ([90, 180, 221], [120, 210, 251]),
        ([97, 187, 152], [127, 217, 182]),
        ([178, 194, 110], [208, 224, 140]),
        ([218, 150, 107], [248, 180, 137]),
        ([89, 155, 236], [119, 185, 255]),
        ([210, 119, 144], [240, 149, 174]),
        ([121, 187, 105], [151, 217, 135]),
        ([222, 136, 196], [252, 166, 226]),
        ]

    CUTTING_POS: int = 45
    DENOISING_DST = None
    DENOISING_H: float = 5
    DENOISING_HCOLOR: float = 5
    DENOISING_TEMP_WINS: int = 7
    DENOISING_SEARCH_WINS: int = 21
    THRESHOLD_THRESHOLD_VAL: int = 38
    THRESHOLD_VAL: int = 255
    CONFIG_PSM_6: str = "-l kor+eng --psm 6"
    CONFIG_PSM_11: str = "-l kor+eng --psm 11"

    def __init__(self):
        pass

    def extract_title_with_img_file(self, img_file) -> List[str]:
        encoded_img = np.fromstring(img_file, dtype=np.uint8)
        img = cv.imdecode(encoded_img, cv.IMREAD_COLOR)

        img, img_gray, img_thresh = self.__prepro_img(img)

        return self.__detect_color_box(img, img_thresh)

    def extract_title_with_img_path(self, img_path) -> List[str]:
        img = cv.imread(img_path)

        img, img_gray, img_thresh = self.__prepro_img(img)

        return self.__detect_color_box(img, img_thresh)

    def __prepro_img(self, img) -> Tuple:
        img = img[self.CUTTING_POS:, self.CUTTING_POS:]

        img = cv.fastNlMeansDenoisingColored(img, self.DENOISING_DST, self.DENOISING_H, self.DENOISING_HCOLOR,
                                             self.DENOISING_TEMP_WINS, self.DENOISING_SEARCH_WINS)

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gray = 255 - img_gray

        ret, img_thresh = cv.threshold(img_gray, self.THRESHOLD_THRESHOLD_VAL, self.THRESHOLD_VAL, cv.THRESH_BINARY)

        return img, img_gray, img_thresh

    def __detect_color_box(self, img, img_thresh) -> List[str]:
        title_list: List[str] = []

        for (lower, upper) in self.boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv.inRange(img, lower, upper)
            output = cv.bitwise_and(img, img, mask=mask)

            gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)

            ret, thresh = cv.threshold(gray, self.THRESHOLD_THRESHOLD_VAL, self.THRESHOLD_VAL, cv.THRESH_BINARY)

            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            title_list += self.__extract(img_thresh, contours, hierarchy)

        return title_list

    def __extract(self, img_thresh, contours, hierarchy) -> List[str]:
        title_list = []
        for idx in range(len(contours)):
            if hierarchy[0][idx][3] == -1 and hierarchy[0][idx][2] != -1:
                x, y, w, h = cv.boundingRect(contours[idx])
                one_subject = img_thresh[y:y + h, x:x + w]

                title_psm_6 = self.__prepro_info(image_to_string(one_subject, config=self.CONFIG_PSM_6))
                title_psm_11 = self.__prepro_info(image_to_string(one_subject, config=self.CONFIG_PSM_11))

                title_list.extend([title_psm_6, title_psm_11])

        return list(set(title_list))

    @staticmethod
    def __prepro_info(info: str) -> str:
        info = info.strip()
        room_info = re.findall("[0-9]{3}-", info)
        if len(info) == 0:
            return ""
        if len(room_info) == 0:
            info = re.findall("[^0-9|a-z]+", info)[0]

        else:
            info = re.sub("\n[0-9]{3}-.*", "", info, flags=re.DOTALL)

        title = re.sub("\n", "", info)
        return title



