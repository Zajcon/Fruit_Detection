import json
from pathlib import Path
from typing import Dict
import numpy as np
import click
import cv2
from tqdm import tqdm


def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.
    Parameters
    ----------
    img_path : str
        Path to processed image.
    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    #image = cv2.imread('data/00.jpg')
    image = cv2.resize(image, None, fx = 0.2, fy = 0.2)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    oranges1 = cv2.inRange(image_hsv, (12, 171, 117), (17, 255, 255))
    oranges2 = cv2.inRange(image_hsv, (12, 75, 236), (30, 140, 255))
    oranges = oranges1 | oranges2
    oranges_contours = find_contours(oranges)
    oranges_count = 0
    for i in range(len(oranges_contours)):
        if calculate_area(oranges_contours[i], oranges) > 8000:
            oranges_count += 1
            cv2.drawContours(image, oranges_contours, i, (255, 0, 0), 5)
    print(oranges_count)

    apples_1 = cv2.inRange(image_hsv, (147, 20, 0), (180, 255, 255))
    apples_2 = cv2.inRange(image_hsv, (0, 50, 0), (16, 255, 255))
    apples_3 = cv2.inRange(image_hsv, (10, 140, 20), (20, 220, 255))
    apples = apples_1 | apples_2 | apples_3
    apples[oranges == 255] = 0
    apples_contours = find_contours(apples)
    apples_count = 0
    for i in range(len(apples_contours)):
        if calculate_area(apples_contours[i], apples) > 7000:
            apples_count += 1
            cv2.drawContours(image, apples_contours, i, (0, 255, 0), 5)
    print(apples_count)

    bananas = cv2.inRange(image_hsv, (20, 100, 50), (39, 255, 255))
    bananas_contours = find_contours(bananas)
    bananas_count = 0
    for i in range(len(bananas_contours)):
        if calculate_area(bananas_contours[i], bananas) > 8000:
            bananas_count += 1
            cv2.drawContours(image, bananas_contours, i, (0, 0, 255), 5)
    print(bananas_count)

    cv2.imshow('image', image)
    cv2.imshow('image', oranges)
    cv2.imshow('apples', apples)
    cv2.imshow('bananas', bananas)
    cv2.waitKey()

    #apple = 0
    #banana = 0
    #orange = 0

    return {'apple': apples_count, 'banana': bananas_count, 'orange': oranges_count}


@click.command()
@click.option('-p', '--data_path', help = 'Path to data directory', type = click.Path(exists = True, file_okay = False,
                                                                                      path_type = Path),
              required = True)
@click.option('-o', '--output_file_path', help = 'Path to output file',
              type = click.Path(dir_okay = False, path_type = Path),
              required = True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


def find_contours(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def calculate_area(contour, mask):
    contour_mask = np.zeros_like(mask)
    contour_mask = cv2.drawContours(contour_mask, [contour], 0, 255, -1)
    return np.count_nonzero(mask[contour_mask == 255])




#if __name__ == '__main__':
#    main()