import cv2
import numpy as np
import sys


def imshow(title: str, image: np.ndarray) -> None:
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_contours_and_print_area(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([10, 150, 150])
    upper_red1 = np.array([30, 255, 255])

    mask = cv2.inRange(img_hsv, lower_red1, upper_red1)
    imshow("Binary Mask", mask)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    imshow("Threshold", thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    tray_contour = contours_sorted[0]

    cv2.drawContours(image, [tray_contour], -1, (0, 255, 0), 3)

    print("Contour area:", cv2.contourArea(tray_contour))

    return image, tray_contour


def find_coins_and_count(image: np.ndarray, min_radius: int, max_radius: int, contour: np.ndarray,
                         name_of_looking_coin: str) -> None:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    coins = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT,
                             dp=1, minDist=20, param1=100, param2=50, minRadius=min_radius, maxRadius=max_radius)
    coins = np.uint16(np.around(coins))

    for coin in coins[0, :]:
        cv2.circle(cimg, (coin[0], coin[1]), coin[2], (0, 255, 0), 2)
        cv2.circle(cimg, (coin[0], coin[1]), 2, (0, 0, 255), 3)

    imshow("Coins", cimg)

    counter = int()

    for coin in coins[0, :]:
        center = (int(coin[0]), int(coin[1]))
        result = cv2.pointPolygonTest(contour, center, False)
        if result >= 0:
            cv2.circle(cimg, center, coin[2], (0, 255, 0), 2)
            counter += 1
        else:
            cv2.circle(cimg, center, coin[2], (0, 0, 255), 2)

    imshow("Found coins", cimg)
    print(f"Found {counter} - {name_of_looking_coin} coins")


def main():
    img = cv2.imread("tray8.jpg")
    if img is None:
        sys.exit("Could not read the image.")
    imshow("Original", img)

    img_contour, contour = find_contours_and_print_area(img)
    imshow("Finding", img_contour)

    find_coins_and_count(img, 0, 32, contour, name_of_looking_coin="Groszówki")
    find_coins_and_count(img, 32, 45, contour, name_of_looking_coin="Złotówki")


if __name__ == "__main__":
    main()
