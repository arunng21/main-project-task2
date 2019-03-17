import cv2
import numpy
import math
import matplotlib.pyplot as plt


def hough(img):
    width, height = img.shape
    theta = numpy.deg2rad(numpy.arange(-90.0, 90.0, 1))
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = numpy.linspace(-diag_len, diag_len, diag_len * 2)
    cos_t = numpy.cos(theta)
    sin_t = numpy.sin(theta)
    num_thetas = len(theta)

    accumulator = numpy.zeros((2 * diag_len, num_thetas), dtype=numpy.uint8)
    edges = img > 0
    y_idxs, x_idxs = numpy.nonzero(edges)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, theta, rhos


def show_hough_line(accumulator, thetas, rhos):
    plt.imshow(accumulator, cmap='gray', extent=[numpy.rad2deg(thetas[-1]), numpy.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    plt.title('Hough transform')
    plt.xlabel('Angles (degrees)')
    plt.ylabel('Distance (pixels)')
    plt.axis('image')
    plt.show()


def getlength(p1, p2):
    len = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return len


def linelen(img, acc):
    length = 0
    max = numpy.amax(acc, axis=1)
    print("Line end points are:")
    for i in max:
        if i > 230:
            result = numpy.where(acc == i)
            cords = list(zip(result[0], result[1]))
            for cord in cords:
                lines = []
                width, height = img.shape
                theta = numpy.deg2rad(numpy.arange(-90.0, 90.0, 1))
                diag_len = int(round(math.sqrt(width * width + height * height)))
                cos_t = numpy.cos(theta)
                sin_t = numpy.sin(theta)
                num_thetas = len(theta)
                edges = img > 0
                y_idxs, x_idxs = numpy.nonzero(edges)

                for j in range(len(x_idxs)):
                    x = x_idxs[j]
                    y = y_idxs[j]

                    for t_idx in range(num_thetas):
                        if t_idx!=cord[1]:
                            continue
                        rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
                        if rho == cord[0]:
                            lines.append((x, y))
                print lines[0], lines[-1]
                length = length + getlength(lines[0], lines[-1])
    print "Total length: ", int(round(length))


image = cv2.imread("lines.jpg")
cv2.imshow("image", image)
cv2.waitKey(0)
greyimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurimg = cv2.GaussianBlur(greyimg, (5, 5), 0)
edgeimg = cv2.Canny(blurimg, 50, 150)
acc, theta, rho = hough(edgeimg)
cv2.imshow("image", edgeimg)
cv2.waitKey(0)
linelen(edgeimg, acc)
show_hough_line(acc, theta, rho)
