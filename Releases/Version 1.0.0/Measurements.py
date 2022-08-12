import numpy as np
import cv2

from skimage.filters import threshold_otsu
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage


class Measure(object):
    """
    Helper class for obtaining different measurements of objects in an image.
    Requires numpy, cv2, scipy and scikit-image.
    For installation, run:
    pip install numpy
    pip install opencv-python
    pip install scipy
    pip install scikit-image
    """
    def __init__(self, img, pixelDistance=1.0, knownDistance=1.0, unit='pixels', threshold=-1.0, darkBackground=False, applyWatershed=True, excludeEdges=True, grayscaleImage=None):
        """
        :param img: Image (as numpy array)
        :param pixelDistance: For calibration: Known distance of an object in pixels
        :param knownDistance: For calibration: Known distance of an object in real units
        :param unit: For calibration: Unit used in the calibration
        :param threshold: If the image is not a black/white image, use this value for the threshold (-1 means Otsu)
        :param darkBackground: Is the background dark or bright (only relevant if the image is thresholded)
        :param excludeEdges: Exclude particles touching the edges of the image
        :param grayscaleImage: The grayscale image (only relevant if img is a mask of a corresponding grayscale image
                               and mean intensities under the mask are needed)
        """
        # If not a grayscale image, convert to grayscale and apply
        # threshold and watershed (needed by cv2.findContours())
        if len(img.shape) != 2:
            self.image = cv2.cvtColor(src=img.copy(), code=cv2.COLOR_BGR2GRAY).copy()
        if np.any((img > 1) & (img < 255)) or np.all((img >= 0) & (img <= 1)):
            self.image = self.segment(img, threshold=threshold, darkBackground=darkBackground, applyWatershed=applyWatershed)
        else:
            self.image = np.asarray(img.copy(), dtype='uint8')

        if not (grayscaleImage is None):
            if len(grayscaleImage.shape) != 2:
                self.gsImage = cv2.cvtColor(src=grayscaleImage.copy(), code=cv2.COLOR_BGR2GRAY).copy()
            else:
                self.gsImage = np.asarray(grayscaleImage.copy())

        self._allContours = None
        self._convexHullUpper = None
        self._convexHullLower = None
        self.areas = None
        self.completenessScores = None
        self.contours = None
        self.contourHierarchy = None
        self.convexHulls = None
        self.convexnessScores = None
        self.maxFeretDiameters = None
        self.maxFeretPoints = None
        self.minFeretDiameters = None
        self.minFeretPoints = None
        self.number = None
        self.perimeters = None
        self.meanIntensities = None
        self.minAreaRects = None

        self.pixelDistance = pixelDistance
        self.knownDistance = knownDistance
        self.unit = unit
        self.excludeEdges = excludeEdges

        # Get the contours because they are needed for all further calculations
        self.__calculateContours()

    @staticmethod
    def __orientation(p, q, r):
        """
        http://code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets/
        convex hull (Graham scan by x-coordinate) and diameter of a set of points
        David Eppstein, UC Irvine, 7 Mar 2002
        :param p: 2D Point
        :param q: 2D Point
        :param r: 2D Point
        :return: Positive if p-q-r are clockwise, neg if counterclockwise, zero if co-linear.
        """
        return (q[1] - p[1]) * (r[0] - p[0]) - (q[0] - p[0]) * (r[1] - p[1])

    @staticmethod
    def __polygon_area(x, y):
        """
        Implementation of the shoelace formula
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        :param x: List of x coordinates of the polygon
        :param y: List of x coordinates of the polygon
        :return: Area of the polygon
        """
        x_ = x - x.mean()
        y_ = y - y.mean()
        correction = x_[-1] * y_[0] - y_[-1] * x_[0]
        main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
        return 0.5 * np.abs(main_area + correction)

    @staticmethod
    def __dist(p, q):
        """
        Calculates the distance between two 2D Points
        :param p: Point 1
        :param q: Point 2
        :return: Euclidian distance between p and q
        """
        return ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5

    def __calculateContours(self):
        """
        Wrapper method for cv2.findContours
        Is executed in the constructor during initialization, and sets the class variables
        contours and contourHierarchy on which all further measurements depend.
        """
        self._allContours, self.contourHierarchy = cv2.findContours(image=self.image, mode=cv2.RETR_TREE,
                                                                    method=cv2.CHAIN_APPROX_SIMPLE)

        self.contours = self._allContours.copy()
        for i in range(len(self.contours)-1, -1, -1):
            c = self.contours[i]
            # If a contour touches the edge of the image, remove it from the list
            if np.any(c.transpose()[0, 0] >= (self.image.shape[1]-1))\
                    or np.any(c.transpose()[1, 0] >= (self.image.shape[0]-1)) or np.any(c == 0):
                if self.excludeEdges:
                    del self.contours[i]
                    np.delete(self.contourHierarchy, i)
            # If a contour consists of less than 5 points and is very short, remove it from the list
            elif len(c) < 5:
                # calculate perimeter
                perim = 0
                for j in range(0, len(c)):
                    p0 = c[j][0]
                    p1 = c[(j + 1) % len(c)][0]
                    perim += ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
                # If less than 2 pixels, omit
                if perim < 8:
                    del self.contours[i]
                    np.delete(self.contourHierarchy, i)
        self.number = len(self.contours)

    def __removeShapeMeasurements(self, shapeIndex):
        """
        Remove all values associated with the shape of the specified index
        :param: shapeIndex: Index of the shape for which the values should be deleted
        """
        if not self.areas is None:
            del self.areas[shapeIndex]
        if not self.completenessScores is None:
            del self.completenessScores[shapeIndex]
        if not self.convexnessScores is None:
            del self.convexnessScores[shapeIndex]
        if not self.contours is None:
            del self.contours[shapeIndex]
        if not self._convexHullLower is None:
            del self._convexHullLower[shapeIndex]
        if not self._convexHullUpper is None:
            del self._convexHullUpper[shapeIndex]
        if not self.convexHulls is None:
            del self.convexHulls[shapeIndex]
        if not self.maxFeretDiameters is None:
            del self.maxFeretDiameters[shapeIndex]
        if not self.maxFeretPoints is None:
            del self.maxFeretPoints[shapeIndex]
        if not self.minFeretDiameters is None:
            del self.minFeretDiameters[shapeIndex]
        if not self.minFeretPoints is None:
            del self.minFeretPoints[shapeIndex]
        if not self.perimeters is None:
            del self.perimeters[shapeIndex]
        if not self.meanIntensities is None:
            del self.meanIntensities[shapeIndex]
        if not self.minAreaRects is None:
            del self.minAreaRects[shapeIndex]
        np.delete(self.contourHierarchy, shapeIndex)

    def __rotatingCalipers(self, shapeIndex):
        """
        http://code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets/
        Given a list of 2d points, finds all ways of sandwiching the points
        between two parallel lines that touch one point each, and yields the sequence
        of pairs of points touched by each pair of lines.
        :param shapeIndex: Index of the shape in the contours array for which to perform the calculation
        """
        if self.convexHulls is None:
            self.calculateConvexHulls()
        U = self._convexHullUpper[shapeIndex].copy()
        L = self._convexHullLower[shapeIndex].copy()

        i = 0
        j = len(L) - 1

        while i < len(U) - 1 or j > 0:
            yield U[i], L[j]

            # if all the way through one side of hull, advance the other side
            if i == len(U) - 1:
                j -= 1
            elif j == 0:
                i += 1

            # still points left on both lists, compare slopes of next hull edges
            # being careful to avoid divide-by-zero in slope calculation
            elif (U[i + 1][1] - U[i][1]) * (L[j][0] - L[j - 1][0]) > \
                    (L[j][1] - L[j - 1][1]) * (U[i + 1][0] - U[i][0]):
                i += 1
            else:
                j -= 1

    def segment(self, image, threshold=-1.0, applyWatershed=True, darkBackground=False):
        """
        Segments a grayscale image based on threshold and watershed.
        :param threshold: Threshold value (if < 0, Otsu threshold is applied)
        :param applyWatershed: Whether or not to apply Watershed after thresholding
        :param darkBackground: If True, values > threshold are set to 255, if False, values < Threshold are set to 255.
        :return: Segmented image as numpy array of dtype=uint8, values are 0 and 255.
        """
        MIN_DISTANCE = 5
        img = image.copy()

        if threshold < 0:
            threshold = threshold_otsu(img)
        if darkBackground:
            mask = img > threshold
        else:
            mask = img < threshold

        if not applyWatershed:
            return np.asarray(mask * 255, dtype='uint8')

        # Get the Euclidian Distance Map
        distance = ndimage.distance_transform_edt(mask)

        # Get the local maxima of the distance map
        local_maxi = peak_local_max(distance, indices=False, min_distance=MIN_DISTANCE)

        # Merge maxima that are close by
        local_maxi = ndimage.binary_dilation(local_maxi, iterations=2)
        # Contract the maxima back into a single point
        # local_maxi = ndimage.binary_erosion(local_maxi, iterations=2)

        markers = ndimage.label(local_maxi)[0]
        labels = watershed(-distance, markers, connectivity=np.ones((3, 3)), mask=mask, watershed_line=applyWatershed)
        # labels = watershed(-distance, watershed_line=True)
        # labels = np.logical_and(labels, mask)

        return np.asarray((labels > 0) * 255, dtype='uint8')

    def calculateAreas(self):
        """
        Calculates the areas of all shapes in the class variable contours (taking calibration into account),
         adds them to a list stored in the class variable areas, and also returns the list.
        :return: List of areas of the particles (taking calibration into account)
        """
        self.areas = []
        for i in range(0, self.number):
            x = np.asarray([p[0][0] for p in self.contours[i]], dtype='uint16')
            y = np.asarray([p[0][1] for p in self.contours[i]], dtype='uint16')
            areaShape = Measure.__polygon_area(x, y)
            self.areas.append(areaShape * (self.knownDistance ** 2) / (self.pixelDistance ** 2))
        return self.areas

    def calculateMeanIntensities(self):
        """
        Calculates the mean intensity of all shapes in the class variable contours (taking calibration into account),
        adds them to a list stored in the class variable mean intensities, and also returns the list.
        :return: List of mean intensities of the particles (taking calibration into account)
        """
        self.meanIntensities = []
        for i in range(0, self.number):
            integral = 0.0
            a = 0
            x_ = np.asarray([p[0][0] for p in self.contours[i]], dtype='uint16')
            y_ = np.asarray([p[0][1] for p in self.contours[i]], dtype='uint16')
            for x in range(np.min(x_), np.max(x_) + 1):
                for y in range(np.min(y_), np.max(y_) + 1):
                    if cv2.pointPolygonTest(self.contours[i], (x, y), False) >= 0:
                        integral += self.gsImage[y, x]
                        a += 1
            if integral > 0:
                self.meanIntensities.append(integral / (a * (self.knownDistance ** 2) / (self.pixelDistance ** 2)))
            else:
                self.meanIntensities.append(0.0)
        return self.meanIntensities

    def calculatePerimeters(self):
        """
        Calculates the perimeters of all shapes in the class variable contours (taking calibration into account),
        adds them to a list stored in the class variable perimeters, and also returns the list.
        :return: List of perimeters of the particles (taking calibration into account)
        """
        self.perimeters = []
        for i in range(0, len(self.contours)):
            perimeterShape = 0
            for j in range(0, len(self.contours[i])):
                p = self.contours[i][j][0]
                q = self.contours[i][(j + 1) % len(self.contours[i])][0]
                perimeterShape += self.__dist(p, q)
            self.perimeters.append(perimeterShape * self.knownDistance / self.pixelDistance)
        return self.perimeters

    def calculateConvexHulls(self):
        """
        http://code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets/
        Graham scan to find upper and lower convex hulls of a set of 2D points.
        Calculates the convex hulls and stores them in the private class variables _convexHullUpper and _convexHullLower
        (used by the rotating caliper algorithm), as well as in a public field convexHulls (formatted in a way so that
        it can easily be drawn with cv2.drawContours()). It also returns the list convex hull points.
        :return: List of convex hull points
        """
        self._convexHullUpper = []
        self._convexHullLower = []
        self.convexHulls = []
        for shape in self.contours:
            U = []
            L = []
            Points = sorted(shape.copy(), key=lambda point: point[0][0])
            for p in Points:
                while len(U) > 1 and Measure.__orientation(U[-2], U[-1], p[0]) <= 0:
                    U.pop()
                while len(L) > 1 and Measure.__orientation(L[-2], L[-1], p[0]) >= 0:
                    L.pop()
                U.append(p[0])
                L.append(p[0])

            self._convexHullUpper.append(U.copy())
            self._convexHullLower.append(L.copy())
            L.reverse()
            hull = U + L[1:-1]
            self.convexHulls.append([np.asarray([[j] for j in hull])])
        return self.convexHulls

    def calculateMinFeretDiameters(self):
        """
        Adapted from:
        http://code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets/
        Calculate the minimum Feret Diameters (taking calibration into account) and the corresponding points and store
        them as a list in the class variables minFeretDiameters and minFeretPoints. The points are stored in a way so
        that they can easily be drawn with the cv2.drawContours() method. It also returns both lists.
        :return: List of minimum Feret Diameters and the coordinates of the corresponding points
        """
        # See also https://blogs.mathworks.com/steve/2018/02/20/minimum-feret-diameter/
        self.minFeretDiameters = []
        self.minFeretPoints = []
        for i in range(0, self.number):
            caliperPoints = [(self.__dist(p, q), (p, q)) for p, q in self.__rotatingCalipers(i)]

            # caliperPoints contains all ways of sandwiching two points on the convex hull between two parallel lines
            # that touch one point each. For minFeret, one of the parallel lines always goes through two vertices.
            # For determining minFeret, loop through all caliper points (antipodal pairs), construct the corresponding
            # triangles, and measure their heights

            mF = []
            for j in range(0, len(caliperPoints)):
                for k in range(j + 1, len(caliperPoints)):
                    pair1 = np.asarray(caliperPoints[j][1])
                    pair2 = np.asarray(caliperPoints[k][1])
                    # True and False ar just a lazy way to refer to indices 0 and 1 here.
                    for m in [False, True]:
                        for n in [False, True]:
                            # Check that 2 points from different pairs are identical
                            if np.all(pair1[int(m)] == pair2[int(n)]):
                                # If so, make sure that the other 2 points are adjacent points of the convex hull
                                for l in range(0, self.convexHulls[i][0].shape[0]):
                                    if (np.all(pair1[int(not m)] == self.convexHulls[i][0][l % self.convexHulls[i][0].shape[0]][0]) and np.all(pair2[int(not n)] == self.convexHulls[i][0][(l + 1) % self.convexHulls[i][0].shape[0]][0])) or (np.all(pair2[int(not n)] == self.convexHulls[i][0][l % self.convexHulls[i][0].shape[0]][0]) and np.all(pair1[int(not m)] == self.convexHulls[i][0][(l + 1) % self.convexHulls[i][0].shape[0]][0])):
                                        a = caliperPoints[j][0]
                                        b = caliperPoints[k][0]
                                        c = self.__dist(pair1[int(not m)], pair2[int(not n)])
                                        s = (a + b + c) / 2
                                        h = 2 * (s * (s - a) * (s - b) * (s - c)) ** 0.5 / c
                                        if a ** 2 - h ** 2 < 0:
                                            o = 0
                                        else:
                                            o = (a ** 2 - h ** 2) ** 0.5

                                        p0 = np.asarray(pair1[int(not m)] + o * (pair2[int(not n)] - pair1[int(not m)]) /
                                                        self.__dist(pair2[int(not n)], pair1[int(not m)]), dtype='uint16')
                                        p1 = pair1[int(m)]
                                        p = np.asarray([p0, p1])
                                        mF.append([h, p.tolist()])
                                        break

            x = min(mF)
            self.minFeretDiameters.append(x[0] * self.knownDistance / self.pixelDistance)
            self.minFeretPoints.append(np.asarray(x[1]))

        return self.minFeretDiameters, self.minFeretPoints

    def calculateMaxFeretDiameters(self):
        """
        Calculate the maximum Feret Diameters (taking calibration into account) and the corresponding points and store
        them as a list in the class variables maxFeretDiameters and maxFeretPoints. The points are stored in a way so
        that they can easily be drawn with the cv2.drawContours() method. It also returns both lists.
        :return: List of maximum Feret Diameters and the coordinates of the corresponding points
        """
        self.maxFeretDiameters = []
        self.maxFeretPoints = []
        for i in range(0, self.number):
            # Workaround: Convert tuple with numpy arrays to list to avoid ambiguity in sorting algorithm
            caliperPoints = [(self.__dist(p, q), (p, q)) for p, q in self.__rotatingCalipers(i)]
            mF, pMF = max([(a, [b[0].tolist(), b[1].tolist()]) for a, b in caliperPoints])
            self.maxFeretDiameters.append(mF * self.knownDistance / self.pixelDistance)
            self.maxFeretPoints.append(np.asarray(pMF))

        return self.maxFeretDiameters, self.maxFeretPoints

    def calculateConvexnessScores(self, dim=2):
        """
        Calculates the convexness score of the shapes in the contours array, stores them in the class variable
        convexnessScore, and also returns the list. If the dimensionality is set to 1, this is the ratio of the
        convex hull of the shape to its perimeter, if it is set to 2 it is the ratio of the area of the shape to
        the area of its convex hull. Either way, a value close to 1 means that the shape is more convex.
        :param dim: Dimensionality used for calculating the convexness (has to be 1 or 2)
        :return: List of convexness scores of the shapes
        """
        assert dim in [1, 2]
        if self.convexHulls is None:
            self.calculateConvexHulls()

        self.convexnessScores = []

        for i in range(0, self.number):
            perimHull = 0

            if dim == 1:
                for j in range(0, len(self.convexHulls[i][0])):
                    p0 = self.convexHulls[i][0][j][0]
                    p1 = self.convexHulls[i][0][(j + 1) % len(self.convexHulls[i][0])][0]
                    perimHull += ((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)**0.5
                if self.perimeters is None:
                    self.calculatePerimeters()
                self.convexnessScores.append(perimHull*self.knownDistance/self.pixelDistance/self.perimeters[i])
            else:
                x = np.asarray([p[0][0] for p in self.convexHulls[i][0]], dtype='uint16')
                y = np.asarray([p[0][1] for p in self.convexHulls[i][0]], dtype='uint16')
                areaHull = Measure.__polygon_area(x, y)
                if self.areas is None:
                    self.calculateAreas()
                if areaHull is None or areaHull == 0 or areaHull == np.nan:
                    # Usually occurs when there are only two points in the convex hull and hence areaHull==0
                    self.convexnessScores.append(1.0)
                else:
                    self.convexnessScores.append(self.areas[i]/(areaHull*(self.knownDistance/self.pixelDistance)**2))

        return self.convexnessScores

    def calculateCompletenessScores(self):
        """
        Calculates the completeness score of the shapes in the contours array, stores them in the class variable
        completenessScores, and also returns the list. The completeness score is defined as the ratio of the area
        of the particle to the area of an ellipse fitted to the particle.
        :return: List of completeness scores
        """
        self.completenessScores = []
        if self.areas is None:
            self.calculateAreas()

        for i in range(0, len(self.contours)):
            if len(self.contours[i]) < 5:
                # Must have at least 5 points to fit an ellipse.
                # If there are less, take the number of pixels as the area of the ellipse
                self.completenessScores.append(self.areas[i]/len(self.contours[i]))
            else:
                # Fit an Ellipse to the contour. Omit (x, y) coordinates because we don't need them here
                (__, (MA, ma), angle) = cv2.fitEllipse(self.contours[i])
                if MA is None or ma is None or MA == np.nan or ma == np.nan or MA == 0 or ma == 0:
                    self.completenessScores.append(2)
                else:
                    self.completenessScores.append(self.areas[i] /
                                                   (np.pi*MA/2.0*ma/2.0*(self.knownDistance/self.pixelDistance)**2))
        return self.completenessScores

    def calculateMinAreaRects(self):
        """
        Calculates the (non-axis-aligned) bounding box of all shapes in the class variable contours (taking calibration
        into account for sizes but NOT for pixel coordinates), adds them to a list stored in the class variable
        minAreaRects, and also returns the list.
        :return: List of non-axis-aligned bounding boxes of the particles (center of mass, shape, angle)
        """
        self.minAreaRects = [cv2.minAreaRect(c) for c in self.contours]
        for i in range(0, len(self.minAreaRects)):  # Use calibration for rectangle width and height
            self.minAreaRects[i] = (self.minAreaRects[i][0], (self.minAreaRects[i][1][0] * self.knownDistance / self.pixelDistance, self.minAreaRects[i][1][1] * self.knownDistance / self.pixelDistance), self.minAreaRects[i][2])
        return self.minAreaRects

    def filterResults(self, filterType, minValue=0.0, maxValue=-1.0):
        """
        Filter the results based on area, completeness score, convexness score, mean intensity,
        maxFeretDianmeter, minFeretDiameter, perimeter
        :param filterType: Quantity based on which the results are filtered
        :param minValue: Minimum value of that quantity that should still be included
        :param maxValue: Maximum value of that quantity that should still be included
        """
        assert filterType in {'area', 'completenessScore', 'convexnessScore', 'meanIntensity', 'maxFeretDiameter',
                              'minFeretDiameter', 'perimeter'}

        if minValue == 0 and maxValue < minValue:
            return

        if filterType == 'area':
            if self.areas is None:
                self.calculateAreas()
            for i in range(self.number-1, -1, -1):
                if self.areas[i] < minValue or (self.areas[i] > maxValue and maxValue >= minValue):
                    self.__removeShapeMeasurements(i)

        elif filterType == 'completenessScore':
            if self.completenessScores is None:
                self.calculateCompletenessScores()
            for i in range(self.number-1, -1, -1):
                if self.completenessScores[i] < minValue or (self.completenessScores[i] > maxValue and maxValue >= minValue):
                    self.__removeShapeMeasurements(i)

        elif filterType == 'convexnessScore':
            if self.convexnessScores is None:
                self.calculateConvexnessScores()
            for i in range(self.number-1, -1, -1):
                if self.convexnessScores[i] < minValue or (self.convexnessScores[i] > maxValue and maxValue >= minValue):
                    self.__removeShapeMeasurements(i)

        elif filterType == 'meanIntensity':
            if self.meanIntensities is None:
                self.calculateMeanIntensities()
            for i in range(self.number-1, -1, -1):
                if self.meanIntensities[i] < minValue or (self.meanIntensities[i] > maxValue and maxValue >= minValue):
                    self.__removeShapeMeasurements(i)

        elif filterType == 'maxFeretDiameter':
            if self.maxFeretDiameters is None:
                self.calculateMaxFeretDiameters()
            for i in range(self.number-1, -1, -1):
                if self.maxFeretDiameters[i] < minValue or (self.maxFeretDiameters[i] > maxValue and maxValue >= minValue):
                    self.__removeShapeMeasurements(i)

        elif filterType == 'minFeretDiameter':
            if self.minFeretDiameters is None:
                self.calculateMinFeretDiameters()
            for i in range(self.number-1, -1, -1):
                if self.minFeretDiameters[i] < minValue or (self.minFeretDiameters[i] > maxValue and maxValue >= minValue):
                    self.__removeShapeMeasurements(i)

        elif filterType == 'perimeter':
            if self.perimeters is None:
                self.calculatePerimeters()
            for i in range(self.number-1, -1, -1):
                if self.perimeters[i] < minValue or (self.perimeters[i] > maxValue and maxValue >= minValue):
                    self.__removeShapeMeasurements(i)

        elif filterType == 'minAreaRects':
            if self.minAreaRects is None:
                self.calculateMinAreaRects()
            for i in range(self.number-1, -1, -1):
                if max(self.minAreaRects[i][1][0], self.minAreaRects[i][1][1]) < minValue or (min(self.minAreaRects[i][1][0], self.minAreaRects[i][1][1]) > maxValue and maxValue >= minValue):
                    self.__removeShapeMeasurements(i)

        self.number = len(self.contours)
