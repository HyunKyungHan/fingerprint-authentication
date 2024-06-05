############### FEATURE EXTRACTION #################
import cv2
import numpy as np
import skimage.morphology
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
import matplotlib.pyplot as plt
import math

class MinutiaeFeature(object):
  def __init__(self, locX, locY, Orientation, Type):
    self.locX = locX
    self.locY = locY
    self.Orientation = Orientation
    self.Type = Type

class FingerprintFeatureExtractor(object):
  def __init__(self):
    self._mask = []
    self._skel = []
    self.minutiaeTerm = []
    self.minutiaeBif = []

  def __skeletonize(self, img):
    img = np.uint8(img > 128)
    # self._skel = skimage.morphology.skeletonize(img)
    self._skel = np.uint8(img) * 255
    self._mask = img * 255

  def __computeAngle(self, block, minutiaeType):
    angle = []
    (blkRows, blkCols) = np.shape(block)
    CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
    if (minutiaeType.lower() == 'termination'):
        sumVal = 0
        for i in range(blkRows):
            for j in range(blkCols):
                if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                    angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                    sumVal += 1
                    if (sumVal > 1):
                        angle.append(float('nan'))
        return (angle)

    elif (minutiaeType.lower() == 'bifurcation'):
        (blkRows, blkCols) = np.shape(block)
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        angle = []
        sumVal = 0
        for i in range(blkRows):
            for j in range(blkCols):
                if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                    angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                    sumVal += 1
        if (sumVal != 3):
            angle.append(float('nan'))
        return (angle)

  def __getTerminationBifurcation(self, img):
    self._skel = self._skel == 255
    (rows, cols) = self._skel.shape
    self.minutiaeTerm = np.zeros(self._skel.shape)
    self.minutiaeBif = np.zeros(self._skel.shape)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (self._skel[i][j] == 1):
                block = self._skel[i - 1:i + 2, j - 1:j + 2]
                block_val = np.sum(block)
                if (block_val == 2):
                    self.minutiaeTerm[i, j] = 1
                elif (block_val == 4):
                    self.minutiaeBif[i, j] = 1

    self._mask = convex_hull_image(self._mask > 0)
    self._mask = erosion(self._mask, square(5))  # Structuing element for mask erosion = square(5)
    self.minutiaeTerm = np.uint8(self._mask) * self.minutiaeTerm

  def __removeSpuriousMinutiae(self, minutiaeList, img, thresh):
    img = img * 0
    SpuriousMin = []
    real_dist = []
    numPoints = len(minutiaeList)
    D = np.zeros((numPoints, numPoints))
    for i in range(1,numPoints):
        for j in range(0, i):
            (X1,Y1) = minutiaeList[i]['centroid']
            (X2,Y2) = minutiaeList[j]['centroid']

            dist = np.sqrt((X2-X1)**2 + (Y2-Y1)**2)
            D[i][j] = dist
            if(dist < thresh): # threshold이하면 제외
                SpuriousMin.append(i)
                SpuriousMin.append(j)
            else:
              real_dist.append(dist)

    SpuriousMin = np.unique(SpuriousMin) 
    for i in range(0,numPoints):
        if(not i in SpuriousMin):
            (X,Y) = np.int16(minutiaeList[i]['centroid']) # 제외하지 않을 애들을 X, Y 값으로 불러오기
            img[X,Y] = 1
    img = np.uint8(img)
    return(img)

  def __cleanMinutiae(self, img):
    self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
    RP = skimage.measure.regionprops(self.minutiaeTerm)
    self.minutiaeTerm = self.__removeSpuriousMinutiae(RP, np.uint8(img), 10)

  def __performFeatureExtraction(self):
    FeaturesTerm = []
    self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
    RP = skimage.measure.regionprops(np.uint8(self.minutiaeTerm))

    WindowSize = 2  # --> For Termination, the block size must can be 3x3, or 5x5. Hence the window selected is 1 or 2
    FeaturesTerm = []
    for num, i in enumerate(RP):
        # print(num)
        (row, col) = np.int16(np.round(i['Centroid']))
        block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
        angle = self.__computeAngle(block, 'Termination')
        if(len(angle) == 1):
            FeaturesTerm.append(MinutiaeFeature(row, col, angle, 'Termination'))
    term_cnt = num
    FeaturesBif = []
    min_cnt = 0
    self.minutiaeBif = skimage.measure.label(self.minutiaeBif, connectivity=2)
    RP = skimage.measure.regionprops(np.uint8(self.minutiaeBif))
    WindowSize = 1  # --> For Bifurcation, the block size must be 3x3. Hence the window selected is 1
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
        angle = self.__computeAngle(block, 'Bifurcation')
        # print('bif:', (row, col))
        if(len(angle) == 3):
            FeaturesBif.append(MinutiaeFeature(row, col, angle, 'Bifurcation'))
        min_cnt += 1
    return (FeaturesTerm, FeaturesBif, term_cnt, min_cnt)

  def extractMinutiaeFeatures(self, img):
    self.__skeletonize(img)

    self.__getTerminationBifurcation(img)

    self.__cleanMinutiae(img)

    FeaturesTerm, FeaturesBif, term_cnt, min_cnt = self.__performFeatureExtraction()
    return(FeaturesTerm, FeaturesBif, term_cnt, min_cnt)

  def showResults(self):
    BifLabel = skimage.measure.label(self.minutiaeBif, connectivity=2)
    TermLabel = skimage.measure.label(self.minutiaeTerm, connectivity=2)

    minutiaeBif = TermLabel * 0
    minutiaeTerm = BifLabel * 0

    (rows, cols) = self._skel.shape
    DispImg = np.zeros((rows, cols, 3), np.uint8)
    DispImg[:, :, 0] = 255*self._skel
    DispImg[:, :, 1] = 255*self._skel
    DispImg[:, :, 2] = 255*self._skel

    RP = skimage.measure.regionprops(BifLabel)
    for idx, i in enumerate(RP):
        (row, col) = np.int16(np.round(i['Centroid']))
        minutiaeBif[row, col] = 1
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
        skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))

    RP = skimage.measure.regionprops(TermLabel)
    for idx, i in enumerate(RP):
        (row, col) = np.int16(np.round(i['Centroid']))
        minutiaeTerm[row, col] = 1
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
        skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))
    
    plt.imsave(f'results/minutiae_matched/{idx}_matched.png', DispImg, cmap='gray')

def extract_minutiae_features(img, showResult=False):
  feature_extractor = FingerprintFeatureExtractor()
  FeaturesTerm, FeaturesBif, term_cnt, min_cnt = feature_extractor.extractMinutiaeFeatures(img)

  if(True):
      feature_extractor.showResults()

  return(FeaturesTerm, FeaturesBif, term_cnt, min_cnt)