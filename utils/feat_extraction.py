import math

class Minutiae:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    def __repr__(self):
        return f'Minutiae(x={self.x}, y={self.y}, angle={self.angle})'

def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def angular_distance(angle1, angle2):
    return min(abs(angle1 - angle2), 360 - abs(angle1 - angle2))

def minutiae_distance(minutiae1, minutiae2, spatial_threshold=15, angular_threshold=10):
    spatial_dist = euclidean_distance(minutiae1, minutiae2)
    angular_dist = angular_distance(minutiae1.angle, minutiae2.angle)
    return spatial_dist < spatial_threshold and angular_dist < angular_threshold

def match_minutiae(minutiae_list1, minutiae_list2, spatial_threshold=15, angular_threshold=10):
    matched_pairs = []
    for m1 in minutiae_list1:
        for m2 in minutiae_list2:
            if minutiae_distance(m1, m2, spatial_threshold, angular_threshold):
                matched_pairs.append((m1, m2))
                break  # Assuming each minutiae in list1 matches with at most one in list2
    return matched_pairs

def calculate_similarity_score(matched_pairs, total_minutiae):
    return len(matched_pairs) / total_minutiae


def feat_extraction(img):
    '''
    input: skeletonized image
    output: [(x1, y1, angle1), (x2, y2, angle2), ...] of Termination and Bifurcation points
    '''
    FeaturesTerminations, FeaturesBifurcations, term_cnt, min_cnt = extract_minutiae_features(img=img, showResult=True)
    # Add to class Minutiae
    Minutiae_input = []
    for idx, curr_minutiae in enumerate(FeaturesBifurcations):
        row, col = curr_minutiae.locX, curr_minutiae.locY, 
        orientation, type = curr_minutiae.Orientation, curr_minutiae.Type
        print(row, col, orientation[0])
        Minutiae_input.append(Minutiae(row, col, orientation[0]))
    for idx, curr_minutiae in enumerate(FeaturesTerminations):
        row, col = curr_minutiae.locX, curr_minutiae.locY, 
        orientation, type = curr_minutiae.Orientation, curr_minutiae.Type
        print(row, col, float(orientation[0]))
        Minutiae_input.append(Minutiae(row, col, orientation[0]))

    return Minutiae_input