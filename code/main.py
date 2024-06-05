import os
import glob
from PIL import Image
# from utils import *

from feat_extraction import *
from minutiae import *
from metrics import metrics
from preprocess import read_and_preprocess
from calculate_similarity import calculate_similarity

if __name__ == '__main__':
    results = []
    matches = []
    answer = []
    train_path = 'data\\database'
    test2_path = 'data\\query'

    list_train = [os.path.join(train_path, x) for x in os.listdir(train_path)]
    list_test2 = [os.path.join(test2_path, x) for x in os.listdir(test2_path)]

    for i in range(len(list_test2)):
        best_sim = 0
        matched_res = None
        print("Fingerprint matching on:", list_test2[i])
        test_path = os.listdir(test2_path)[i]
        test_num = test_path.split('_')[0]
        answer.append(test_num)

        img1 = Image.open(list_test2[i]).convert('L')
        img1 = np.array(img1)
        query = read_and_preprocess(img1, test_path, True)
        FeaturesTerminations, FeaturesBifurcations, term_cnt, min_cnt = extract_minutiae_features(img=query, showResult=True)
        query_minutiae = feat_extraction(FeaturesTerminations, FeaturesBifurcations)

        for j in range(len(list_train)):
            train_num = os.listdir(train_path)[j]
            train_num = train_num.split('.')[0]

            img2 = Image.open(list_train[j]).convert('L')
            img2 = np.array(img2)
            db = read_and_preprocess(img2, None, False)

            FeaturesTerminations, FeaturesBifurcations, term_cnt, min_cnt = extract_minutiae_features(img=db, showResult=True)
            db_minutiae = feat_extraction(FeaturesTerminations, FeaturesBifurcations)

            sim = calculate_similarity(query_minutiae, db_minutiae)
            if sim > best_sim:
                best_sim = sim
                matched_res = train_num

        print("best similarity:", best_sim)

        if matched_res == test_num:
            matches.append(matched_res)

    metrics(matches=matches, answer=answer)