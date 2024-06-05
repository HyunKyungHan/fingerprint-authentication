from minutiae import match_minutiae

def calculate_similarity_score(matched_pairs, total_minutiae):
    return len(matched_pairs) / total_minutiae

def calculate_similarity(minutiae_list1, minutiae_list2):
    matched_pairs = match_minutiae(minutiae_list1, minutiae_list2)
    similarity_score = calculate_similarity_score(matched_pairs, max(len(minutiae_list1), len(minutiae_list2)))

    # print(f'Matched Pairs: {matched_pairs}')
    print(f'Similarity Score: {similarity_score:.2f}')

    return similarity_score