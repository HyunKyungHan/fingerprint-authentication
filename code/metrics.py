# 정답 list와 실제 prediction 결과 list를 가지고 precision, recall, accuracy 계산
from sklearn.metrics import accuracy_score, precision_score, recall_score

def metrics(matches, answer):
    # Calculate accuracy
    accuracy = accuracy_score(matches, answer)
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = precision_score(matches, answer, average="macro")
    print("Precision:", precision)

    # Calculate recall (sensitivity)
    recall = recall_score(matches, answer, average="macro")
    print("Recall (Sensitivity):", recall)

    return accuracy, precision, recall