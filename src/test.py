# import nmi from sklearn
from sklearn.metrics import normalized_mutual_info_score


def test_nmi():
    print(normalized_mutual_info_score([1, 2, 3], [2, 3, 1]))

for num_groups in range(3, 4):  # Loops from 3 to 7
    for t in range(5,10):  # Loops from t0 to t9
        print("num_groups:", num_groups, "t:", t)