import sys
sys.path.append("../")
from util import *
from sklearn.neighbors import KNeighborsClassifier

def knn_ps2(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        df_new = pd.DataFrame()
        df_new["year"] = (1 + df["year"]) * 10.
        df_new["hour"] = (1 + df["hour"]) * 4.
        df_new["weekday"] = (1 + df["weekday"]) * 3.11
        df_new["month"] = (1 + df["month"]) * 2.11
        df_new["accuracy"] = df["accuracy"].apply(lambda x: np.log10(x)) * 10.
        df_new["x"] = df["x"] * 465.
        df_new["y"] = df["y"] * 975.
        return df_new
    logging.info("train knn_ps2 model")
    df_cell_train_feats_knn = prepare_feats(df_cell_train_feats)
    clf = KNeighborsClassifier(n_neighbors=np.floor(np.sqrt(len(y_train))/5.3).astype(int),
                               weights=lambda x: x ** -2, metric='manhattan', n_jobs=-1)
    clf.fit(df_cell_train_feats_knn, y_train)
    df_cell_test_feats_knn = prepare_feats(df_cell_test_feats)
    y_test_pred = clf.predict_proba(df_cell_test_feats_knn)
    return y_test_pred

model_dict = {"knn_ps2": knn_ps2}

if __name__ == "__main__":
    config_name = sys.argv[1]
    data_path = "../../data/"
    run_model(config_name, model_dict, data_path)