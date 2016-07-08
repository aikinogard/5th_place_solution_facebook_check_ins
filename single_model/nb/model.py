import sys
sys.path.append("../")
from util import *
from sklearn.naive_bayes import GaussianNB

def nb_xyat_weight1(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        df_new = pd.DataFrame()
        df_new["x"] = df["x"]
        df_new["y"] = df["y"]
        df_new["hour"] = df["hour"]
        df_new["weekday"] = df["weekday"]
        df_new["accuracy"] = df["accuracy"].apply(np.log10)
        return df_new
    logging.info("train nb_xyat_weight1 model")
    clf = GaussianNB()
    clf.fit(prepare_feats(df_cell_train_feats), y_train, df_cell_train_feats["time"] ** 2)
    y_test_pred = clf.predict_proba(prepare_feats(df_cell_test_feats))
    return y_test_pred

model_dict = {"nb_xyat_weight1": nb_xyat_weight1}

if __name__ == "__main__":
    config_name = sys.argv[1]
    data_path = "../../data/"
    run_model(config_name, model_dict, data_path)