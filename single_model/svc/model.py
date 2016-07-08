import sys
sys.path.append("../")
from util import *
from sklearn.svm import SVC
from sklearn import preprocessing

def svc_rbf_xyat(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        df_new = pd.DataFrame()
        df_new["x"] = df["x"]
        df_new["y"] = df["y"]
        df_new["hour"] = df["hour"]
        df_new["weekday"] = df["weekday"]
        df_new["accuracy"] = df["accuracy"].apply(np.log10)
        return preprocessing.scale(df_new.values)

    logging.info("train svc_rbf_xyat model")
    clf = SVC(kernel='rbf', probability=True, cache_size=3000)
    clf.fit(prepare_feats(df_cell_train_feats), y_train)
    y_test_pred = clf.predict_proba(prepare_feats(df_cell_test_feats))
    return y_test_pred

def svc_lin_xyat(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        df_new = pd.DataFrame()
        df_new["x"] = df["x"]
        df_new["y"] = df["y"]
        df_new["hour"] = df["hour"]
        df_new["weekday"] = df["weekday"]
        df_new["accuracy"] = df["accuracy"].apply(np.log10)
        return preprocessing.scale(df_new.values)

    logging.info("train svc_lin_xyat model")
    clf = SVC(kernel='linear', probability=True, cache_size=3000)
    clf.fit(prepare_feats(df_cell_train_feats), y_train)
    y_test_pred = clf.predict_proba(prepare_feats(df_cell_test_feats))
    return y_test_pred

def svc_rbf_xyatu(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        df_new = pd.DataFrame()
        df_new["x"] = df["x"]
        df_new["y"] = df["y"]
        df_new["hour"] = df["hour"]
        df_new["weekday"] = df["weekday"]
        df_new["accuracy"] = df["accuracy"]
        return preprocessing.scale(df_new.values)

    logging.info("train svc_rbf_xyatu model")
    clf = SVC(kernel='rbf', probability=True, cache_size=3000)
    clf.fit(prepare_feats(df_cell_train_feats), y_train)
    y_test_pred = clf.predict_proba(prepare_feats(df_cell_test_feats))
    return y_test_pred

def svc_lin_xyatu(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        df_new = pd.DataFrame()
        df_new["x"] = df["x"]
        df_new["y"] = df["y"]
        df_new["hour"] = df["hour"]
        df_new["weekday"] = df["weekday"]
        df_new["accuracy"] = df["accuracy"]
        return preprocessing.scale(df_new.values)

    logging.info("train svc_lin_xyatu model")
    clf = SVC(kernel='linear', probability=True, cache_size=3000)
    clf.fit(prepare_feats(df_cell_train_feats), y_train)
    y_test_pred = clf.predict_proba(prepare_feats(df_cell_test_feats))
    return y_test_pred

model_dict = {"svc_rbf_xyat": svc_rbf_xyat, "svc_lin_xyat": svc_lin_xyat,
              "svc_rbf_xyatu": svc_rbf_xyatu, "svc_lin_xyatu": svc_lin_xyatu}

if __name__ == "__main__":
    config_name = sys.argv[1]
    data_path = "../../data/"
    run_model(config_name, model_dict, data_path)

