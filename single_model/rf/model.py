import sys
sys.path.append("../")
from util import *
from sklearn.ensemble import RandomForestClassifier

def rf_opt1(df_cell_train_feats, y_train, df_cell_test_feats):
    logging.info("train rf_opt1 model")
    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, max_features="log2", min_samples_split=3, min_samples_leaf=2)
    clf.fit(df_cell_train_feats, y_train)
    y_test_pred = clf.predict_proba(df_cell_test_feats)
    return y_test_pred

model_dict = {"rf_opt1": rf_opt1}

if __name__ == "__main__":
    config_name = sys.argv[1]
    data_path = "../../data/"
    run_model(config_name, model_dict, data_path)