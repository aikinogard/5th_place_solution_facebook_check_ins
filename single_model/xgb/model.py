import sys
sys.path.append("../")
from util import *
import xgboost as xgb

def xgb0(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        return df.drop(['time'], axis=1)
    logging.info("train xgb0 model")
    clf = xgb.XGBClassifier()
    clf.fit(prepare_feats(df_cell_train_feats), y_train)
    y_test_pred = clf.predict_proba(prepare_feats(df_cell_test_feats))
    return y_test_pred

def xgb150opt(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        return df.drop(['time'], axis=1)
    logging.info("train xgb150opt model")
    clf = xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, min_child_weight=3, subsample=0.667, colsample_bytree=1)
    clf.fit(prepare_feats(df_cell_train_feats), y_train)
    y_test_pred = clf.predict_proba(prepare_feats(df_cell_test_feats))
    return y_test_pred

def xgb150opt2(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        return df.drop(['time'], axis=1)
    logging.info("train xgb150opt2 model")
    clf = xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, min_child_weight=1, subsample=0.85263, colsample_bytree=0.657894, reg_alpha=1.55556, reg_lambda=1.22222, gamma=0.3333333)
    clf.fit(prepare_feats(df_cell_train_feats), y_train)
    y_test_pred = clf.predict_proba(prepare_feats(df_cell_test_feats))
    return y_test_pred

model_dict = {"xgb0": xgb0,
              "xgb150opt": xgb150opt,
              "xgb150opt2": xgb150opt2}

if __name__ == "__main__":
    config_name = sys.argv[1]
    data_path = "../../data/"
    run_model(config_name, model_dict, data_path)

