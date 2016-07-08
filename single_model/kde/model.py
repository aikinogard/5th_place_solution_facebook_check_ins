import sys
sys.path.append("../")
from util import *
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from scipy.stats import gaussian_kde
from kde import kde as kdeBGK10

def kde_opt1(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        df_new = pd.DataFrame()
        df_new["hour"] = (1 + df["hour"]) * 3.92105
        df_new["weekday"] = (1 + df["weekday"]) * 4.28947
        df_new["accuracy"] = df["accuracy"].apply(lambda x: np.log10(x)) * 9.44736
        df_new["x"] = df["x"] * 424.489
        df_new["y"] = df["y"] * 959.183
        return df_new
    logging.info("train kde_opt1 model")
    df_cell_train_feats_kde = prepare_feats(df_cell_train_feats)
    df_cell_test_feats_kde = prepare_feats(df_cell_test_feats)
    n_class = len(np.unique(y_train))
    y_test_pred = np.zeros((len(df_cell_test_feats_kde), n_class), "d")
    Xte = df_cell_test_feats_kde.values
    for i in range(n_class):
        X = df_cell_train_feats_kde[y_train == i].values
        cstd = np.std(np.sum(np.abs(X), axis=1))
        gridcv = GridSearchCV(KernelDensity(kernel='gaussian', metric='manhattan'), {'bandwidth': cstd * np.logspace(-1, 1, 10)}, cv=5)
        gridcv.fit(X)
        y_test_pred[:, i] += np.exp(gridcv.best_estimator_.score_samples(Xte))
    return y_test_pred

def kde_opt2c(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        df_new = pd.DataFrame()
        df_new["hour"] = df["hour"]
        df_new["weekday"] = df["weekday"] + df["hour"] / 24.
        df_new["accuracy"] = df["accuracy"].apply(lambda x: np.log10(x))
        df_new["x"] = df["x"]
        df_new["y"] = df["y"]
        return df_new
    logging.info("train kde_opt2c model")
    df_cell_train_feats_kde = prepare_feats(df_cell_train_feats)
    df_cell_test_feats_kde = prepare_feats(df_cell_test_feats)
    n_class = len(np.unique(y_train))
    y_test_pred = np.zeros((len(df_cell_test_feats_kde), n_class), "d")
    for i in range(n_class):
        X = df_cell_train_feats_kde[y_train == i]
        y_test_pred_i = np.ones(len(df_cell_test_feats_kde), "d")
        for feat in df_cell_train_feats_kde.columns.values:
            X_feat = X[feat].values
            kde = gaussian_kde(X_feat, "scott")
            y_test_pred_i *= kde.evaluate(df_cell_test_feats_kde[feat].values)
        y_test_pred[:, i] += y_test_pred_i
    return y_test_pred

def kde_opt3c(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        df_new = pd.DataFrame()
        df_new["hour"] = df["hour"]
        df_new["weekday"] = df["weekday"] + df["hour"] / 24.
        df_new["accuracy"] = df["accuracy"].apply(lambda x: np.log10(x))
        df_new["x"] = df["x"]
        df_new["y"] = df["y"]
        return df_new
    logging.info("train kde_opt3c model")
    df_cell_train_feats_kde = prepare_feats(df_cell_train_feats)
    df_cell_test_feats_kde = prepare_feats(df_cell_test_feats)
    n_class = len(np.unique(y_train))
    y_test_pred = np.zeros((len(df_cell_test_feats_kde), n_class), "d")
    for i in range(n_class):
        X = df_cell_train_feats_kde[y_train == i]
        y_test_pred_i = np.ones(len(df_cell_test_feats_kde), "d")
        for feat in df_cell_train_feats_kde.columns.values:
            X_feat = X[feat].values
            kde = gaussian_kde(X_feat, "scott")
            kde = gaussian_kde(X_feat, kde.factor * 0.741379)
            y_test_pred_i *= kde.evaluate(df_cell_test_feats_kde[feat].values)
        y_test_pred[:, i] += y_test_pred_i
    return y_test_pred

def kde_opt4(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        df_new = pd.DataFrame()
        df_new["hour"] = df["hour"]
        df_new["weekday"] = df["weekday"] + df["hour"] / 24.
        df_new["accuracy"] = df["accuracy"].apply(lambda x: np.log10(x))
        df_new["x"] = df["x"]
        df_new["y"] = df["y"]
        return df_new
    logging.info("train kde_opt4 model")
    df_cell_train_feats_kde = prepare_feats(df_cell_train_feats)
    df_cell_test_feats_kde = prepare_feats(df_cell_test_feats)
    n_class = len(np.unique(y_train))
    y_test_pred = np.zeros((len(df_cell_test_feats_kde), n_class), "d")
    for i in range(n_class):
        X = df_cell_train_feats_kde[y_train == i]
        y_test_pred_i = np.ones(len(df_cell_test_feats_kde), "d")
        for feat in df_cell_train_feats_kde.columns.values:
            X_feat = X[feat].values
            BGK10_output = kdeBGK10(X_feat)
            if BGK10_output is None:
                kde = gaussian_kde(X_feat, "scott")
                kde = gaussian_kde(X_feat, kde.factor * 0.741379)
                y_test_pred_i *= kde.evaluate(df_cell_test_feats_kde[feat].values)
            else:
                bandwidth, mesh, density = BGK10_output
                kde = KernelDensity(kernel='gaussian', metric='manhattan', bandwidth=bandwidth)
                kde.fit(X_feat[:, np.newaxis])
                y_test_pred_i *= np.exp(kde.score_samples(df_cell_test_feats_kde[feat].values[:, np.newaxis]))
        y_test_pred[:, i] += y_test_pred_i
    return y_test_pred

model_dict = {"kde_opt1": kde_opt1, "kde_opt2c": kde_opt2c, "kde_opt3c": kde_opt3c, "kde_opt4": kde_opt4}

if __name__ == "__main__":
    config_name = sys.argv[1]
    data_path = "../../data/"
    run_model(config_name, model_dict, data_path)

