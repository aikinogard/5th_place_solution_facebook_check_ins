import sys
import os
import glob
import logging
import ConfigParser
from collections import defaultdict
from operator import itemgetter
import pandas as pd
import numpy as np
import ml_metrics as metrics
from bayes_opt import BayesianOptimization

def generate_cs(n_model):
    craw = np.sort(np.append(np.random.rand(n_model - 1), [0., 1.]))
    return np.diff(craw)

def check_exist(c, rec):
    c_rec = rec.drop("map3_score", axis=1).values
    if len(c_rec) == 0:
        return False
    for ci in c_rec:
        if np.mean(np.abs(c - ci)) < 0.001:
            return True
    return False

def load_models(model_output_paths):
    dfs = []
    for model_output_path in model_output_paths:
        logging.info("loading folder %s" % model_output_path)
        files = glob.glob(os.path.join(model_output_path, "*.csv"))
        df_model = pd.DataFrame(columns=["row_id", "place_id", "proba"])
        for f in files:
            logging.info("loading... %s" % f)
            df = pd.read_csv(f, dtype={"row_id": int, "place_id": int, "proba": float})
            df_model = df_model.append(df, ignore_index=True)
        dfs.append(df_model)
    return dfs

def ensemble_score(dfs, model_output_paths, df_valid, **cs):
    c = [cs["c%d" % m] for m in range(len(cs))]
    probas = defaultdict(lambda: defaultdict(float))
    for m, df_model in enumerate(dfs):
        #logging.info("scoring %d, %s" % (m, model_output_paths[m]))
        for i in range(len(df_model)):
            probas[df_model["row_id"][i]][df_model["place_id"][i]] += c[m] * df_model["proba"][i]
    df = pd.DataFrame()
    df["row_id"] = probas.keys()
    df["place_id"] = df["row_id"].apply(lambda x: map(itemgetter(0),
                        sorted(probas[x].items(), key=itemgetter(1), reverse=True)[:3]))
    df_merge = pd.merge(df, df_valid, how="left", on="row_id")
    valid_score = metrics.mapk(df_merge.place_id_label.values[:, None],
                               df_merge.place_id.values, 3)
    return valid_score

if __name__ == "__main__":
    config_name = sys.argv[1]
    config = ConfigParser.ConfigParser()
    try:
        config.read("ensembles.config")
        valid_mode_on = config.getboolean(config_name, "valid_mode_on")
        if valid_mode_on:
            valid_file = "../data/train-va.csv"
        else:
            valid_file = None
        model_output_paths = map(lambda x: x.strip(), config.get(config_name, "model_output_paths").split(","))
        try:
            cs = map(float, config.get(config_name, "cs").split(","))
            assert len(cs) == len(model_output_paths)
        except ConfigParser.NoOptionError:
            cs = np.ones(len(model_output_paths))
    except Exception as e:
        logging.error("Could not load configuration file from models.config")
        logging.error(str(e))

    df_valid = pd.read_csv(valid_file, usecols=["row_id", "place_id"])
    df_valid.rename(columns={"place_id": "place_id_label"}, inplace=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    dfs = load_models(model_output_paths)
    def target(**cs):
        return ensemble_score(dfs, model_output_paths, df_valid, **cs)

    bo = BayesianOptimization(target, {"c%d" % m: (0., 1.) for m in range(len(model_output_paths))})
    bo.maximize(n_iter=100, kappa=5)


