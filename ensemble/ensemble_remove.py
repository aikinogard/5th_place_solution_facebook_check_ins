# scan the ensemble to see whether we should remove one from the ensemble

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


def folderToDict(model_output_path, c, probas=None):
    logging.info("merge folder %s" % model_output_path)
    files = glob.glob(os.path.join(model_output_path, "*.csv"))
    if probas is None:
        probas = defaultdict(lambda: defaultdict(float))
    for f in files:
        logging.info("loading... %s" % f)
        df = pd.read_csv(f, dtype={"row_id": int, "place_id": int, "proba": float})
        for i in range(len(df)):
            probas[df["row_id"][i]][df["place_id"][i]] += c * df["proba"][i]
    try:
        with open(os.path.join(model_output_path, "map3.txt"), "r") as f_score:
            logging.info("map3=%6.6f" % float(f_score.read()))
    except:
        pass
    return probas

def parseDict(probas, output_name, valid_file=None):
    df = pd.DataFrame()
    df["row_id"] = probas.keys()
    df["place_id"] = df["row_id"].apply(lambda x: map(itemgetter(0),
                        sorted(probas[x].items(), key=itemgetter(1), reverse=True)[:3]))
    if valid_file is not None:
        df_valid = pd.read_csv(valid_file, usecols=["row_id", "place_id"])
        df_valid.rename(columns={"place_id": "place_id_label"}, inplace=True)
        df_merge = pd.merge(df, df_valid, how="left", on="row_id")
        valid_score = metrics.mapk(df_merge.place_id_label.values[:, None],
                                   df_merge.place_id.values, 3)
        logging.info("total validation score: %f" % valid_score)
        del df_valid
        del df_merge
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

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    rec = []
    for ridx in range(len(model_output_paths)):
        # ridx is the index of model removed from the ensemble
        probas = None
        for m, model_output_path in enumerate(model_output_paths):
            if m == ridx:
                continue
            probas = folderToDict(model_output_path, c=cs[m], probas=probas)
        valid_score = parseDict(probas, config_name, valid_file)
        logging.info("scan result: %s, %f" % (model_output_paths[ridx], valid_score))
        rec.append((model_output_paths[ridx], valid_score))
    logging.info("FINAL RESULT:")
    logging.info(rec)
    logging.info("SORTED RESULT:")
    logging.info(sorted(rec, key=lambda x: x[1], reverse=True))





