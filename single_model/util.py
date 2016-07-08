import sys
import os
import time
import logging
import ConfigParser
import operator
import numpy as np
import pandas as pd
import ml_metrics as metrics
from sklearn.preprocessing import LabelEncoder

def prepare_data(df, n_cell_x, n_cell_y):
    """
    Feature engineering and computation of the grid.
    """
    #Creating the grid
    size_x = 10. / n_cell_x
    size_y = 10. / n_cell_y
    eps = 0.00001  
    xs = np.where(df.x.values < eps, 0, df.x.values - eps)
    ys = np.where(df.y.values < eps, 0, df.y.values - eps)
    pos_x = (xs / size_x).astype(np.int)
    pos_y = (ys / size_y).astype(np.int)
    df['grid_cell_x'] = pos_x
    df['grid_cell_y'] = pos_y
    
    #Feature engineering
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]') 
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') 
                               for mn in df.time.values)

    df['hour'] = d_times.hour + d_times.minute / 60.
    df['weekday'] = d_times.weekday
    df['day'] = d_times.dayofyear
    df['month'] = d_times.month
    df['year'] = d_times.year - 2013

    df = df.drop(['time'], axis=1)
    return df

def train_hour_periodic(df_train, time_edge):
    logging.info("add train data to show period in hour, time_edge=%f" % time_edge)
    add_data = df_train[df_train.hour < time_edge].copy().reindex()
    add_data.hour += 24
    df_train = pd.concat([df_train, add_data])

    add_data = df_train[df_train.hour > 24 - time_edge].copy().reindex()
    add_data.hour -= 24
    df_train = pd.concat([df_train, add_data])
    return df_train

def process_one_cell(df_train, df_test, valid_mode_on,
                     gx_id, gy_id, x_border, y_border, th, model_list):
    """   
    Classification inside one grid cell.
    """
    #Working on df_train
    #filtering occurance smaller than th
    #consider border of cell
    df_cell_train = df_train.loc[(df_train.grid_cell_x == gx_id) & (df_train.grid_cell_y == gy_id)]
    x_min = df_cell_train.x.min()
    x_max = df_cell_train.x.max()
    y_min = df_cell_train.y.min()
    y_max = df_cell_train.y.max()
    df_cell_train = df_train.loc[(df_train.x >= x_min - x_border) & (df_train.x <= x_max + x_border)
                                  & (df_train.y >= y_min - y_border) & (df_train.y <= y_max + y_border)]
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    df_cell_train = df_cell_train.loc[mask]

    #Working on df_test
    df_cell_test = df_test.loc[(df_test.grid_cell_x == gx_id) & (df_test.grid_cell_y == gy_id)]
    row_ids = df_cell_test.row_id.values

    #Preparing data
    #remove columns and encode label
    le = LabelEncoder()
    y_train = le.fit_transform(df_cell_train.place_id.values)
    l_train = df_cell_train.shape[0]
    l_test = df_cell_test.shape[0]
    n_class = len(le.classes_)
    logging.info("number of class: %d" % n_class)
    if valid_mode_on:
        logging.info("validation mode")
        logging.info("train size: %d, validation size: %d" % (l_train, l_test))
        logging.info("%d labels in validation is not in train" 
                     % len(set(df_cell_test.place_id.values) - set(df_cell_train.place_id.values)))
    else:
        logging.info("prediction mode")
        logging.info("train size: %d, test size: %d" % (l_train, l_test))

    df_cell_train_feats = df_cell_train.drop(['place_id', 'grid_cell_x', 'grid_cell_y', 'row_id'], axis=1)
    feats = df_cell_train_feats.columns.values
    df_cell_test_feats = df_cell_test[feats]

    y_test_pred = np.zeros((df_cell_test_feats.shape[0], n_class))
    for clf in model_list:
        y_test_pred_model = clf(df_cell_train_feats, y_train, df_cell_test_feats)
        y_test_pred += y_test_pred_model

    if valid_mode_on:
        pred_labels = le.inverse_transform(np.argsort(y_test_pred, axis=1)[:, ::-1][:, :10])
        valid_score = metrics.mapk(df_cell_test.place_id.values[:, None], pred_labels, 3)
        logging.info("valid score = %6.6f" % valid_score)
    else:
        valid_score = None

    #return list of (row_id, place_id, proba)
    top10_label = le.inverse_transform(np.argsort(y_test_pred, axis=1)[:, ::-1][:, :10])
    top10_proba_raw = np.sort(y_test_pred, axis=1)[:, ::-1][:, :10]
    top10_proba = top10_proba_raw / np.sum(top10_proba_raw, axis=1)[:, None]
    probas = []
    for i, rid in enumerate(row_ids):
        if i == 0:
            probas = np.array([[rid] * 10, top10_label[i], top10_proba[i]]).T
        else:
            probas = np.vstack([probas, np.array([[rid] * 10, top10_label[i], top10_proba[i]]).T])
    return probas, valid_score, l_test

def process_grid(df_train, df_test, valid_mode_on, x_border, y_border, th,
                 n_cell_x, n_cell_y, model_list, output_path, gx_ids=None):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """ 
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    if valid_mode_on:
        valid_score_total = 0.
        valid_size_total = 0

    if valid_mode_on:
        # take the middle 1/10, 1/10 as validation sample
        n_cell_xs = range(n_cell_x / 2, 3 * n_cell_x / 5)
        n_cell_ys =range(n_cell_y / 2, 3 * n_cell_y / 5)
    else:
        if gx_ids is not None and len(gx_ids) == 2:
            # customize x
            n_cell_xs = range(gx_ids[0], gx_ids[1])
            n_cell_ys = range(n_cell_y)
        else:
            # use the whole set
            n_cell_xs = range(n_cell_x)
            n_cell_ys = range(n_cell_y)

    time_start = time.time()
    for gx_id in n_cell_xs:
        new_output = True
        for gy_id in n_cell_ys:
            logging.info("=====Grid: (%d, %d)=====" % (gx_id, gy_id))
            #Applying classifier to one grid cell
            probas, valid_score, valid_size = process_one_cell(df_train, df_test, valid_mode_on,
                                                               gx_id, gy_id, x_border, y_border,
                                                               th, model_list)
            if valid_mode_on:
                valid_score_total += valid_score * valid_size
                valid_size_total += valid_size                
            #Updating predictions
            if new_output:
                probas_rec = probas
                new_output = False
            else:
                probas_rec = np.vstack([probas_rec, probas])
        logging.info("output probability file...")
        df_probas = pd.DataFrame(probas_rec, columns=["row_id", "place_id", "proba"])
        df_probas["row_id"] = df_probas["row_id"].astype(int)
        df_probas["place_id"] = df_probas["place_id"].astype(int)
        df_probas["proba"] = df_probas["proba"].astype(float)
        df_probas.to_csv(os.path.join(output_path, "%d.csv" % gx_id), index=False)
    logging.info("******Total Run Time: %4.2f minutes******" % round((time.time() - time_start) / 60., 2))

    if valid_mode_on:
        logging.info("total validation score: %f" % (valid_score_total / valid_size_total))
        np.savetxt(os.path.join(output_path, "map3.txt"), [valid_score_total / valid_size_total], fmt="%f")
        return valid_score_total / valid_size_total

def run_model(config_name, model_dict, data_path):
    config = ConfigParser.ConfigParser()
    try:
        config.read("models.config")
        valid_mode_on = config.getboolean(config_name, "valid_mode_on")
        if valid_mode_on:
            train_path = os.path.join(data_path, "train-tr.csv")
            test_path = os.path.join(data_path, "train-va.csv")
        else:
            train_path = os.path.join(data_path, "train.csv")
            test_path = os.path.join(data_path, "test.csv")
        n_cell_x = config.getint(config_name, "n_cell_x")
        n_cell_y = config.getint(config_name, "n_cell_y")
        x_border = config.getfloat(config_name, "x_border")
        y_border = config.getfloat(config_name, "y_border")
        th = config.getint(config_name, "th")
        model_list = map(lambda x: model_dict[x.strip()], config.get(config_name, "model_list").split(","))
        output_path = os.path.join(config.get(config_name, "output_path"), config_name)

        try:
            gx_ids = map(int, config.get(config_name, "gx_ids").split(","))
        except ConfigParser.NoOptionError:
            gx_ids = None

        try:
            time_edge = config.getfloat(config_name, "time_edge")
        except ConfigParser.NoOptionError:
            time_edge = -1

    except Exception as e:
        logging.error("Could not load configuration file from models.config")
        logging.error(str(e))

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info("Running config: [%s]" % config_name)
    logging.info('Loading data')
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    logging.info('Preparing train data')
    df_train = prepare_data(df_train, n_cell_x, n_cell_y)
    if time_edge > 0:
        df_train = train_hour_periodic(df_train, time_edge)
    logging.info('Preparing test data')
    df_test = prepare_data(df_test, n_cell_x, n_cell_y)

    return process_grid(df_train, df_test, valid_mode_on, x_border, y_border, th,
                 n_cell_x, n_cell_y, model_list, output_path, gx_ids)





