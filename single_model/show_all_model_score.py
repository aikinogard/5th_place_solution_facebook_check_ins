import sys
import os

for model_path in ["knn/", "kde/", "xgb/", "rf/", "et/", "nb/", "svc/"]:
    print "=====%s=====" % model_path
    for d in sorted(os.listdir(model_path), key=lambda x: x[:-2]):
        if os.path.isdir(model_path + d):
            try:
                with open(os.path.join(model_path + d, "map3.txt"), "r") as f:
                    print d, f.read()
            except:
                print d