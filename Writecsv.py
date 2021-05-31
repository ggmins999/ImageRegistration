import xlwt
import pandas as pd
import os
from pandas import ExcelWriter
import numpy as np
import re
path = "/Users/gigiminsky/Downloads/C4"
rootdir = path
filenames = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith(".csv"):
            filenames.append([file,subdir])
            #print (os.path.join(subdir, file))



def save_xls(filenames, xls_path):
    with ExcelWriter(xls_path) as writer:
        for file,subdir in filenames:
            df = pd.read_csv(os.path.join(subdir,file))
            print(subdir,file)
            n = df["taildamage"].size
            if n == 0:
                continue
            c = np.zeros(df["taildamage"].size)
            c[0] = df.mean(axis=0)['taildamage']
            df["damageaverage"] = c
            s = np.zeros(df["taildamage"].size)
            s[0] = df.std(axis=0)['taildamage']
            df["damagestd"] = s
            dirname = subdir.split("/")[-1]
            suffix = dirname.split("_")[0]
            print(suffix)
            df.to_excel(writer,suffix)
        writer.save()

save_xls(filenames, os.path.join(path,"combined.xls"))