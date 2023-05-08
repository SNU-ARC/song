import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Used average of 3 iterations for whole batch. 
for file in ["batch"] :

    f = open(f"./result/{file}.txt", 'r')
    df = pd.DataFrame(columns=["dataset", "top_k", "pq_size", "count", "latency", "QPS", "recall"])

    while True:
        line = f.readline()
        if not line :
            break

        line = re.split(r"\s|:|@|%", line)
        
        if "DATASET" in line :
            dataset = line[1]
            top_k = line[3]
            pq_size = line[5]
            row = [dataset, top_k, pq_size, 0, sys.float_info.max, 0, 0]
        
        elif "microseconds" in line and "using" in line :
            latency = line[1]
            row[4] = min(float(latency), row[4])
            row[3] = row[3] + 1

        elif "QPS" in line :
            QPS = line[3]
            row[5] = max(float(QPS), row[5])

        elif "Recall" in line and row[3] == 3 :
            recall = line[3]
            row[6] = float(recall) / 100
            df.loc[len(df)] = row

        else :
            pass

    for dataset in ["sift1m", "gist1m",  "crawl", "deep1m", "msong", "glove-100"] :
        for top_k in [1, 10] :
            df_save = df.loc[df["dataset"] == dataset]
            df_save = df_save.loc[df_save["top_k"] == str(top_k)]
            df_save = df_save.drop(["dataset", "top_k", "pq_size", "count", "latency"], axis=1)
            df_save = df_save.sort_values(by=["recall", "QPS"], ascending=[True, False]).reindex(columns=["recall", "QPS"])
            # Remove others except highest QPS
            df_save = df_save.drop_duplicates(subset=["recall"], keep="first").T
            df_save.to_csv(f"./result/song_whole_batch/{file}_{dataset}_top_{top_k}.csv", index=False, header=False)
    f.close()

# Used 1 iteration for single cache since result get cached
for file in ["single"] :

    f = open(f"./result/{file}.txt", 'r')
    df = pd.DataFrame(columns=["dataset", "top_k", "pq_size", "count", "latency", "QPS", "recall"])

    while True:
        line = f.readline()
        if not line :
            break

        line = re.split(r"\s|:|@|%", line)

        if "DATASET" in line :
            dataset = line[1]
            top_k = line[3]
            pq_size = line[5]
            row = [dataset, top_k, pq_size, 0, 0, 0, 0]

        elif "microseconds" in line and "using" in line :
            latency = line[1]
            row[4] = float(latency) + row[4]
            row[3] = row[3] + 1

        elif "QPS" in line :
            QPS = line[3]
            row[5] = float(QPS) + row[5]

        elif "Recall" in line :
            recall = line[3]
            row[6] = float(recall) / 100
            row[4] = row[4] / row[3]
            row[5] = row[5] / row[3]
            df.loc[len(df)] = row

        else :
            pass

    for dataset in ["sift1m", "gist1m",  "crawl", "deep1m", "msong", "glove-100"] :
        for top_k in [1, 10] :
            df_save = df.loc[df["dataset"] == dataset]
            df_save = df_save.loc[df_save["top_k"] == str(top_k)]
            df_save = df_save.drop(["dataset", "top_k", "pq_size", "count", "QPS"], axis=1)
            df_save = df_save.sort_values(by=["recall", "latency"], ascending=[True, True]).reindex(columns=["recall", "latency"])
            # Remove others excpet lowest latency
            df_save = df_save.drop_duplicates(subset=["recall"], keep="first").T
            df_save.to_csv(f"./result/song_single_batch/{file}_{dataset}_top_{top_k}.csv", index=False, header=False)
    f.close()
