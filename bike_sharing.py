import pandas as pd
import numpy as np

train = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.train.csv").to_numpy()
test_data = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.test.csv").to_numpy()
test_label = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.sampleSolution.csv").to_numpy

#Linear Regression


