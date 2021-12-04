import os
import functools

data_dir = r"/home/mloper23/Datasets/E/Holors"
data_real = r"/home/mloper23/Datasets/E/Reconstructions"

x_set = os.listdir(f"{data_dir}/D1") + os.listdir(f"{data_dir}/D2") + os.listdir(f"{data_dir}/D3") + os.listdir(
    f"{data_dir}/D4") + os.listdir(f"{data_dir}/D5")
y_set = os.listdir(f"{data_real}/D1") + os.listdir(f"{data_real}/D2") + os.listdir(f"{data_real}/D3") + os.listdir(
    f"{data_real}/D4") + os.listdir(f"{data_real}/D5")

if functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, x_set, y_set), True):
    print("The lists are the same")
else:
    print("The lists are not the same")
