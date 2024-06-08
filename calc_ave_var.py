import sys
import math
from scipy.stats import ttest_ind

pred_ren = [96, 192, 336, 720]

def read_file(file_name):
    with open(file_name, 'r') as file:
        data = file.readlines()
    return data

def get_each_pred(values, target_num):
    result = []

    for value in values:
        if pred_ren[target_num] == value[0]:
            result.append(value[1])
    
    return result

def calc_ave_var(values):
    ave = sum(values) / len(values)
    var = sum([(value - ave) ** 2 for value in values]) / len(values)

    return ave, math.sqrt(var)

def main(filename):
    mse = []
    msa = []

    pred_num = 0

    data = read_file(filename)
    for line in data:
        if line[:3] == "mse":
            line = line.replace("\n", "")
            line = line.replace(" ", "")
            values = line.split(",")

            mse.append((pred_ren[pred_num], float(values[0][4:])))
            msa.append((pred_ren[pred_num], float(values[1][4:])))
            
            if pred_num == 3:
                pred_num = 0
            else:
                pred_num += 1
    
    print("MSE")
    print(pred_ren[0], calc_ave_var(get_each_pred(mse, 0)))
    print(pred_ren[1], calc_ave_var(get_each_pred(mse, 1)))
    print(pred_ren[2], calc_ave_var(get_each_pred(mse, 2)))
    print(pred_ren[3], calc_ave_var(get_each_pred(mse, 3)))

    print("MSA")
    print(pred_ren[0], calc_ave_var(get_each_pred(msa, 0)))
    print(pred_ren[1], calc_ave_var(get_each_pred(msa, 1)))
    print(pred_ren[2], calc_ave_var(get_each_pred(msa, 2)))
    print(pred_ren[3], calc_ave_var(get_each_pred(msa, 3)))

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0])