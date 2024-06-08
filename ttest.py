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

def main(filename1, filename2):
    mse1 = []
    mae1 = []

    pred_num = 0

    data = read_file(filename1)
    for line in data:
        if line[:3] == "mse":
            line = line.replace("\n", "")
            line = line.replace(" ", "")
            values = line.split(",")

            mse1.append((pred_ren[pred_num], float(values[0][4:])))
            mae1.append((pred_ren[pred_num], float(values[1][4:])))
            
            if pred_num == 3:
                pred_num = 0
            else:
                pred_num += 1
    
    mse2 = []
    mae2 = []

    data = read_file(filename2)
    for line in data:
        if line[:3] == "mse":
            line = line.replace("\n", "")
            line = line.replace(" ", "")
            values = line.split(",")

            mse2.append((pred_ren[pred_num], float(values[0][4:])))
            mae2.append((pred_ren[pred_num], float(values[1][4:])))
            
            if pred_num == 3:
                pred_num = 0
            else:
                pred_num += 1

    print("T test")
    print("96", ttest_ind(get_each_pred(mae1, 0), get_each_pred(mae2, 0)))
    print("192", ttest_ind(get_each_pred(mae1, 1), get_each_pred(mae2, 1)))
    print("336", ttest_ind(get_each_pred(mae1, 2), get_each_pred(mae2, 2)))
    print("720", ttest_ind(get_each_pred(mae1, 3), get_each_pred(mae2, 3)))

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0], args[1])