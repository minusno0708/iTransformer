import pandas as pd

year = 2023

rawdate = -0.0918

day_of_year = int((rawdate + 0.5) * 365 + 1 +0.5)

print(day_of_year)