# From: gluonts/src/gluonts/time_feature/_base.py
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import DateOffset

import math


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5

class Year(TimeFeature):
    """1900年から2100年までの値を[-0.5から0.5]に変換する"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.year - 1900) / 200.0 - 0.5


class SCSecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5] translated by sin and cos"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        period = index.second / 59.0 * (2*math.pi)
        return [np.sin(period)/2, np.cos(period)/2]


class SCMinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5] translated by sin and cos"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        period = index.minute / 59.0 * (2*math.pi)
        return [np.sin(period)/2, np.cos(period)/2]


class SCHourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5] translated by sin and cos"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        period = index.hour / 23.0 * (2*math.pi)
        return [np.sin(period)/2, np.cos(period)/2]


class SCDayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5] translated by sin and cos"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        period = index.dayofweek / 6.0 * (2*math.pi)
        return [np.sin(period)/2, np.cos(period)/2]


class SCDayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5] translated by sin and cos"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        period = (index.day - 1) / 30.0 * (2*math.pi)
        return [np.sin(period)/2, np.cos(period)/2]


class SCDayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5] translated by sin and cos"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        period = (index.dayofyear - 1) / 365.0 * (2*math.pi)
        return [np.sin(period)/2, np.cos(period)/2]


class SCMonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5] translated by sin and cos"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        period = (index.month - 1) / 11.0 * (2*math.pi)
        return [np.sin(period)/2, np.cos(period)/2]


class SCWeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5] translated by sin and cos"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        period = (index.isocalendar().week - 1) / 52.0 * (2*math.pi)
        return [np.sin(period)/2, np.cos(period)/2]

class WeatherOffset(DateOffset):
    def __init__(self):
        super().__init__()


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        WeatherOffset: [
            SCHourOfDay,
            SCDayOfWeek,
            SCDayOfMonth,
            SCDayOfYear
        ]
    }
    freq_str = "Weather"

    if freq_str == "Weather":
        offset = WeatherOffset()
    else:
        offset = to_offset(freq_str)


    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
