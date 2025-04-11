# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Necessary packages
import numpy as np
import pandas as pd


def data_loader (file_name):
    '''Loads datasets and introduce missingness.
    Args:
    - file_name: file_name with file path

    Returns:
    data_x: original data
    data_m: indicator matrix for missing components
    col_names: original variable names in the dataset
    '''

    # Load data
    data_x = pd.read_csv(file_name)
    col_names = data_x.columns.tolist()
    data_x = data_x.to_numpy()
    data_m = 1 - (1 * np.isnan(data_x))
    
    return data_x, data_m, col_names
