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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import os

from data_loader import data_loader
from gain import gain


def main (data_name, batch_size, hint_rate, alpha, iterations):
    '''Main function for simulation 
  
    Args:
        - data_name: file name input
        - batch:size: batch size
        - hint_rate: hint rate
        - alpha: hyperparameter
        - iterations: iterations
    
    Returns:
        - imputed_data_x: imputed data
    '''
  
    gain_parameters = {'batch_size': batch_size,
                       'hint_rate': hint_rate,
                       'alpha': alpha,
                       'iterations': iterations}
  
    # Load data and introduce missingness
    miss_data_x, data_m, col_names = data_loader(data_name)
  
    # Impute missing data
    imputed_data_list, loss_df = gain(miss_data_x, col_names, gain_parameters)
  
    return imputed_data_list, loss_df

foldername = "../../simulations/gain"
os.makedirs(foldername, exist_ok=True)
for i in range(1, 1001):
    k = str(i).zfill(4)
    filename = "../../data/" + "SRS_" + k + ".csv"
    imputed_data_list, loss_df = main(filename, 128, 0.95, 100, 2000)
    loss_df.to_csv(foldername + "/LOSS_" + k + ".csv")
    with pd.ExcelWriter(foldername + "/GAIN_" + k + ".xlsx") as writer:
        for i in range(len(imputed_data_list)):
            df = pd.DataFrame(imputed_data_list[i])
            df.to_excel(writer, sheet_name=f"Sheet_{i+1}", index=False)
