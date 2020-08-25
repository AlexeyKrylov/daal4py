import openpyxl
import numpy as np
import pandas as pd
import daal4py.sklearn.metrics as daal_sklm
from numpy import genfromtxt
import sklearn.metrics as sklm
from timeit import default_timer as timer
from sklearn.datasets import make_regression
from openpyxl.utils.dataframe import dataframe_to_rows


def ttt():
  print("-----------------------------------------------------------------------")

def Error(a, b):
    count = 0;
    for i in range(len(a)):
        if a[i] != b[i]:
            print(f"Error : Daal : {a[i]}\n\t Skl : {b[i]}")
            count+=1
        if count >= 4: 
            return
                                   
def main():

  wb = openpyxl.Workbook()
  ws = wb.active

  rows = [2, 4, 8, 64, 128, 256, 512, 1024, 2048, 4096]
  cols = [4, 8, 10, 16, 21, 32, 50]
  ppp = [0, 1, 2, 3, 4, 5]
  data = list()
  i = 0
  for p_ in ppp:
    for n_cols in cols:
      for n_rows in rows:

        i+=1
        ttt()
        print(i)
        print(f'({p_}, {n_cols}, {n_rows})')

        data1, empt = make_regression(n_samples=n_rows, n_features=n_cols, n_targets=0, random_state=777 + n_rows + n_cols)
        data2, empt = make_regression(n_samples=n_rows, n_features=n_cols, n_targets=0, random_state=778 + n_rows + n_cols)
        
        t10 = timer() 
        result1 = daal_sklm.pairwise_distances(data1, data2, metric='minkowski', p=p_)
        t11 = timer()
        d_time = t11 - t10
        print(f"DAAL implementation:  {t11 - t10} ")

        t20 = timer() 
        if p_ == 0:
            result2 = sklm.pairwise_distances(data1, data2, metric='chebyshev', p=p_)
        else:
            result2 = sklm.pairwise_distances(data1, data2, metric='minkowski', p=p_)
        t21 = timer()
        s_time = t21 - t20
        print(f"Original implementation:  {t21 - t20} ") 
        ttt()
        
        data.append({'p' : p_, 'n_rows' : n_rows, 'n_cols' : n_cols, 'n_rows * n_cols' : n_rows * n_cols, 'DAAL_time_avx512' : d_time, 'Orig_time_avx512' : s_time, 'Orig_time / D_time avx512' : s_time / d_time, 'Error avx512' : ((np.abs(result1 - result2)) / result2).ravel().mean(), 'Error % avx512' : (result1 != result2).ravel().mean(),  'Error MAX otn avx512' : ((np.abs(result1 - result2)) / result2).ravel().max(), 'Error MAX avx512' : (np.abs(result1 - result2)).ravel().max()})
         
        if (result1 != result2).ravel().mean() > 0.01:
            ttt()
            Error(result1.ravel(), result2.ravel())
            ttt()

  for r in dataframe_to_rows(pd.DataFrame(data).sort_values(by=['n_rows * n_cols']), index=False, header=True):
   ws.append(r)

  wb.save("test_minkowski_daal_math_00002_avx512.xlsx")

  print("Complete TEST")


if __name__ == '__main__':
    main()