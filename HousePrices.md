

```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
```


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
print('shape of train is: ', train.shape)
print('shape of test is: ', test.shape)
```

    shape of train is:  (1460, 81)
    shape of test is:  (1459, 80)



```python
print(train.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
    Id               1460 non-null int64
    MSSubClass       1460 non-null int64
    MSZoning         1460 non-null object
    LotFrontage      1201 non-null float64
    LotArea          1460 non-null int64
    Street           1460 non-null object
    Alley            91 non-null object
    LotShape         1460 non-null object
    LandContour      1460 non-null object
    Utilities        1460 non-null object
    LotConfig        1460 non-null object
    LandSlope        1460 non-null object
    Neighborhood     1460 non-null object
    Condition1       1460 non-null object
    Condition2       1460 non-null object
    BldgType         1460 non-null object
    HouseStyle       1460 non-null object
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    RoofStyle        1460 non-null object
    RoofMatl         1460 non-null object
    Exterior1st      1460 non-null object
    Exterior2nd      1460 non-null object
    MasVnrType       1452 non-null object
    MasVnrArea       1452 non-null float64
    ExterQual        1460 non-null object
    ExterCond        1460 non-null object
    Foundation       1460 non-null object
    BsmtQual         1423 non-null object
    BsmtCond         1423 non-null object
    BsmtExposure     1422 non-null object
    BsmtFinType1     1423 non-null object
    BsmtFinSF1       1460 non-null int64
    BsmtFinType2     1422 non-null object
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    Heating          1460 non-null object
    HeatingQC        1460 non-null object
    CentralAir       1460 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    KitchenQual      1460 non-null object
    TotRmsAbvGrd     1460 non-null int64
    Functional       1460 non-null object
    Fireplaces       1460 non-null int64
    FireplaceQu      770 non-null object
    GarageType       1379 non-null object
    GarageYrBlt      1379 non-null float64
    GarageFinish     1379 non-null object
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    GarageQual       1379 non-null object
    GarageCond       1379 non-null object
    PavedDrive       1460 non-null object
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    PoolQC           7 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    SaleType         1460 non-null object
    SaleCondition    1460 non-null object
    SalePrice        1460 non-null int64
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    None

Data Dictionary

SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
MSSubClass: The building class
MSZoning: The general zoning classification
LotFrontage: Linear feet of street connected to property
LotArea: Lot size in square feet
Street: Type of road access
Alley: Type of alley access
LotShape: General shape of property
LandContour: Flatness of the property
Utilities: Type of utilities available
LotConfig: Lot configuration
LandSlope: Slope of property
Neighborhood: Physical locations within Ames city limits
Condition1: Proximity to main road or railroad
Condition2: Proximity to main road or railroad (if a second is present)
BldgType: Type of dwelling
HouseStyle: Style of dwelling
OverallQual: Overall material and finish quality
OverallCond: Overall condition rating
YearBuilt: Original construction date
YearRemodAdd: Remodel date
RoofStyle: Type of roof
RoofMatl: Roof material
Exterior1st: Exterior covering on house
Exterior2nd: Exterior covering on house (if more than one material)
MasVnrType: Masonry veneer type
MasVnrArea: Masonry veneer area in square feet
ExterQual: Exterior material quality
ExterCond: Present condition of the material on the exterior
Foundation: Type of foundation
BsmtQual: Height of the basement
BsmtCond: General condition of the basement
BsmtExposure: Walkout or garden level basement walls
BsmtFinType1: Quality of basement finished area
BsmtFinSF1: Type 1 finished square feet
BsmtFinType2: Quality of second finished area (if present)
BsmtFinSF2: Type 2 finished square feet
BsmtUnfSF: Unfinished square feet of basement area
TotalBsmtSF: Total square feet of basement area
Heating: Type of heating
HeatingQC: Heating quality and condition
CentralAir: Central air conditioning
Electrical: Electrical system
1stFlrSF: First Floor square feet
2ndFlrSF: Second floor square feet
LowQualFinSF: Low quality finished square feet (all floors)
GrLivArea: Above grade (ground) living area square feet
BsmtFullBath: Basement full bathrooms
BsmtHalfBath: Basement half bathrooms
FullBath: Full bathrooms above grade
HalfBath: Half baths above grade
Bedroom: Number of bedrooms above basement level
Kitchen: Number of kitchens
KitchenQual: Kitchen quality
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
Functional: Home functionality rating
Fireplaces: Number of fireplaces
FireplaceQu: Fireplace quality
GarageType: Garage location
GarageYrBlt: Year garage was built
GarageFinish: Interior finish of the garage
GarageCars: Size of garage in car capacity
rGarageArea: Size of garage in square feet
GarageQual: Garage quality
GarageCond: Garage condition
PavedDrive: Paved driveway
WoodDeckSF: Wood deck area in square feet
OpenPorchSF: Open porch area in square feet
EnclosedPorch: Enclosed porch area in square feet
3SsnPorch: Three season porch area in square feet
ScreenPorch: Screen porch area in square feet
PoolArea: Pool area in square feet
PoolQC: Pool quality
Fence: Fence quality
MiscFeature: Miscellaneous feature not covered in other categories
MiscVal: $Value of miscellaneous feature
MoSold: Month Sold
YrSold: Year Sold
SaleType: Type of sale
SaleCondition: Condition of sale

```python
x = pd.DataFrame(train.dtypes[train.dtypes == int]).reset_index()
```


```python
x
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Id</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSSubClass</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LotArea</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OverallQual</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OverallCond</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>5</th>
      <td>YearBuilt</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>YearRemodAdd</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BsmtFinSF1</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BsmtFinSF2</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BsmtUnfSF</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TotalBsmtSF</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1stFlrSF</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2ndFlrSF</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>13</th>
      <td>LowQualFinSF</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GrLivArea</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BsmtFullBath</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>16</th>
      <td>BsmtHalfBath</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>17</th>
      <td>FullBath</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HalfBath</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BedroomAbvGr</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>20</th>
      <td>KitchenAbvGr</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>21</th>
      <td>TotRmsAbvGrd</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Fireplaces</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>23</th>
      <td>GarageCars</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>24</th>
      <td>GarageArea</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>25</th>
      <td>WoodDeckSF</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>26</th>
      <td>OpenPorchSF</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>27</th>
      <td>EnclosedPorch</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3SsnPorch</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>29</th>
      <td>ScreenPorch</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PoolArea</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>31</th>
      <td>MiscVal</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>32</th>
      <td>MoSold</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>33</th>
      <td>YrSold</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>34</th>
      <td>SalePrice</td>
      <td>int64</td>
    </tr>
  </tbody>
</table>
</div>




```python
if train.columns[6] == x['index'][1]:
    print(train[train.columns[1]].isnull().sum())
```


```python
train.select_dtypes(exclude = ['object']).isnull().sum()
```




    Id                 0
    MSSubClass         0
    LotFrontage      259
    LotArea            0
    OverallQual        0
    OverallCond        0
    YearBuilt          0
    YearRemodAdd       0
    MasVnrArea         8
    BsmtFinSF1         0
    BsmtFinSF2         0
    BsmtUnfSF          0
    TotalBsmtSF        0
    1stFlrSF           0
    2ndFlrSF           0
    LowQualFinSF       0
    GrLivArea          0
    BsmtFullBath       0
    BsmtHalfBath       0
    FullBath           0
    HalfBath           0
    BedroomAbvGr       0
    KitchenAbvGr       0
    TotRmsAbvGrd       0
    Fireplaces         0
    GarageYrBlt       81
    GarageCars         0
    GarageArea         0
    WoodDeckSF         0
    OpenPorchSF        0
    EnclosedPorch      0
    3SsnPorch          0
    ScreenPorch        0
    PoolArea           0
    MiscVal            0
    MoSold             0
    YrSold             0
    SalePrice          0
    dtype: int64




```python
train.select_dtypes(include = ['object']).isnull().sum()
```




    MSZoning            0
    Street              0
    Alley            1369
    LotShape            0
    LandContour         0
    Utilities           0
    LotConfig           0
    LandSlope           0
    Neighborhood        0
    Condition1          0
    Condition2          0
    BldgType            0
    HouseStyle          0
    RoofStyle           0
    RoofMatl            0
    Exterior1st         0
    Exterior2nd         0
    MasVnrType          8
    ExterQual           0
    ExterCond           0
    Foundation          0
    BsmtQual           37
    BsmtCond           37
    BsmtExposure       38
    BsmtFinType1       37
    BsmtFinType2       38
    Heating             0
    HeatingQC           0
    CentralAir          0
    Electrical          1
    KitchenQual         0
    Functional          0
    FireplaceQu       690
    GarageType         81
    GarageFinish       81
    GarageQual         81
    GarageCond         81
    PavedDrive          0
    PoolQC           1453
    Fence            1179
    MiscFeature      1406
    SaleType            0
    SaleCondition       0
    dtype: int64


These missing values are a result of those aspects of the property not being present.  The nan value means that that attribute is not avalailable.

```python
(train.select_dtypes(exclude = ['object']).isnull().sum()).plot(kind = 'bar')
plt.show()
```


![png](output_11_0.png)



```python
(train.select_dtypes(include = ['object']).isnull().sum()).plot(kind = 'bar')
plt.show()
```


![png](output_12_0.png)



```python
(train.YearRemodAdd == train.YearBuilt).sum()
```




    764




```python
(train.YearRemodAdd != train.YearBuilt).sum()
```




    696


There were 764 homes that were not remodeled and 696 homes that were remodeled.

```python
pd.DataFrame([(train.YearRemodAdd == train.YearBuilt).sum(), (train.YearRemodAdd != train.YearBuilt).sum()]).plot(kind ='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0686096ba8>




![png](output_16_1.png)



```python
dict = pd.DataFrame({'yes' : ((train.YearRemodAdd != train.YearBuilt).sum()),
'no' : ((train.YearRemodAdd == train.YearBuilt).sum())}, index = [0])
```


```python
dict.unstack().plot(kind = 'bar')
plt.xlabel('Remodel Status')
plt.ylabel('Count')
plt.xticks([0,1],['Yes', 'No'])
plt.show()
```


![png](output_18_0.png)

There appear to be about 50% of the houses we are selling have been remodeled and it therefore shows that the homes are taken care of and modern.

```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>...</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>730.500000</td>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>...</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>421.610009</td>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>...</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>365.750000</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>730.500000</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1095.250000</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>...</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>




```python
(train.isnull().sum().sum())/(train.shape[0]*train.shape[1])
```




    0.05889565364451209




```python
(test.isnull().sum().sum())/(test.shape[0]*test.shape[1])
```




    0.059972583961617545




```python
train.duplicated().sum()
```




    0




```python
train.MSZoning.value_counts()
```




    RL         1151
    RM          218
    FV           65
    RH           16
    C (all)      10
    Name: MSZoning, dtype: int64




```python
train.select_dtypes(include = ['object']).columns[0]
```




    'MSZoning'




```python
def cat_barplot():
    for n in range(0, 43):
        plt.subplot(11, 4, n+1)
        train.select_dtypes(include = ['object']).iloc[:,n].value_counts().plot(kind = 'bar')
        plt.xlabel(train.select_dtypes(include = ['object']).iloc[:,n].name)
plt.figure(figsize = (25, 60))
cat_barplot()
plt.show()
```


![png](output_26_0.png)



```python
def num_densityplot():
    for n in range(0, 38):
        plt.subplot(10, 4, n+1)
        train.select_dtypes(exclude = ['object']).iloc[:,n].plot.kde()
        plt.xlabel(train.select_dtypes(exclude = ['object']).iloc[:,n].name)
plt.figure(figsize = (25, 60))
num_densityplot()
plt.show()
```


![png](output_27_0.png)



```python
train.boxplot("SalePrice", "Neighborhood", figsize=(15, 8), rot = 90)
plt.show()
```


![png](output_28_0.png)



```python
#train.select_dtypes(exclude = ['object']).corr()
```


```python
plt.figure(figsize = (12,10))
sns.heatmap(train.select_dtypes(exclude = ['object']).corr())
plt.show()
```


![png](output_30_0.png)



```python
train.select_dtypes(exclude = ['object']).corr()['SalePrice'].sort_values( axis = 0, ascending = False)[1:9]
```




    OverallQual     0.790982
    GrLivArea       0.708624
    GarageCars      0.640409
    GarageArea      0.623431
    TotalBsmtSF     0.613581
    1stFlrSF        0.605852
    FullBath        0.560664
    TotRmsAbvGrd    0.533723
    Name: SalePrice, dtype: float64




```python
train.select_dtypes(exclude = ['object']).corr()['SalePrice'].sort_values( axis = 0, ascending = True)[0:4]
```




    KitchenAbvGr    -0.135907
    EnclosedPorch   -0.128578
    MSSubClass      -0.084284
    OverallCond     -0.077856
    Name: SalePrice, dtype: float64




```python
#list(pd.DataFrame(train.select_dtypes(exclude = ['object']).corr()['SalePrice'].sort_values( axis = 0, ascending = True)[0:4]).index)
```


```python
list = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','KitchenAbvGr', 'EnclosedPorch', 'MSSubClass', 'OverallCond']
```


```python
sns.pairplot(train, x_vars = list[0: 4], y_vars = ['SalePrice'], kind = 'reg')
sns.pairplot(train, x_vars = list[4: 8], y_vars = ['SalePrice'], kind = 'reg')
sns.pairplot(train, x_vars = list[8: 12], y_vars = ['SalePrice'], kind = 'reg')
plt.show()
```


![png](output_35_0.png)



![png](output_35_1.png)



![png](output_35_2.png)



```python
(train.select_dtypes(include = ['object']).columns)[1]
```




    'Street'




```python
def cat_Sales():
    for n in range(0, 43):
        plt.subplot(11, 4, n+1)
        sns.boxplot(y = train["SalePrice"], x = train.select_dtypes(include = ['object']).iloc[:,n])
        plt.xticks(rotation = 90)
        #plt.show()
```


```python
plt.figure(figsize=(20, 60))
cat_Sales()
plt.show()
```


![png](output_38_0.png)



```python
train_num = train.select_dtypes(exclude = ['object'])
train_cat = train.select_dtypes(include = ['object'])

test_num = test.select_dtypes(exclude = ['object'])
test_cat = test.select_dtypes(include = ['object'])
```


```python
train_num = train_num.fillna(0)
test_num = test_num.fillna(0)
```


```python
train_cat = train_cat.fillna('Unavailable')
test_cat = test_cat.fillna('Unavailable')
```


```python
train_cat1 = pd.get_dummies(train_cat)
test_cat1 = pd.get_dummies(test_cat)
```


```python
train_cat1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSZoning_C (all)</th>
      <th>MSZoning_FV</th>
      <th>MSZoning_RH</th>
      <th>MSZoning_RL</th>
      <th>MSZoning_RM</th>
      <th>Street_Grvl</th>
      <th>Street_Pave</th>
      <th>Alley_Grvl</th>
      <th>Alley_Pave</th>
      <th>Alley_Unavailable</th>
      <th>...</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 268 columns</p>
</div>




```python
from sklearn.preprocessing import StandardScaler
train_num.iloc[:, :-1] = StandardScaler().fit_transform(train_num.iloc[:, :-1])
test_num.iloc[:, :] = StandardScaler().fit_transform(test_num.iloc[:, :])
```

    /home/m/anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /home/m/anaconda3/lib/python3.6/site-packages/sklearn/base.py:462: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, **fit_params).transform(X)
    /home/m/anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /home/m/anaconda3/lib/python3.6/site-packages/sklearn/base.py:462: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, **fit_params).transform(X)



```python
train_num.SalePrice = np.log(train_num.SalePrice)
```


```python
train_X = pd.concat([train_num, train_cat1], axis = 1)
test_X = pd.concat([test_num, test_cat1], axis = 1)
```


```python
X = train_X.drop('SalePrice', axis = 1)
y = train_X.SalePrice
```


```python
X = X[pd.DataFrame([set(X.columns).intersection(set(test_X.columns))]).T[0]]
test_X = test_X[pd.DataFrame([set(test_X.columns).intersection(set(X.columns))]).T[0]]
```


```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 1)
```


```python
y.hist(bins = 50)
plt.show()
```


![png](output_50_0.png)

We are now going to use OLS (ordinary least squares)

```python
simple_model = sm.OLS(y_train, X_train)
simple_results = simple_model.fit()
simple_results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>SalePrice</td>    <th>  R-squared:         </th> <td>   0.942</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.926</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   61.21</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 04 Jan 2019</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>13:43:26</td>     <th>  Log-Likelihood:    </th> <td>  1087.6</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1168</td>      <th>  AIC:               </th> <td>  -1687.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   924</td>      <th>  BIC:               </th> <td>  -451.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>   243</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
              <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>BsmtFinType1_ALQ</th>         <td>    0.1457</td> <td>    0.017</td> <td>    8.684</td> <td> 0.000</td> <td>    0.113</td> <td>    0.179</td>
</tr>
<tr>
  <th>Fence_MnPrv</th>              <td>    0.1936</td> <td>    0.020</td> <td>    9.790</td> <td> 0.000</td> <td>    0.155</td> <td>    0.232</td>
</tr>
<tr>
  <th>ExterQual_Fa</th>             <td>    0.1971</td> <td>    0.054</td> <td>    3.649</td> <td> 0.000</td> <td>    0.091</td> <td>    0.303</td>
</tr>
<tr>
  <th>LotConfig_FR3</th>            <td>    0.1557</td> <td>    0.055</td> <td>    2.841</td> <td> 0.005</td> <td>    0.048</td> <td>    0.263</td>
</tr>
<tr>
  <th>FireplaceQu_TA</th>           <td>    0.1563</td> <td>    0.017</td> <td>    9.088</td> <td> 0.000</td> <td>    0.123</td> <td>    0.190</td>
</tr>
<tr>
  <th>PoolQC_Unavailable</th>       <td>    3.9217</td> <td>    0.415</td> <td>    9.461</td> <td> 0.000</td> <td>    3.108</td> <td>    4.735</td>
</tr>
<tr>
  <th>BsmtCond_Fa</th>              <td>    0.1677</td> <td>    0.056</td> <td>    3.001</td> <td> 0.003</td> <td>    0.058</td> <td>    0.277</td>
</tr>
<tr>
  <th>LotConfig_Inside</th>         <td>    0.1929</td> <td>    0.021</td> <td>    9.401</td> <td> 0.000</td> <td>    0.153</td> <td>    0.233</td>
</tr>
<tr>
  <th>YrSold</th>                   <td>   -0.0021</td> <td>    0.004</td> <td>   -0.575</td> <td> 0.566</td> <td>   -0.009</td> <td>    0.005</td>
</tr>
<tr>
  <th>WoodDeckSF</th>               <td>    0.0114</td> <td>    0.004</td> <td>    2.863</td> <td> 0.004</td> <td>    0.004</td> <td>    0.019</td>
</tr>
<tr>
  <th>Neighborhood_NoRidge</th>     <td>    0.1061</td> <td>    0.027</td> <td>    3.975</td> <td> 0.000</td> <td>    0.054</td> <td>    0.159</td>
</tr>
<tr>
  <th>HouseStyle_1.5Unf</th>        <td>    0.1295</td> <td>    0.101</td> <td>    1.279</td> <td> 0.201</td> <td>   -0.069</td> <td>    0.328</td>
</tr>
<tr>
  <th>Condition1_PosN</th>          <td>    0.1680</td> <td>    0.034</td> <td>    4.958</td> <td> 0.000</td> <td>    0.102</td> <td>    0.235</td>
</tr>
<tr>
  <th>Exterior2nd_VinylSd</th>      <td>    0.1165</td> <td>    0.114</td> <td>    1.018</td> <td> 0.309</td> <td>   -0.108</td> <td>    0.341</td>
</tr>
<tr>
  <th>HouseStyle_1.5Fin</th>        <td>    0.1198</td> <td>    0.086</td> <td>    1.389</td> <td> 0.165</td> <td>   -0.049</td> <td>    0.289</td>
</tr>
<tr>
  <th>SaleCondition_Abnorml</th>    <td>    0.1401</td> <td>    0.033</td> <td>    4.186</td> <td> 0.000</td> <td>    0.074</td> <td>    0.206</td>
</tr>
<tr>
  <th>LandSlope_Sev</th>            <td>    0.2350</td> <td>    0.049</td> <td>    4.829</td> <td> 0.000</td> <td>    0.139</td> <td>    0.330</td>
</tr>
<tr>
  <th>MiscFeature_Othr</th>         <td>    0.5050</td> <td>    0.129</td> <td>    3.926</td> <td> 0.000</td> <td>    0.253</td> <td>    0.757</td>
</tr>
<tr>
  <th>SaleType_ConLw</th>           <td>    0.0319</td> <td>    0.065</td> <td>    0.490</td> <td> 0.624</td> <td>   -0.096</td> <td>    0.160</td>
</tr>
<tr>
  <th>LandContour_Bnk</th>          <td>    0.2160</td> <td>    0.024</td> <td>    8.938</td> <td> 0.000</td> <td>    0.169</td> <td>    0.263</td>
</tr>
<tr>
  <th>Neighborhood_CollgCr</th>     <td>    0.0390</td> <td>    0.016</td> <td>    2.410</td> <td> 0.016</td> <td>    0.007</td> <td>    0.071</td>
</tr>
<tr>
  <th>Electrical_SBrkr</th>         <td>   -0.0464</td> <td>    0.115</td> <td>   -0.403</td> <td> 0.687</td> <td>   -0.272</td> <td>    0.180</td>
</tr>
<tr>
  <th>BsmtQual_Ex</th>              <td>    0.2205</td> <td>    0.027</td> <td>    8.134</td> <td> 0.000</td> <td>    0.167</td> <td>    0.274</td>
</tr>
<tr>
  <th>MSSubClass</th>               <td>   -0.0263</td> <td>    0.020</td> <td>   -1.343</td> <td> 0.179</td> <td>   -0.065</td> <td>    0.012</td>
</tr>
<tr>
  <th>Exterior2nd_Stone</th>        <td>    0.0070</td> <td>    0.138</td> <td>    0.050</td> <td> 0.960</td> <td>   -0.264</td> <td>    0.278</td>
</tr>
<tr>
  <th>MiscFeature_Shed</th>         <td>    0.4949</td> <td>    0.113</td> <td>    4.388</td> <td> 0.000</td> <td>    0.274</td> <td>    0.716</td>
</tr>
<tr>
  <th>Neighborhood_Veenker</th>     <td>    0.0598</td> <td>    0.042</td> <td>    1.437</td> <td> 0.151</td> <td>   -0.022</td> <td>    0.142</td>
</tr>
<tr>
  <th>BldgType_1Fam</th>            <td>    0.1822</td> <td>    0.042</td> <td>    4.363</td> <td> 0.000</td> <td>    0.100</td> <td>    0.264</td>
</tr>
<tr>
  <th>GarageQual_Po</th>            <td>   -0.5819</td> <td>    0.202</td> <td>   -2.877</td> <td> 0.004</td> <td>   -0.979</td> <td>   -0.185</td>
</tr>
<tr>
  <th>HalfBath</th>                 <td>    0.0117</td> <td>    0.006</td> <td>    2.051</td> <td> 0.041</td> <td>    0.001</td> <td>    0.023</td>
</tr>
<tr>
  <th>MSZoning_RH</th>              <td>    0.3075</td> <td>    0.035</td> <td>    8.757</td> <td> 0.000</td> <td>    0.239</td> <td>    0.376</td>
</tr>
<tr>
  <th>Fence_Unavailable</th>        <td>    0.2066</td> <td>    0.019</td> <td>   11.140</td> <td> 0.000</td> <td>    0.170</td> <td>    0.243</td>
</tr>
<tr>
  <th>Condition1_Artery</th>        <td>    0.0490</td> <td>    0.026</td> <td>    1.900</td> <td> 0.058</td> <td>   -0.002</td> <td>    0.100</td>
</tr>
<tr>
  <th>Functional_Maj2</th>          <td>   -0.0679</td> <td>    0.059</td> <td>   -1.146</td> <td> 0.252</td> <td>   -0.184</td> <td>    0.048</td>
</tr>
<tr>
  <th>LotShape_IR3</th>             <td>    0.2226</td> <td>    0.044</td> <td>    5.008</td> <td> 0.000</td> <td>    0.135</td> <td>    0.310</td>
</tr>
<tr>
  <th>Foundation_PConc</th>         <td>    0.2013</td> <td>    0.022</td> <td>    9.013</td> <td> 0.000</td> <td>    0.157</td> <td>    0.245</td>
</tr>
<tr>
  <th>CentralAir_N</th>             <td>    0.4600</td> <td>    0.038</td> <td>   12.045</td> <td> 0.000</td> <td>    0.385</td> <td>    0.535</td>
</tr>
<tr>
  <th>CentralAir_Y</th>             <td>    0.5240</td> <td>    0.039</td> <td>   13.376</td> <td> 0.000</td> <td>    0.447</td> <td>    0.601</td>
</tr>
<tr>
  <th>Functional_Min1</th>          <td>    0.2236</td> <td>    0.034</td> <td>    6.641</td> <td> 0.000</td> <td>    0.158</td> <td>    0.290</td>
</tr>
<tr>
  <th>Exterior2nd_CmentBd</th>      <td>    0.2560</td> <td>    0.143</td> <td>    1.787</td> <td> 0.074</td> <td>   -0.025</td> <td>    0.537</td>
</tr>
<tr>
  <th>2ndFlrSF</th>                 <td>    0.0359</td> <td>    0.009</td> <td>    3.817</td> <td> 0.000</td> <td>    0.017</td> <td>    0.054</td>
</tr>
<tr>
  <th>BsmtFinType1_Rec</th>         <td>    0.1365</td> <td>    0.018</td> <td>    7.608</td> <td> 0.000</td> <td>    0.101</td> <td>    0.172</td>
</tr>
<tr>
  <th>KitchenQual_Ex</th>           <td>    0.2929</td> <td>    0.025</td> <td>   11.857</td> <td> 0.000</td> <td>    0.244</td> <td>    0.341</td>
</tr>
<tr>
  <th>LandContour_Low</th>          <td>    0.2245</td> <td>    0.030</td> <td>    7.519</td> <td> 0.000</td> <td>    0.166</td> <td>    0.283</td>
</tr>
<tr>
  <th>FireplaceQu_Fa</th>           <td>    0.1392</td> <td>    0.024</td> <td>    5.760</td> <td> 0.000</td> <td>    0.092</td> <td>    0.187</td>
</tr>
<tr>
  <th>Condition1_RRNe</th>          <td>    0.0768</td> <td>    0.074</td> <td>    1.043</td> <td> 0.297</td> <td>   -0.068</td> <td>    0.221</td>
</tr>
<tr>
  <th>BldgType_2fmCon</th>          <td>    0.2750</td> <td>    0.041</td> <td>    6.628</td> <td> 0.000</td> <td>    0.194</td> <td>    0.356</td>
</tr>
<tr>
  <th>Alley_Grvl</th>               <td>    0.3317</td> <td>    0.030</td> <td>   11.020</td> <td> 0.000</td> <td>    0.273</td> <td>    0.391</td>
</tr>
<tr>
  <th>GarageFinish_Fin</th>         <td>    0.3478</td> <td>    0.053</td> <td>    6.554</td> <td> 0.000</td> <td>    0.244</td> <td>    0.452</td>
</tr>
<tr>
  <th>LotConfig_CulDSac</th>        <td>    0.2395</td> <td>    0.024</td> <td>   10.063</td> <td> 0.000</td> <td>    0.193</td> <td>    0.286</td>
</tr>
<tr>
  <th>LowQualFinSF</th>             <td>    0.0049</td> <td>    0.006</td> <td>    0.824</td> <td> 0.410</td> <td>   -0.007</td> <td>    0.017</td>
</tr>
<tr>
  <th>Heating_GasW</th>             <td>    0.1565</td> <td>    0.084</td> <td>    1.873</td> <td> 0.061</td> <td>   -0.007</td> <td>    0.320</td>
</tr>
<tr>
  <th>TotalBsmtSF</th>              <td>    0.0183</td> <td>    0.008</td> <td>    2.388</td> <td> 0.017</td> <td>    0.003</td> <td>    0.033</td>
</tr>
<tr>
  <th>GarageType_Unavailable</th>   <td>   -0.0331</td> <td>    0.148</td> <td>   -0.225</td> <td> 0.822</td> <td>   -0.323</td> <td>    0.257</td>
</tr>
<tr>
  <th>Exterior1st_Plywood</th>      <td>   -0.0232</td> <td>    0.086</td> <td>   -0.272</td> <td> 0.786</td> <td>   -0.191</td> <td>    0.145</td>
</tr>
<tr>
  <th>GarageCond_Fa</th>            <td>    0.2456</td> <td>    0.056</td> <td>    4.395</td> <td> 0.000</td> <td>    0.136</td> <td>    0.355</td>
</tr>
<tr>
  <th>FireplaceQu_Po</th>           <td>    0.2373</td> <td>    0.029</td> <td>    8.136</td> <td> 0.000</td> <td>    0.180</td> <td>    0.295</td>
</tr>
<tr>
  <th>Heating_Wall</th>             <td>    0.1852</td> <td>    0.105</td> <td>    1.770</td> <td> 0.077</td> <td>   -0.020</td> <td>    0.390</td>
</tr>
<tr>
  <th>Fence_MnWw</th>               <td>    0.2007</td> <td>    0.038</td> <td>    5.253</td> <td> 0.000</td> <td>    0.126</td> <td>    0.276</td>
</tr>
<tr>
  <th>RoofStyle_Shed</th>           <td>    0.0075</td> <td>    0.236</td> <td>    0.032</td> <td> 0.975</td> <td>   -0.456</td> <td>    0.471</td>
</tr>
<tr>
  <th>Exterior1st_HdBoard</th>      <td>   -0.0663</td> <td>    0.084</td> <td>   -0.791</td> <td> 0.429</td> <td>   -0.231</td> <td>    0.098</td>
</tr>
<tr>
  <th>Condition2_Artery</th>        <td>    0.0072</td> <td>    0.125</td> <td>    0.057</td> <td> 0.954</td> <td>   -0.239</td> <td>    0.253</td>
</tr>
<tr>
  <th>LotShape_IR2</th>             <td>    0.2591</td> <td>    0.028</td> <td>    9.194</td> <td> 0.000</td> <td>    0.204</td> <td>    0.314</td>
</tr>
<tr>
  <th>GarageCars</th>               <td>    0.0137</td> <td>    0.010</td> <td>    1.434</td> <td> 0.152</td> <td>   -0.005</td> <td>    0.033</td>
</tr>
<tr>
  <th>HeatingQC_Gd</th>             <td>    0.2485</td> <td>    0.021</td> <td>   11.836</td> <td> 0.000</td> <td>    0.207</td> <td>    0.290</td>
</tr>
<tr>
  <th>BsmtQual_TA</th>              <td>    0.2011</td> <td>    0.023</td> <td>    8.923</td> <td> 0.000</td> <td>    0.157</td> <td>    0.245</td>
</tr>
<tr>
  <th>TotRmsAbvGrd</th>             <td>    0.0088</td> <td>    0.009</td> <td>    1.032</td> <td> 0.303</td> <td>   -0.008</td> <td>    0.026</td>
</tr>
<tr>
  <th>Exterior1st_BrkComm</th>      <td>   -0.4272</td> <td>    0.168</td> <td>   -2.543</td> <td> 0.011</td> <td>   -0.757</td> <td>   -0.098</td>
</tr>
<tr>
  <th>BsmtFinType1_Unavailable</th> <td>    0.1825</td> <td>    0.051</td> <td>    3.571</td> <td> 0.000</td> <td>    0.082</td> <td>    0.283</td>
</tr>
<tr>
  <th>Heating_Grav</th>             <td>   -0.1551</td> <td>    0.097</td> <td>   -1.594</td> <td> 0.111</td> <td>   -0.346</td> <td>    0.036</td>
</tr>
<tr>
  <th>BsmtExposure_Av</th>          <td>    0.1967</td> <td>    0.028</td> <td>    7.033</td> <td> 0.000</td> <td>    0.142</td> <td>    0.252</td>
</tr>
<tr>
  <th>Exterior1st_VinylSd</th>      <td>   -0.0335</td> <td>    0.090</td> <td>   -0.372</td> <td> 0.710</td> <td>   -0.211</td> <td>    0.144</td>
</tr>
<tr>
  <th>MSZoning_C (all)</th>         <td>   -0.1571</td> <td>    0.047</td> <td>   -3.334</td> <td> 0.001</td> <td>   -0.250</td> <td>   -0.065</td>
</tr>
<tr>
  <th>LandSlope_Mod</th>            <td>    0.3935</td> <td>    0.033</td> <td>   11.775</td> <td> 0.000</td> <td>    0.328</td> <td>    0.459</td>
</tr>
<tr>
  <th>SaleCondition_Normal</th>     <td>    0.2075</td> <td>    0.031</td> <td>    6.618</td> <td> 0.000</td> <td>    0.146</td> <td>    0.269</td>
</tr>
<tr>
  <th>GarageType_CarPort</th>       <td>    0.2095</td> <td>    0.056</td> <td>    3.773</td> <td> 0.000</td> <td>    0.101</td> <td>    0.318</td>
</tr>
<tr>
  <th>HeatingQC_Fa</th>             <td>    0.2332</td> <td>    0.026</td> <td>    9.124</td> <td> 0.000</td> <td>    0.183</td> <td>    0.283</td>
</tr>
<tr>
  <th>GarageCond_Unavailable</th>   <td>   -0.0331</td> <td>    0.148</td> <td>   -0.225</td> <td> 0.822</td> <td>   -0.323</td> <td>    0.257</td>
</tr>
<tr>
  <th>Neighborhood_Gilbert</th>     <td>    0.0373</td> <td>    0.021</td> <td>    1.782</td> <td> 0.075</td> <td>   -0.004</td> <td>    0.078</td>
</tr>
<tr>
  <th>Neighborhood_SWISU</th>       <td>    0.0364</td> <td>    0.032</td> <td>    1.144</td> <td> 0.253</td> <td>   -0.026</td> <td>    0.099</td>
</tr>
<tr>
  <th>Neighborhood_StoneBr</th>     <td>    0.2155</td> <td>    0.030</td> <td>    7.285</td> <td> 0.000</td> <td>    0.157</td> <td>    0.274</td>
</tr>
<tr>
  <th>KitchenQual_Gd</th>           <td>    0.2305</td> <td>    0.021</td> <td>   10.840</td> <td> 0.000</td> <td>    0.189</td> <td>    0.272</td>
</tr>
<tr>
  <th>MasVnrType_BrkCmn</th>        <td>    0.1818</td> <td>    0.032</td> <td>    5.620</td> <td> 0.000</td> <td>    0.118</td> <td>    0.245</td>
</tr>
<tr>
  <th>Neighborhood_IDOTRR</th>      <td>   -0.0111</td> <td>    0.036</td> <td>   -0.310</td> <td> 0.756</td> <td>   -0.082</td> <td>    0.059</td>
</tr>
<tr>
  <th>Neighborhood_Sawyer</th>      <td>    0.0035</td> <td>    0.020</td> <td>    0.180</td> <td> 0.857</td> <td>   -0.035</td> <td>    0.042</td>
</tr>
<tr>
  <th>Exterior2nd_BrkFace</th>      <td>    0.0974</td> <td>    0.127</td> <td>    0.765</td> <td> 0.444</td> <td>   -0.152</td> <td>    0.347</td>
</tr>
<tr>
  <th>OverallCond</th>              <td>    0.0398</td> <td>    0.005</td> <td>    7.368</td> <td> 0.000</td> <td>    0.029</td> <td>    0.050</td>
</tr>
<tr>
  <th>ExterQual_Ex</th>             <td>    0.2687</td> <td>    0.032</td> <td>    8.279</td> <td> 0.000</td> <td>    0.205</td> <td>    0.332</td>
</tr>
<tr>
  <th>Foundation_Slab</th>          <td>    0.1573</td> <td>    0.047</td> <td>    3.369</td> <td> 0.001</td> <td>    0.066</td> <td>    0.249</td>
</tr>
<tr>
  <th>LandContour_Lvl</th>          <td>    0.2698</td> <td>    0.023</td> <td>   11.785</td> <td> 0.000</td> <td>    0.225</td> <td>    0.315</td>
</tr>
<tr>
  <th>Functional_Typ</th>           <td>    0.2457</td> <td>    0.027</td> <td>    9.013</td> <td> 0.000</td> <td>    0.192</td> <td>    0.299</td>
</tr>
<tr>
  <th>SaleType_WD</th>              <td>    0.0365</td> <td>    0.024</td> <td>    1.538</td> <td> 0.124</td> <td>   -0.010</td> <td>    0.083</td>
</tr>
<tr>
  <th>GarageType_Attchd</th>        <td>    0.1885</td> <td>    0.031</td> <td>    6.103</td> <td> 0.000</td> <td>    0.128</td> <td>    0.249</td>
</tr>
<tr>
  <th>GarageCond_Ex</th>            <td>   -0.1069</td> <td>    0.153</td> <td>   -0.696</td> <td> 0.486</td> <td>   -0.408</td> <td>    0.194</td>
</tr>
<tr>
  <th>Exterior1st_AsphShn</th>      <td> 6.833e-15</td> <td> 9.99e-16</td> <td>    6.838</td> <td> 0.000</td> <td> 4.87e-15</td> <td> 8.79e-15</td>
</tr>
<tr>
  <th>ExterCond_Gd</th>             <td>    0.1893</td> <td>    0.036</td> <td>    5.230</td> <td> 0.000</td> <td>    0.118</td> <td>    0.260</td>
</tr>
<tr>
  <th>ScreenPorch</th>              <td>    0.0150</td> <td>    0.004</td> <td>    4.076</td> <td> 0.000</td> <td>    0.008</td> <td>    0.022</td>
</tr>
<tr>
  <th>BsmtCond_Po</th>              <td>    0.2429</td> <td>    0.163</td> <td>    1.491</td> <td> 0.136</td> <td>   -0.077</td> <td>    0.563</td>
</tr>
<tr>
  <th>KitchenQual_TA</th>           <td>    0.2290</td> <td>    0.021</td> <td>   10.818</td> <td> 0.000</td> <td>    0.187</td> <td>    0.271</td>
</tr>
<tr>
  <th>BsmtQual_Gd</th>              <td>    0.1947</td> <td>    0.023</td> <td>    8.505</td> <td> 0.000</td> <td>    0.150</td> <td>    0.240</td>
</tr>
<tr>
  <th>HouseStyle_2.5Unf</th>        <td>    0.2205</td> <td>    0.099</td> <td>    2.237</td> <td> 0.026</td> <td>    0.027</td> <td>    0.414</td>
</tr>
<tr>
  <th>RoofMatl_WdShngl</th>         <td>    0.1991</td> <td>    0.110</td> <td>    1.809</td> <td> 0.071</td> <td>   -0.017</td> <td>    0.415</td>
</tr>
<tr>
  <th>Exterior2nd_ImStucc</th>      <td>    0.1029</td> <td>    0.128</td> <td>    0.803</td> <td> 0.422</td> <td>   -0.149</td> <td>    0.354</td>
</tr>
<tr>
  <th>MSZoning_FV</th>              <td>    0.2925</td> <td>    0.035</td> <td>    8.467</td> <td> 0.000</td> <td>    0.225</td> <td>    0.360</td>
</tr>
<tr>
  <th>Neighborhood_MeadowV</th>     <td>   -0.1465</td> <td>    0.045</td> <td>   -3.292</td> <td> 0.001</td> <td>   -0.234</td> <td>   -0.059</td>
</tr>
<tr>
  <th>YearBuilt</th>                <td>    0.0421</td> <td>    0.013</td> <td>    3.230</td> <td> 0.001</td> <td>    0.017</td> <td>    0.068</td>
</tr>
<tr>
  <th>MSZoning_RM</th>              <td>    0.2506</td> <td>    0.024</td> <td>   10.513</td> <td> 0.000</td> <td>    0.204</td> <td>    0.297</td>
</tr>
<tr>
  <th>GarageArea</th>               <td>    0.0340</td> <td>    0.010</td> <td>    3.547</td> <td> 0.000</td> <td>    0.015</td> <td>    0.053</td>
</tr>
<tr>
  <th>Neighborhood_Timber</th>      <td>    0.0687</td> <td>    0.025</td> <td>    2.756</td> <td> 0.006</td> <td>    0.020</td> <td>    0.118</td>
</tr>
<tr>
  <th>BldgType_Twnhs</th>           <td>    0.1489</td> <td>    0.032</td> <td>    4.715</td> <td> 0.000</td> <td>    0.087</td> <td>    0.211</td>
</tr>
<tr>
  <th>BldgType_TwnhsE</th>          <td>    0.1717</td> <td>    0.027</td> <td>    6.467</td> <td> 0.000</td> <td>    0.120</td> <td>    0.224</td>
</tr>
<tr>
  <th>BsmtExposure_Gd</th>          <td>    0.2416</td> <td>    0.029</td> <td>    8.376</td> <td> 0.000</td> <td>    0.185</td> <td>    0.298</td>
</tr>
<tr>
  <th>ExterQual_Gd</th>             <td>    0.2554</td> <td>    0.026</td> <td>    9.668</td> <td> 0.000</td> <td>    0.204</td> <td>    0.307</td>
</tr>
<tr>
  <th>GarageQual_TA</th>            <td>   -0.4045</td> <td>    0.153</td> <td>   -2.649</td> <td> 0.008</td> <td>   -0.704</td> <td>   -0.105</td>
</tr>
<tr>
  <th>BsmtFinSF1</th>               <td>    0.0142</td> <td>    0.005</td> <td>    2.663</td> <td> 0.008</td> <td>    0.004</td> <td>    0.025</td>
</tr>
<tr>
  <th>Condition2_Feedr</th>         <td>    0.0068</td> <td>    0.100</td> <td>    0.068</td> <td> 0.946</td> <td>   -0.190</td> <td>    0.203</td>
</tr>
<tr>
  <th>Neighborhood_Somerst</th>     <td>    0.0774</td> <td>    0.031</td> <td>    2.510</td> <td> 0.012</td> <td>    0.017</td> <td>    0.138</td>
</tr>
<tr>
  <th>PavedDrive_Y</th>             <td>    0.3453</td> <td>    0.027</td> <td>   12.795</td> <td> 0.000</td> <td>    0.292</td> <td>    0.398</td>
</tr>
<tr>
  <th>SaleCondition_AdjLand</th>    <td>    0.2479</td> <td>    0.111</td> <td>    2.238</td> <td> 0.025</td> <td>    0.030</td> <td>    0.465</td>
</tr>
<tr>
  <th>RoofMatl_CompShg</th>         <td>    0.1654</td> <td>    0.093</td> <td>    1.776</td> <td> 0.076</td> <td>   -0.017</td> <td>    0.348</td>
</tr>
<tr>
  <th>Foundation_Stone</th>         <td>    0.2428</td> <td>    0.051</td> <td>    4.759</td> <td> 0.000</td> <td>    0.143</td> <td>    0.343</td>
</tr>
<tr>
  <th>LandSlope_Gtl</th>            <td>    0.3555</td> <td>    0.034</td> <td>   10.333</td> <td> 0.000</td> <td>    0.288</td> <td>    0.423</td>
</tr>
<tr>
  <th>Neighborhood_BrDale</th>      <td>   -0.0080</td> <td>    0.045</td> <td>   -0.178</td> <td> 0.859</td> <td>   -0.096</td> <td>    0.080</td>
</tr>
<tr>
  <th>MSZoning_RL</th>              <td>    0.2905</td> <td>    0.023</td> <td>   12.589</td> <td> 0.000</td> <td>    0.245</td> <td>    0.336</td>
</tr>
<tr>
  <th>RoofMatl_Tar&Grv</th>         <td>   -0.0319</td> <td>    0.106</td> <td>   -0.302</td> <td> 0.762</td> <td>   -0.239</td> <td>    0.175</td>
</tr>
<tr>
  <th>ExterCond_Ex</th>             <td>    0.2488</td> <td>    0.074</td> <td>    3.370</td> <td> 0.001</td> <td>    0.104</td> <td>    0.394</td>
</tr>
<tr>
  <th>Condition1_Norm</th>          <td>    0.1466</td> <td>    0.017</td> <td>    8.494</td> <td> 0.000</td> <td>    0.113</td> <td>    0.180</td>
</tr>
<tr>
  <th>SaleType_COD</th>             <td>    0.0605</td> <td>    0.030</td> <td>    1.998</td> <td> 0.046</td> <td>    0.001</td> <td>    0.120</td>
</tr>
<tr>
  <th>MoSold</th>                   <td>    0.0011</td> <td>    0.004</td> <td>    0.289</td> <td> 0.773</td> <td>   -0.006</td> <td>    0.008</td>
</tr>
<tr>
  <th>Exterior2nd_AsphShn</th>      <td>    0.1120</td> <td>    0.152</td> <td>    0.734</td> <td> 0.463</td> <td>   -0.187</td> <td>    0.411</td>
</tr>
<tr>
  <th>Exterior1st_WdShing</th>      <td>   -0.0560</td> <td>    0.089</td> <td>   -0.633</td> <td> 0.527</td> <td>   -0.230</td> <td>    0.118</td>
</tr>
<tr>
  <th>SaleCondition_Alloca</th>     <td>    0.1806</td> <td>    0.049</td> <td>    3.683</td> <td> 0.000</td> <td>    0.084</td> <td>    0.277</td>
</tr>
<tr>
  <th>BsmtCond_Gd</th>              <td>    0.2018</td> <td>    0.056</td> <td>    3.634</td> <td> 0.000</td> <td>    0.093</td> <td>    0.311</td>
</tr>
<tr>
  <th>PavedDrive_N</th>             <td>    0.3280</td> <td>    0.028</td> <td>   11.549</td> <td> 0.000</td> <td>    0.272</td> <td>    0.384</td>
</tr>
<tr>
  <th>Exterior1st_MetalSd</th>      <td>   -0.0281</td> <td>    0.099</td> <td>   -0.285</td> <td> 0.776</td> <td>   -0.222</td> <td>    0.165</td>
</tr>
<tr>
  <th>BedroomAbvGr</th>             <td>    0.0130</td> <td>    0.006</td> <td>    2.057</td> <td> 0.040</td> <td>    0.001</td> <td>    0.025</td>
</tr>
<tr>
  <th>SaleType_ConLI</th>           <td>   -0.0257</td> <td>    0.057</td> <td>   -0.453</td> <td> 0.651</td> <td>   -0.137</td> <td>    0.086</td>
</tr>
<tr>
  <th>Condition2_Norm</th>          <td>    0.0144</td> <td>    0.071</td> <td>    0.202</td> <td> 0.840</td> <td>   -0.126</td> <td>    0.154</td>
</tr>
<tr>
  <th>GarageType_2Types</th>        <td>    0.0322</td> <td>    0.057</td> <td>    0.569</td> <td> 0.570</td> <td>   -0.079</td> <td>    0.144</td>
</tr>
<tr>
  <th>Neighborhood_Edwards</th>     <td>   -0.0681</td> <td>    0.018</td> <td>   -3.777</td> <td> 0.000</td> <td>   -0.104</td> <td>   -0.033</td>
</tr>
<tr>
  <th>Exterior2nd_Wd Shng</th>      <td>    0.1226</td> <td>    0.123</td> <td>    0.997</td> <td> 0.319</td> <td>   -0.119</td> <td>    0.364</td>
</tr>
<tr>
  <th>BsmtExposure_Unavailable</th> <td>    0.1664</td> <td>    0.089</td> <td>    1.875</td> <td> 0.061</td> <td>   -0.008</td> <td>    0.340</td>
</tr>
<tr>
  <th>LotConfig_FR2</th>            <td>    0.1885</td> <td>    0.025</td> <td>    7.421</td> <td> 0.000</td> <td>    0.139</td> <td>    0.238</td>
</tr>
<tr>
  <th>PoolQC_Gd</th>                <td>   -1.5171</td> <td>    0.173</td> <td>   -8.758</td> <td> 0.000</td> <td>   -1.857</td> <td>   -1.177</td>
</tr>
<tr>
  <th>BsmtCond_TA</th>              <td>    0.1892</td> <td>    0.054</td> <td>    3.474</td> <td> 0.001</td> <td>    0.082</td> <td>    0.296</td>
</tr>
<tr>
  <th>GarageQual_Gd</th>            <td>   -0.4254</td> <td>    0.159</td> <td>   -2.681</td> <td> 0.007</td> <td>   -0.737</td> <td>   -0.114</td>
</tr>
<tr>
  <th>MasVnrType_None</th>          <td>    0.2058</td> <td>    0.020</td> <td>   10.441</td> <td> 0.000</td> <td>    0.167</td> <td>    0.244</td>
</tr>
<tr>
  <th>HouseStyle_1Story</th>        <td>    0.0930</td> <td>    0.094</td> <td>    0.988</td> <td> 0.324</td> <td>   -0.092</td> <td>    0.278</td>
</tr>
<tr>
  <th>BsmtQual_Fa</th>              <td>    0.1854</td> <td>    0.028</td> <td>    6.512</td> <td> 0.000</td> <td>    0.129</td> <td>    0.241</td>
</tr>
<tr>
  <th>LandContour_HLS</th>          <td>    0.2737</td> <td>    0.027</td> <td>   10.248</td> <td> 0.000</td> <td>    0.221</td> <td>    0.326</td>
</tr>
<tr>
  <th>Heating_GasA</th>             <td>    0.0738</td> <td>    0.076</td> <td>    0.967</td> <td> 0.334</td> <td>   -0.076</td> <td>    0.224</td>
</tr>
<tr>
  <th>HeatingQC_Ex</th>             <td>    0.2667</td> <td>    0.021</td> <td>   12.760</td> <td> 0.000</td> <td>    0.226</td> <td>    0.308</td>
</tr>
<tr>
  <th>BsmtFinType1_BLQ</th>         <td>    0.1394</td> <td>    0.017</td> <td>    8.056</td> <td> 0.000</td> <td>    0.105</td> <td>    0.173</td>
</tr>
<tr>
  <th>HeatingQC_TA</th>             <td>    0.2356</td> <td>    0.021</td> <td>   11.319</td> <td> 0.000</td> <td>    0.195</td> <td>    0.276</td>
</tr>
<tr>
  <th>HouseStyle_2Story</th>        <td>    0.0921</td> <td>    0.086</td> <td>    1.076</td> <td> 0.282</td> <td>   -0.076</td> <td>    0.260</td>
</tr>
<tr>
  <th>BsmtFinType2_LwQ</th>         <td>    0.1239</td> <td>    0.029</td> <td>    4.243</td> <td> 0.000</td> <td>    0.067</td> <td>    0.181</td>
</tr>
<tr>
  <th>Neighborhood_NWAmes</th>      <td>    0.0027</td> <td>    0.019</td> <td>    0.141</td> <td> 0.888</td> <td>   -0.034</td> <td>    0.040</td>
</tr>
<tr>
  <th>Condition1_PosA</th>          <td>    0.1289</td> <td>    0.042</td> <td>    3.063</td> <td> 0.002</td> <td>    0.046</td> <td>    0.212</td>
</tr>
<tr>
  <th>Exterior1st_AsbShng</th>      <td>   -0.0134</td> <td>    0.113</td> <td>   -0.118</td> <td> 0.906</td> <td>   -0.235</td> <td>    0.208</td>
</tr>
<tr>
  <th>GarageType_BuiltIn</th>       <td>    0.1735</td> <td>    0.034</td> <td>    5.065</td> <td> 0.000</td> <td>    0.106</td> <td>    0.241</td>
</tr>
<tr>
  <th>LotShape_Reg</th>             <td>    0.2557</td> <td>    0.026</td> <td>    9.734</td> <td> 0.000</td> <td>    0.204</td> <td>    0.307</td>
</tr>
<tr>
  <th>BsmtUnfSF</th>                <td>    0.0023</td> <td>    0.005</td> <td>    0.466</td> <td> 0.641</td> <td>   -0.007</td> <td>    0.012</td>
</tr>
<tr>
  <th>ExterCond_Fa</th>             <td>    0.1292</td> <td>    0.044</td> <td>    2.932</td> <td> 0.003</td> <td>    0.043</td> <td>    0.216</td>
</tr>
<tr>
  <th>Utilities_AllPub</th>         <td>    0.3895</td> <td>    0.130</td> <td>    3.001</td> <td> 0.003</td> <td>    0.135</td> <td>    0.644</td>
</tr>
<tr>
  <th>Neighborhood_NPkVill</th>     <td>    0.0417</td> <td>    0.060</td> <td>    0.695</td> <td> 0.487</td> <td>   -0.076</td> <td>    0.159</td>
</tr>
<tr>
  <th>BsmtFinType2_Unf</th>         <td>    0.1465</td> <td>    0.028</td> <td>    5.289</td> <td> 0.000</td> <td>    0.092</td> <td>    0.201</td>
</tr>
<tr>
  <th>BsmtFinSF2</th>               <td>    0.0035</td> <td>    0.007</td> <td>    0.505</td> <td> 0.614</td> <td>   -0.010</td> <td>    0.017</td>
</tr>
<tr>
  <th>PoolQC_Ex</th>                <td>   -0.3075</td> <td>    0.162</td> <td>   -1.896</td> <td> 0.058</td> <td>   -0.626</td> <td>    0.011</td>
</tr>
<tr>
  <th>BsmtFinType1_GLQ</th>         <td>    0.1520</td> <td>    0.017</td> <td>    9.103</td> <td> 0.000</td> <td>    0.119</td> <td>    0.185</td>
</tr>
<tr>
  <th>Electrical_FuseP</th>         <td>   -0.0621</td> <td>    0.186</td> <td>   -0.333</td> <td> 0.739</td> <td>   -0.428</td> <td>    0.304</td>
</tr>
<tr>
  <th>GarageFinish_RFn</th>         <td>    0.3480</td> <td>    0.052</td> <td>    6.651</td> <td> 0.000</td> <td>    0.245</td> <td>    0.451</td>
</tr>
<tr>
  <th>Exterior2nd_Plywood</th>      <td>    0.1026</td> <td>    0.123</td> <td>    0.837</td> <td> 0.403</td> <td>   -0.138</td> <td>    0.343</td>
</tr>
<tr>
  <th>RoofStyle_Flat</th>           <td>    0.2797</td> <td>    0.087</td> <td>    3.219</td> <td> 0.001</td> <td>    0.109</td> <td>    0.450</td>
</tr>
<tr>
  <th>Exterior1st_BrkFace</th>      <td>    0.0235</td> <td>    0.085</td> <td>    0.276</td> <td> 0.783</td> <td>   -0.143</td> <td>    0.190</td>
</tr>
<tr>
  <th>MiscFeature_Unavailable</th>  <td>    0.5239</td> <td>    0.129</td> <td>    4.073</td> <td> 0.000</td> <td>    0.271</td> <td>    0.776</td>
</tr>
<tr>
  <th>RoofStyle_Gambrel</th>        <td>    0.1982</td> <td>    0.064</td> <td>    3.101</td> <td> 0.002</td> <td>    0.073</td> <td>    0.324</td>
</tr>
<tr>
  <th>ExterQual_TA</th>             <td>    0.2628</td> <td>    0.026</td> <td>   10.139</td> <td> 0.000</td> <td>    0.212</td> <td>    0.314</td>
</tr>
<tr>
  <th>Foundation_BrkTil</th>        <td>    0.1526</td> <td>    0.024</td> <td>    6.294</td> <td> 0.000</td> <td>    0.105</td> <td>    0.200</td>
</tr>
<tr>
  <th>HeatingQC_Po</th>             <td> 1.904e-16</td> <td> 2.07e-16</td> <td>    0.919</td> <td> 0.358</td> <td>-2.16e-16</td> <td> 5.97e-16</td>
</tr>
<tr>
  <th>PavedDrive_P</th>             <td>    0.3107</td> <td>    0.031</td> <td>   10.100</td> <td> 0.000</td> <td>    0.250</td> <td>    0.371</td>
</tr>
<tr>
  <th>BsmtHalfBath</th>             <td>    0.0035</td> <td>    0.004</td> <td>    0.861</td> <td> 0.390</td> <td>   -0.005</td> <td>    0.012</td>
</tr>
<tr>
  <th>GarageQual_Unavailable</th>   <td>   -0.0331</td> <td>    0.148</td> <td>   -0.225</td> <td> 0.822</td> <td>   -0.323</td> <td>    0.257</td>
</tr>
<tr>
  <th>Exterior2nd_CBlock</th>       <td>    0.0337</td> <td>    0.099</td> <td>    0.339</td> <td> 0.735</td> <td>   -0.161</td> <td>    0.229</td>
</tr>
<tr>
  <th>Exterior2nd_AsbShng</th>      <td>    0.0656</td> <td>    0.143</td> <td>    0.461</td> <td> 0.645</td> <td>   -0.214</td> <td>    0.345</td>
</tr>
<tr>
  <th>Foundation_Wood</th>          <td>    0.0530</td> <td>    0.062</td> <td>    0.856</td> <td> 0.392</td> <td>   -0.069</td> <td>    0.175</td>
</tr>
<tr>
  <th>Exterior2nd_Wd Sdng</th>      <td>    0.1533</td> <td>    0.122</td> <td>    1.253</td> <td> 0.211</td> <td>   -0.087</td> <td>    0.394</td>
</tr>
<tr>
  <th>Fireplaces</th>               <td>    0.0082</td> <td>    0.009</td> <td>    0.922</td> <td> 0.357</td> <td>   -0.009</td> <td>    0.026</td>
</tr>
<tr>
  <th>LotConfig_Corner</th>         <td>    0.2074</td> <td>    0.021</td> <td>    9.688</td> <td> 0.000</td> <td>    0.165</td> <td>    0.249</td>
</tr>
<tr>
  <th>Neighborhood_BrkSide</th>     <td>    0.0234</td> <td>    0.026</td> <td>    0.892</td> <td> 0.372</td> <td>   -0.028</td> <td>    0.075</td>
</tr>
<tr>
  <th>BsmtFinType2_ALQ</th>         <td>    0.1751</td> <td>    0.035</td> <td>    5.044</td> <td> 0.000</td> <td>    0.107</td> <td>    0.243</td>
</tr>
<tr>
  <th>Functional_Sev</th>           <td>   -0.0363</td> <td>    0.124</td> <td>   -0.292</td> <td> 0.770</td> <td>   -0.280</td> <td>    0.207</td>
</tr>
<tr>
  <th>BsmtFinType2_Rec</th>         <td>    0.1441</td> <td>    0.027</td> <td>    5.308</td> <td> 0.000</td> <td>    0.091</td> <td>    0.197</td>
</tr>
<tr>
  <th>Neighborhood_OldTown</th>     <td>   -0.0308</td> <td>    0.028</td> <td>   -1.113</td> <td> 0.266</td> <td>   -0.085</td> <td>    0.024</td>
</tr>
<tr>
  <th>GarageYrBlt</th>              <td>   -0.1015</td> <td>    0.156</td> <td>   -0.650</td> <td> 0.516</td> <td>   -0.408</td> <td>    0.205</td>
</tr>
<tr>
  <th>GarageCond_Po</th>            <td>    0.4717</td> <td>    0.081</td> <td>    5.794</td> <td> 0.000</td> <td>    0.312</td> <td>    0.631</td>
</tr>
<tr>
  <th>GarageType_Basment</th>       <td>    0.2330</td> <td>    0.041</td> <td>    5.737</td> <td> 0.000</td> <td>    0.153</td> <td>    0.313</td>
</tr>
<tr>
  <th>1stFlrSF</th>                 <td>    0.0393</td> <td>    0.009</td> <td>    4.470</td> <td> 0.000</td> <td>    0.022</td> <td>    0.057</td>
</tr>
<tr>
  <th>Neighborhood_Blmngtn</th>     <td>    0.0919</td> <td>    0.039</td> <td>    2.343</td> <td> 0.019</td> <td>    0.015</td> <td>    0.169</td>
</tr>
<tr>
  <th>Neighborhood_Crawfor</th>     <td>    0.1715</td> <td>    0.023</td> <td>    7.338</td> <td> 0.000</td> <td>    0.126</td> <td>    0.217</td>
</tr>
<tr>
  <th>Neighborhood_NridgHt</th>     <td>    0.1613</td> <td>    0.023</td> <td>    6.945</td> <td> 0.000</td> <td>    0.116</td> <td>    0.207</td>
</tr>
<tr>
  <th>Id</th>                       <td>   -0.0034</td> <td>    0.003</td> <td>   -0.966</td> <td> 0.334</td> <td>   -0.010</td> <td>    0.003</td>
</tr>
<tr>
  <th>Neighborhood_ClearCr</th>     <td>    0.1006</td> <td>    0.030</td> <td>    3.304</td> <td> 0.001</td> <td>    0.041</td> <td>    0.160</td>
</tr>
<tr>
  <th>MiscVal</th>                  <td>   -0.0002</td> <td>    0.015</td> <td>   -0.015</td> <td> 0.988</td> <td>   -0.030</td> <td>    0.030</td>
</tr>
<tr>
  <th>Condition1_RRNn</th>          <td>    0.1378</td> <td>    0.060</td> <td>    2.311</td> <td> 0.021</td> <td>    0.021</td> <td>    0.255</td>
</tr>
<tr>
  <th>GarageQual_Fa</th>            <td>   -0.4561</td> <td>    0.154</td> <td>   -2.966</td> <td> 0.003</td> <td>   -0.758</td> <td>   -0.154</td>
</tr>
<tr>
  <th>SaleType_New</th>             <td>    0.2275</td> <td>    0.083</td> <td>    2.754</td> <td> 0.006</td> <td>    0.065</td> <td>    0.390</td>
</tr>
<tr>
  <th>SaleType_Oth</th>             <td>    0.2500</td> <td>    0.077</td> <td>    3.266</td> <td> 0.001</td> <td>    0.100</td> <td>    0.400</td>
</tr>
<tr>
  <th>Neighborhood_NAmes</th>       <td>   -0.0101</td> <td>    0.015</td> <td>   -0.659</td> <td> 0.510</td> <td>   -0.040</td> <td>    0.020</td>
</tr>
<tr>
  <th>Exterior2nd_HdBoard</th>      <td>    0.1264</td> <td>    0.123</td> <td>    1.025</td> <td> 0.306</td> <td>   -0.116</td> <td>    0.368</td>
</tr>
<tr>
  <th>MasVnrArea</th>               <td>    0.0027</td> <td>    0.006</td> <td>    0.472</td> <td> 0.637</td> <td>   -0.009</td> <td>    0.014</td>
</tr>
<tr>
  <th>Condition1_RRAe</th>          <td>    0.0390</td> <td>    0.040</td> <td>    0.973</td> <td> 0.331</td> <td>   -0.040</td> <td>    0.118</td>
</tr>
<tr>
  <th>SaleType_ConLD</th>           <td>    0.1375</td> <td>    0.044</td> <td>    3.160</td> <td> 0.002</td> <td>    0.052</td> <td>    0.223</td>
</tr>
<tr>
  <th>LotShape_IR1</th>             <td>    0.2466</td> <td>    0.026</td> <td>    9.535</td> <td> 0.000</td> <td>    0.196</td> <td>    0.297</td>
</tr>
<tr>
  <th>MiscFeature_Gar2</th>         <td>    0.5733</td> <td>    0.399</td> <td>    1.437</td> <td> 0.151</td> <td>   -0.210</td> <td>    1.356</td>
</tr>
<tr>
  <th>RoofStyle_Gable</th>          <td>    0.1430</td> <td>    0.052</td> <td>    2.734</td> <td> 0.006</td> <td>    0.040</td> <td>    0.246</td>
</tr>
<tr>
  <th>Foundation_CBlock</th>        <td>    0.1770</td> <td>    0.022</td> <td>    8.031</td> <td> 0.000</td> <td>    0.134</td> <td>    0.220</td>
</tr>
<tr>
  <th>EnclosedPorch</th>            <td>    0.0098</td> <td>    0.004</td> <td>    2.328</td> <td> 0.020</td> <td>    0.002</td> <td>    0.018</td>
</tr>
<tr>
  <th>Exterior2nd_MetalSd</th>      <td>    0.1205</td> <td>    0.132</td> <td>    0.914</td> <td> 0.361</td> <td>   -0.138</td> <td>    0.379</td>
</tr>
<tr>
  <th>Fence_GdWo</th>               <td>    0.1660</td> <td>    0.023</td> <td>    7.320</td> <td> 0.000</td> <td>    0.121</td> <td>    0.211</td>
</tr>
<tr>
  <th>BsmtExposure_No</th>          <td>    0.1866</td> <td>    0.027</td> <td>    6.892</td> <td> 0.000</td> <td>    0.133</td> <td>    0.240</td>
</tr>
<tr>
  <th>Condition1_RRAn</th>          <td>    0.1421</td> <td>    0.030</td> <td>    4.758</td> <td> 0.000</td> <td>    0.083</td> <td>    0.201</td>
</tr>
<tr>
  <th>Exterior1st_CBlock</th>       <td>    0.0337</td> <td>    0.099</td> <td>    0.339</td> <td> 0.735</td> <td>   -0.161</td> <td>    0.229</td>
</tr>
<tr>
  <th>Condition2_PosN</th>          <td>   -0.7711</td> <td>    0.117</td> <td>   -6.579</td> <td> 0.000</td> <td>   -1.001</td> <td>   -0.541</td>
</tr>
<tr>
  <th>SaleType_CWD</th>             <td>    0.0741</td> <td>    0.065</td> <td>    1.134</td> <td> 0.257</td> <td>   -0.054</td> <td>    0.202</td>
</tr>
<tr>
  <th>MasVnrType_BrkFace</th>       <td>    0.2032</td> <td>    0.020</td> <td>   10.390</td> <td> 0.000</td> <td>    0.165</td> <td>    0.242</td>
</tr>
<tr>
  <th>FireplaceQu_Unavailable</th>  <td>    0.1383</td> <td>    0.020</td> <td>    7.031</td> <td> 0.000</td> <td>    0.100</td> <td>    0.177</td>
</tr>
<tr>
  <th>GarageFinish_Unavailable</th> <td>   -0.0331</td> <td>    0.148</td> <td>   -0.225</td> <td> 0.822</td> <td>   -0.323</td> <td>    0.257</td>
</tr>
<tr>
  <th>MasVnrType_Stone</th>         <td>    0.2203</td> <td>    0.021</td> <td>   10.311</td> <td> 0.000</td> <td>    0.178</td> <td>    0.262</td>
</tr>
<tr>
  <th>3SsnPorch</th>                <td>    0.0039</td> <td>    0.003</td> <td>    1.168</td> <td> 0.243</td> <td>   -0.003</td> <td>    0.010</td>
</tr>
<tr>
  <th>BsmtExposure_Mn</th>          <td>    0.1928</td> <td>    0.029</td> <td>    6.736</td> <td> 0.000</td> <td>    0.137</td> <td>    0.249</td>
</tr>
<tr>
  <th>Exterior1st_Stucco</th>       <td>   -0.0694</td> <td>    0.095</td> <td>   -0.728</td> <td> 0.467</td> <td>   -0.256</td> <td>    0.118</td>
</tr>
<tr>
  <th>FireplaceQu_Ex</th>           <td>    0.1532</td> <td>    0.029</td> <td>    5.229</td> <td> 0.000</td> <td>    0.096</td> <td>    0.211</td>
</tr>
<tr>
  <th>GarageCond_Gd</th>            <td>    0.1913</td> <td>    0.068</td> <td>    2.823</td> <td> 0.005</td> <td>    0.058</td> <td>    0.324</td>
</tr>
<tr>
  <th>OverallQual</th>              <td>    0.0614</td> <td>    0.008</td> <td>    8.165</td> <td> 0.000</td> <td>    0.047</td> <td>    0.076</td>
</tr>
<tr>
  <th>YearRemodAdd</th>             <td>    0.0155</td> <td>    0.006</td> <td>    2.391</td> <td> 0.017</td> <td>    0.003</td> <td>    0.028</td>
</tr>
<tr>
  <th>Neighborhood_Blueste</th>     <td>   -0.0548</td> <td>    0.109</td> <td>   -0.502</td> <td> 0.616</td> <td>   -0.269</td> <td>    0.159</td>
</tr>
<tr>
  <th>BldgType_Duplex</th>          <td>    0.2062</td> <td>    0.037</td> <td>    5.529</td> <td> 0.000</td> <td>    0.133</td> <td>    0.279</td>
</tr>
<tr>
  <th>BsmtFullBath</th>             <td>    0.0175</td> <td>    0.005</td> <td>    3.221</td> <td> 0.001</td> <td>    0.007</td> <td>    0.028</td>
</tr>
<tr>
  <th>KitchenAbvGr</th>             <td>   -0.0105</td> <td>    0.008</td> <td>   -1.295</td> <td> 0.196</td> <td>   -0.026</td> <td>    0.005</td>
</tr>
<tr>
  <th>Alley_Unavailable</th>        <td>    0.2920</td> <td>    0.027</td> <td>   10.739</td> <td> 0.000</td> <td>    0.239</td> <td>    0.345</td>
</tr>
<tr>
  <th>BsmtCond_Unavailable</th>     <td>    0.1825</td> <td>    0.051</td> <td>    3.571</td> <td> 0.000</td> <td>    0.082</td> <td>    0.283</td>
</tr>
<tr>
  <th>Functional_Mod</th>           <td>    0.1879</td> <td>    0.043</td> <td>    4.393</td> <td> 0.000</td> <td>    0.104</td> <td>    0.272</td>
</tr>
<tr>
  <th>SaleType_Con</th>             <td>    0.1917</td> <td>    0.076</td> <td>    2.519</td> <td> 0.012</td> <td>    0.042</td> <td>    0.341</td>
</tr>
<tr>
  <th>RoofStyle_Hip</th>            <td>    0.1490</td> <td>    0.053</td> <td>    2.805</td> <td> 0.005</td> <td>    0.045</td> <td>    0.253</td>
</tr>
<tr>
  <th>BsmtFinType1_Unf</th>         <td>    0.1157</td> <td>    0.016</td> <td>    7.060</td> <td> 0.000</td> <td>    0.084</td> <td>    0.148</td>
</tr>
<tr>
  <th>FullBath</th>                 <td>    0.0111</td> <td>    0.007</td> <td>    1.643</td> <td> 0.101</td> <td>   -0.002</td> <td>    0.024</td>
</tr>
<tr>
  <th>Street_Pave</th>              <td>    0.4527</td> <td>    0.056</td> <td>    8.100</td> <td> 0.000</td> <td>    0.343</td> <td>    0.562</td>
</tr>
<tr>
  <th>Functional_Min2</th>          <td>    0.2000</td> <td>    0.036</td> <td>    5.565</td> <td> 0.000</td> <td>    0.129</td> <td>    0.271</td>
</tr>
<tr>
  <th>Street_Grvl</th>              <td>    0.5313</td> <td>    0.060</td> <td>    8.869</td> <td> 0.000</td> <td>    0.414</td> <td>    0.649</td>
</tr>
<tr>
  <th>HouseStyle_SFoyer</th>        <td>    0.1197</td> <td>    0.095</td> <td>    1.265</td> <td> 0.206</td> <td>   -0.066</td> <td>    0.305</td>
</tr>
<tr>
  <th>Condition1_Feedr</th>         <td>    0.0958</td> <td>    0.022</td> <td>    4.341</td> <td> 0.000</td> <td>    0.052</td> <td>    0.139</td>
</tr>
<tr>
  <th>FireplaceQu_Gd</th>           <td>    0.1596</td> <td>    0.017</td> <td>    9.417</td> <td> 0.000</td> <td>    0.126</td> <td>    0.193</td>
</tr>
<tr>
  <th>PoolArea</th>                 <td>    0.3352</td> <td>    0.035</td> <td>    9.509</td> <td> 0.000</td> <td>    0.266</td> <td>    0.404</td>
</tr>
<tr>
  <th>Alley_Pave</th>               <td>    0.3603</td> <td>    0.031</td> <td>   11.608</td> <td> 0.000</td> <td>    0.299</td> <td>    0.421</td>
</tr>
<tr>
  <th>GrLivArea</th>                <td>    0.0592</td> <td>    0.007</td> <td>    8.550</td> <td> 0.000</td> <td>    0.046</td> <td>    0.073</td>
</tr>
<tr>
  <th>Exterior1st_CemntBd</th>      <td>   -0.1456</td> <td>    0.113</td> <td>   -1.291</td> <td> 0.197</td> <td>   -0.367</td> <td>    0.076</td>
</tr>
<tr>
  <th>ExterCond_Po</th>             <td>    0.1969</td> <td>    0.107</td> <td>    1.848</td> <td> 0.065</td> <td>   -0.012</td> <td>    0.406</td>
</tr>
<tr>
  <th>BsmtQual_Unavailable</th>     <td>    0.1825</td> <td>    0.051</td> <td>    3.571</td> <td> 0.000</td> <td>    0.082</td> <td>    0.283</td>
</tr>
<tr>
  <th>BsmtFinType1_LwQ</th>         <td>    0.1123</td> <td>    0.021</td> <td>    5.408</td> <td> 0.000</td> <td>    0.072</td> <td>    0.153</td>
</tr>
<tr>
  <th>BsmtFinType2_Unavailable</th> <td>    0.0801</td> <td>    0.102</td> <td>    0.785</td> <td> 0.433</td> <td>   -0.120</td> <td>    0.281</td>
</tr>
<tr>
  <th>ExterCond_TA</th>             <td>    0.2198</td> <td>    0.035</td> <td>    6.310</td> <td> 0.000</td> <td>    0.151</td> <td>    0.288</td>
</tr>
<tr>
  <th>LotFrontage</th>              <td>   -0.0007</td> <td>    0.004</td> <td>   -0.167</td> <td> 0.867</td> <td>   -0.009</td> <td>    0.008</td>
</tr>
<tr>
  <th>Exterior2nd_Stucco</th>       <td>    0.1137</td> <td>    0.127</td> <td>    0.892</td> <td> 0.373</td> <td>   -0.136</td> <td>    0.364</td>
</tr>
<tr>
  <th>HouseStyle_SLvl</th>          <td>    0.1381</td> <td>    0.092</td> <td>    1.496</td> <td> 0.135</td> <td>   -0.043</td> <td>    0.319</td>
</tr>
<tr>
  <th>KitchenQual_Fa</th>           <td>    0.2316</td> <td>    0.029</td> <td>    8.123</td> <td> 0.000</td> <td>    0.176</td> <td>    0.288</td>
</tr>
<tr>
  <th>SaleCondition_Family</th>     <td>    0.1476</td> <td>    0.039</td> <td>    3.767</td> <td> 0.000</td> <td>    0.071</td> <td>    0.224</td>
</tr>
<tr>
  <th>OpenPorchSF</th>              <td>    0.0033</td> <td>    0.004</td> <td>    0.824</td> <td> 0.410</td> <td>   -0.005</td> <td>    0.011</td>
</tr>
<tr>
  <th>Neighborhood_Mitchel</th>     <td>    0.0195</td> <td>    0.022</td> <td>    0.872</td> <td> 0.383</td> <td>   -0.024</td> <td>    0.063</td>
</tr>
<tr>
  <th>RoofStyle_Mansard</th>        <td>    0.2066</td> <td>    0.068</td> <td>    3.059</td> <td> 0.002</td> <td>    0.074</td> <td>    0.339</td>
</tr>
<tr>
  <th>Exterior1st_Wd Sdng</th>      <td>   -0.0749</td> <td>    0.084</td> <td>   -0.895</td> <td> 0.371</td> <td>   -0.239</td> <td>    0.089</td>
</tr>
<tr>
  <th>Electrical_FuseF</th>         <td>   -0.0499</td> <td>    0.119</td> <td>   -0.419</td> <td> 0.676</td> <td>   -0.284</td> <td>    0.184</td>
</tr>
<tr>
  <th>Functional_Maj1</th>          <td>    0.2310</td> <td>    0.047</td> <td>    4.882</td> <td> 0.000</td> <td>    0.138</td> <td>    0.324</td>
</tr>
<tr>
  <th>SaleCondition_Partial</th>    <td>    0.0602</td> <td>    0.079</td> <td>    0.759</td> <td> 0.448</td> <td>   -0.096</td> <td>    0.216</td>
</tr>
<tr>
  <th>Condition2_PosA</th>          <td>    0.2127</td> <td>    0.176</td> <td>    1.207</td> <td> 0.228</td> <td>   -0.133</td> <td>    0.558</td>
</tr>
<tr>
  <th>GarageCond_TA</th>            <td>    0.2155</td> <td>    0.055</td> <td>    3.945</td> <td> 0.000</td> <td>    0.108</td> <td>    0.323</td>
</tr>
<tr>
  <th>MasVnrType_Unavailable</th>   <td>    0.1729</td> <td>    0.037</td> <td>    4.645</td> <td> 0.000</td> <td>    0.100</td> <td>    0.246</td>
</tr>
<tr>
  <th>Neighborhood_SawyerW</th>     <td>    0.0570</td> <td>    0.021</td> <td>    2.720</td> <td> 0.007</td> <td>    0.016</td> <td>    0.098</td>
</tr>
<tr>
  <th>LotArea</th>                  <td>    0.0255</td> <td>    0.006</td> <td>    4.191</td> <td> 0.000</td> <td>    0.014</td> <td>    0.037</td>
</tr>
<tr>
  <th>BsmtFinType2_BLQ</th>         <td>    0.1155</td> <td>    0.031</td> <td>    3.762</td> <td> 0.000</td> <td>    0.055</td> <td>    0.176</td>
</tr>
<tr>
  <th>BsmtFinType2_GLQ</th>         <td>    0.1988</td> <td>    0.041</td> <td>    4.874</td> <td> 0.000</td> <td>    0.119</td> <td>    0.279</td>
</tr>
<tr>
  <th>Fence_GdPrv</th>              <td>    0.2170</td> <td>    0.023</td> <td>    9.255</td> <td> 0.000</td> <td>    0.171</td> <td>    0.263</td>
</tr>
<tr>
  <th>Exterior2nd_Brk Cmn</th>      <td>    0.1181</td> <td>    0.145</td> <td>    0.815</td> <td> 0.415</td> <td>   -0.166</td> <td>    0.402</td>
</tr>
<tr>
  <th>GarageType_Detchd</th>        <td>    0.1804</td> <td>    0.031</td> <td>    5.730</td> <td> 0.000</td> <td>    0.119</td> <td>    0.242</td>
</tr>
<tr>
  <th>RoofMatl_WdShake</th>         <td>    0.0668</td> <td>    0.129</td> <td>    0.516</td> <td> 0.606</td> <td>   -0.187</td> <td>    0.321</td>
</tr>
<tr>
  <th>Electrical_FuseA</th>         <td>   -0.0139</td> <td>    0.116</td> <td>   -0.120</td> <td> 0.905</td> <td>   -0.242</td> <td>    0.214</td>
</tr>
<tr>
  <th>GarageFinish_Unf</th>         <td>    0.3214</td> <td>    0.053</td> <td>    6.085</td> <td> 0.000</td> <td>    0.218</td> <td>    0.425</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>208.281</td> <th>  Durbin-Watson:     </th> <td>   1.973</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2766.588</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.385</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>10.500</td>  <th>  Cond. No.          </th> <td>1.59e+16</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 1.3e-28. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
score_ols = simple_results.rsquared
print(score_ols)
```

    0.9415116730711321

Now we will use Linear Regression

```python
from sklearn.metrics import mean_squared_error
```


```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
y_pred_lr = model.predict(X_val)
score_lr = model.score(X_val, y_val)
print(mean_squared_error(y_pred_lr, y_val))
mse_lr = mean_squared_error(y_pred_lr, y_val)
score_lr
```

    2.1621580410581166e+18





    -1.2458011824597305e+19


Next model will be Decision Tree

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth = 5, random_state = 1)
```


```python
model = regressor.fit(X_train, y_train)
y_pred_dt = model.predict(X_val)
score_dt = model.score(X_val, y_val)
print(mean_squared_error(y_pred_dt, y_val))
mse_dt = mean_squared_error(y_pred_dt, y_val)
score_dt
```

    0.052162345694257435





    0.6994488343970576


This model we will use Random Forest

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
```


```python
list = []
for i in range(1,10): 
    regr = RandomForestRegressor(n_estimators = 15, max_depth=i,
                             random_state=1)
    model = regr.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    list.append(model.score(X_val, y_val))
pd.DataFrame(list)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.476775</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.669219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.758133</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.808429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.845350</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.857478</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.863039</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.863190</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.868691</td>
    </tr>
  </tbody>
</table>
</div>




```python
list = []
for i in range(15,25): 
    regr = RandomForestRegressor(n_estimators=i, max_depth=9,
                             random_state=1)
    model = regr.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    list.append(model.score(X_val, y_val))
```


```python
list = pd.DataFrame(list)
list
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.868691</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.868021</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.869254</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.871266</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.872545</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.874128</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.873133</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.873233</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.872792</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.872879</td>
    </tr>
  </tbody>
</table>
</div>




```python
regr = RandomForestRegressor(n_estimators= 20, max_depth=9,
                             random_state=1)
model = regr.fit(X_train, y_train)
y_pred_ranf = model.predict(X_val)
score_ranf = model.score(X_val, y_val)
print(mean_squared_error(y_pred_ranf, y_val))
mse_ranf = mean_squared_error(y_pred_ranf, y_val)
score_ranf
```

    0.021845809995216174





    0.8741279064387348


Next model to test will be XgBoost

```python
import xgboost
xgb = xgboost.XGBRegressor(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=30)
```


```python
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_val)
score_xgb = xgb.score(X_val, y_val)
print(mean_squared_error(y_pred_xgb, y_val))
mse_xgb = mean_squared_error(y_pred_xgb, y_val)
print(score_xgb)
```

    0.017485096299964063
    0.8992536473640135



```python
from sklearn.ensemble import GradientBoostingRegressor
```


```python
gbr = GradientBoostingRegressor(n_estimators=79, random_state=1)
```


```python
gbr.fit(X_train,y_train)
y_pred_gbr = gbr.predict(X_val)
score_gbr = gbr.score(X_val, y_val)
print(mean_squared_error(y_pred_gbr, y_val))
mse_gbr = mean_squared_error(y_pred_gbr, y_val)
print(score_gbr)
```

    0.017248372643681103
    0.9006176115392186



```python
scores_list = ['score_ols','score_lr','score_dt','score_ranf','score_xgb','score_gbr']
mse = ['NA', mse_lr,mse_dt,mse_ranf,mse_xgb,mse_gbr]
scores = [score_ols,score_lr,score_dt,score_ranf,score_xgb,score_gbr]
```


```python
score_df = pd.DataFrame([scores_list, scores, mse]).T
```


```python
score_df.index = score_df[0]
del score_df[0]
score_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>score_ols</th>
      <td>0.941512</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>score_lr</th>
      <td>-1.2458e+19</td>
      <td>2.16216e+18</td>
    </tr>
    <tr>
      <th>score_dt</th>
      <td>0.699449</td>
      <td>0.0521623</td>
    </tr>
    <tr>
      <th>score_ranf</th>
      <td>0.874128</td>
      <td>0.0218458</td>
    </tr>
    <tr>
      <th>score_xgb</th>
      <td>0.899254</td>
      <td>0.0174851</td>
    </tr>
    <tr>
      <th>score_gbr</th>
      <td>0.900618</td>
      <td>0.0172484</td>
    </tr>
  </tbody>
</table>
</div>


The best r-sqare value is 90.0618% produced by the Gradient Boosting Regressor model.We will now take the best score of GBR and apply it to the test house price data set.

```python
test_prediction_r = gbr.fit(X_train,y_train).predict(test_X)
```


```python
test_prediction_r = np.exp(test_prediction_r)
```
The histograms below will show us the sale price distribution on the original data versus the test data.

```python
plt.subplot(1, 2, 1)
train.SalePrice.hist(bins = 100, figsize = (15,5))
plt.title('Original Data Set Sales Prices')
plt.subplot(1,2,2)
plt.hist(test_prediction_r, bins = 100)
plt.title('Test Data Set Sales Prices')
plt.show()
```


![png](output_80_0.png)

Google H2O

```python
import h2o
from h2o.automl import H2OAutoML
h2o.init()
```

    Checking whether there is an H2O instance running at http://localhost:54321..... not found.
    Attempting to start a local H2O server...
      Java Version: openjdk version "10.0.2" 2018-07-17; OpenJDK Runtime Environment (build 10.0.2+13-Ubuntu-1ubuntu0.18.04.4); OpenJDK 64-Bit Server VM (build 10.0.2+13-Ubuntu-1ubuntu0.18.04.4, mixed mode)
      Starting server from /home/m/anaconda3/lib/python3.6/site-packages/h2o/backend/bin/h2o.jar
      Ice root: /tmp/tmp811e9_92
      JVM stdout: /tmp/tmp811e9_92/h2o_m_started_from_python.out
      JVM stderr: /tmp/tmp811e9_92/h2o_m_started_from_python.err
      Server is running at http://127.0.0.1:54321
    Connecting to H2O server at http://127.0.0.1:54321... successful.



<div style="overflow:auto"><table style="width:50%"><tr><td>H2O cluster uptime:</td>
<td>03 secs</td></tr>
<tr><td>H2O cluster timezone:</td>
<td>America/New_York</td></tr>
<tr><td>H2O data parsing timezone:</td>
<td>UTC</td></tr>
<tr><td>H2O cluster version:</td>
<td>3.22.0.2</td></tr>
<tr><td>H2O cluster version age:</td>
<td>1 month and 13 days </td></tr>
<tr><td>H2O cluster name:</td>
<td>H2O_from_python_m_bnxuqe</td></tr>
<tr><td>H2O cluster total nodes:</td>
<td>1</td></tr>
<tr><td>H2O cluster free memory:</td>
<td>1.377 Gb</td></tr>
<tr><td>H2O cluster total cores:</td>
<td>4</td></tr>
<tr><td>H2O cluster allowed cores:</td>
<td>4</td></tr>
<tr><td>H2O cluster status:</td>
<td>accepting new members, healthy</td></tr>
<tr><td>H2O connection url:</td>
<td>http://127.0.0.1:54321</td></tr>
<tr><td>H2O connection proxy:</td>
<td>None</td></tr>
<tr><td>H2O internal security:</td>
<td>False</td></tr>
<tr><td>H2O API Extensions:</td>
<td>XGBoost, Algos, AutoML, Core V3, Core V4</td></tr>
<tr><td>Python version:</td>
<td>3.6.5 final</td></tr></table></div>



```python
# Load data into H2O
df = h2o.H2OFrame(train_X)
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%



```python
df.describe()
```

    Rows:1460
    Cols:306
    
    



<table>
<thead>
<tr><th>       </th><th>Id                    </th><th>MSSubClass            </th><th>LotFrontage          </th><th>LotArea                </th><th>OverallQual           </th><th>OverallCond           </th><th>YearBuilt             </th><th>YearRemodAdd         </th><th>MasVnrArea           </th><th>BsmtFinSF1            </th><th>BsmtFinSF2            </th><th>BsmtUnfSF             </th><th>TotalBsmtSF          </th><th>1stFlrSF             </th><th>2ndFlrSF              </th><th>LowQualFinSF          </th><th>GrLivArea              </th><th>BsmtFullBath          </th><th>BsmtHalfBath         </th><th>FullBath              </th><th>HalfBath             </th><th>BedroomAbvGr          </th><th>KitchenAbvGr          </th><th>TotRmsAbvGrd          </th><th>Fireplaces           </th><th>GarageYrBlt           </th><th>GarageCars           </th><th>GarageArea             </th><th>WoodDeckSF          </th><th>OpenPorchSF          </th><th>EnclosedPorch        </th><th>3SsnPorch             </th><th>ScreenPorch           </th><th>PoolArea             </th><th>MiscVal               </th><th>MoSold               </th><th>YrSold               </th><th>SalePrice          </th><th>MSZoning_C (all)   </th><th>MSZoning_FV        </th><th>MSZoning_RH         </th><th>MSZoning_RL       </th><th>MSZoning_RM        </th><th>Street_Grvl        </th><th>Street_Pave        </th><th>Alley_Grvl         </th><th>Alley_Pave          </th><th>Alley_Unavailable  </th><th>LotShape_IR1      </th><th>LotShape_IR2        </th><th>LotShape_IR3       </th><th>LotShape_Reg       </th><th>LandContour_Bnk    </th><th>LandContour_HLS    </th><th>LandContour_Low     </th><th>LandContour_Lvl    </th><th>Utilities_AllPub    </th><th>Utilities_NoSeWa     </th><th>LotConfig_Corner   </th><th>LotConfig_CulDSac  </th><th>LotConfig_FR2      </th><th>LotConfig_FR3        </th><th>LotConfig_Inside   </th><th>LandSlope_Gtl      </th><th>LandSlope_Mod      </th><th>LandSlope_Sev       </th><th>Neighborhood_Blmngtn  </th><th>Neighborhood_Blueste  </th><th>Neighborhood_BrDale  </th><th>Neighborhood_BrkSide  </th><th>Neighborhood_ClearCr  </th><th>Neighborhood_CollgCr  </th><th>Neighborhood_Crawfor  </th><th>Neighborhood_Edwards  </th><th>Neighborhood_Gilbert  </th><th>Neighborhood_IDOTRR  </th><th>Neighborhood_MeadowV  </th><th>Neighborhood_Mitchel  </th><th>Neighborhood_NAmes  </th><th>Neighborhood_NPkVill  </th><th>Neighborhood_NWAmes  </th><th>Neighborhood_NoRidge  </th><th>Neighborhood_NridgHt  </th><th>Neighborhood_OldTown  </th><th>Neighborhood_SWISU  </th><th>Neighborhood_Sawyer  </th><th>Neighborhood_SawyerW  </th><th>Neighborhood_Somerst  </th><th>Neighborhood_StoneBr  </th><th>Neighborhood_Timber  </th><th>Neighborhood_Veenker  </th><th>Condition1_Artery  </th><th>Condition1_Feedr   </th><th>Condition1_Norm  </th><th>Condition1_PosA     </th><th>Condition1_PosN     </th><th>Condition1_RRAe     </th><th>Condition1_RRAn    </th><th>Condition1_RRNe      </th><th>Condition1_RRNn     </th><th>Condition2_Artery    </th><th>Condition2_Feedr   </th><th>Condition2_Norm    </th><th>Condition2_PosA      </th><th>Condition2_PosN      </th><th>Condition2_RRAe      </th><th>Condition2_RRAn      </th><th>Condition2_RRNn      </th><th>BldgType_1Fam     </th><th>BldgType_2fmCon     </th><th>BldgType_Duplex    </th><th>BldgType_Twnhs     </th><th>BldgType_TwnhsE    </th><th>HouseStyle_1.5Fin  </th><th>HouseStyle_1.5Unf   </th><th>HouseStyle_1Story  </th><th>HouseStyle_2.5Fin   </th><th>HouseStyle_2.5Unf   </th><th>HouseStyle_2Story  </th><th>HouseStyle_SFoyer   </th><th>HouseStyle_SLvl    </th><th>RoofStyle_Flat      </th><th>RoofStyle_Gable    </th><th>RoofStyle_Gambrel   </th><th>RoofStyle_Hip     </th><th>RoofStyle_Mansard   </th><th>RoofStyle_Shed       </th><th>RoofMatl_ClyTile     </th><th>RoofMatl_CompShg  </th><th>RoofMatl_Membran     </th><th>RoofMatl_Metal       </th><th>RoofMatl_Roll        </th><th>RoofMatl_Tar&Grv    </th><th>RoofMatl_WdShake    </th><th>RoofMatl_WdShngl   </th><th>Exterior1st_AsbShng  </th><th>Exterior1st_AsphShn  </th><th>Exterior1st_BrkComm  </th><th>Exterior1st_BrkFace  </th><th>Exterior1st_CBlock   </th><th>Exterior1st_CemntBd  </th><th>Exterior1st_HdBoard  </th><th>Exterior1st_ImStucc  </th><th>Exterior1st_MetalSd  </th><th>Exterior1st_Plywood  </th><th>Exterior1st_Stone    </th><th>Exterior1st_Stucco  </th><th>Exterior1st_VinylSd  </th><th>Exterior1st_Wd Sdng  </th><th>Exterior1st_WdShing  </th><th>Exterior2nd_AsbShng  </th><th>Exterior2nd_AsphShn  </th><th>Exterior2nd_Brk Cmn  </th><th>Exterior2nd_BrkFace  </th><th>Exterior2nd_CBlock   </th><th>Exterior2nd_CmentBd  </th><th>Exterior2nd_HdBoard  </th><th>Exterior2nd_ImStucc  </th><th>Exterior2nd_MetalSd  </th><th>Exterior2nd_Other    </th><th>Exterior2nd_Plywood  </th><th>Exterior2nd_Stone   </th><th>Exterior2nd_Stucco  </th><th>Exterior2nd_VinylSd  </th><th>Exterior2nd_Wd Sdng  </th><th>Exterior2nd_Wd Shng  </th><th>MasVnrType_BrkCmn   </th><th>MasVnrType_BrkFace  </th><th>MasVnrType_None   </th><th>MasVnrType_Stone   </th><th>MasVnrType_Unavailable  </th><th>ExterQual_Ex       </th><th>ExterQual_Fa        </th><th>ExterQual_Gd       </th><th>ExterQual_TA       </th><th>ExterCond_Ex        </th><th>ExterCond_Fa        </th><th>ExterCond_Gd     </th><th>ExterCond_Po         </th><th>ExterCond_TA       </th><th>Foundation_BrkTil  </th><th>Foundation_CBlock  </th><th>Foundation_PConc   </th><th>Foundation_Slab    </th><th>Foundation_Stone   </th><th>Foundation_Wood     </th><th>BsmtQual_Ex        </th><th>BsmtQual_Fa         </th><th>BsmtQual_Gd        </th><th>BsmtQual_TA        </th><th>BsmtQual_Unavailable  </th><th>BsmtCond_Fa         </th><th>BsmtCond_Gd        </th><th>BsmtCond_Po          </th><th>BsmtCond_TA        </th><th>BsmtCond_Unavailable  </th><th>BsmtExposure_Av    </th><th>BsmtExposure_Gd    </th><th>BsmtExposure_Mn    </th><th>BsmtExposure_No    </th><th>BsmtExposure_Unavailable  </th><th>BsmtFinType1_ALQ  </th><th>BsmtFinType1_BLQ   </th><th>BsmtFinType1_GLQ  </th><th>BsmtFinType1_LwQ    </th><th>BsmtFinType1_Rec   </th><th>BsmtFinType1_Unavailable  </th><th>BsmtFinType1_Unf  </th><th>BsmtFinType2_ALQ    </th><th>BsmtFinType2_BLQ    </th><th>BsmtFinType2_GLQ    </th><th>BsmtFinType2_LwQ    </th><th>BsmtFinType2_Rec    </th><th>BsmtFinType2_Unavailable  </th><th>BsmtFinType2_Unf   </th><th>Heating_Floor        </th><th>Heating_GasA       </th><th>Heating_GasW        </th><th>Heating_Grav        </th><th>Heating_OthW         </th><th>Heating_Wall         </th><th>HeatingQC_Ex      </th><th>HeatingQC_Fa       </th><th>HeatingQC_Gd       </th><th>HeatingQC_Po         </th><th>HeatingQC_TA       </th><th>CentralAir_N       </th><th>CentralAir_Y       </th><th>Electrical_FuseA   </th><th>Electrical_FuseF    </th><th>Electrical_FuseP    </th><th>Electrical_Mix       </th><th>Electrical_SBrkr  </th><th>Electrical_Unavailable  </th><th>KitchenQual_Ex    </th><th>KitchenQual_Fa     </th><th>KitchenQual_Gd     </th><th>KitchenQual_TA    </th><th>Functional_Maj1     </th><th>Functional_Maj2     </th><th>Functional_Min1     </th><th>Functional_Min2     </th><th>Functional_Mod      </th><th>Functional_Sev       </th><th>Functional_Typ    </th><th>FireplaceQu_Ex     </th><th>FireplaceQu_Fa      </th><th>FireplaceQu_Gd    </th><th>FireplaceQu_Po    </th><th>FireplaceQu_TA     </th><th>FireplaceQu_Unavailable  </th><th>GarageType_2Types  </th><th>GarageType_Attchd  </th><th>GarageType_Basment  </th><th>GarageType_BuiltIn  </th><th>GarageType_CarPort   </th><th>GarageType_Detchd  </th><th>GarageType_Unavailable  </th><th>GarageFinish_Fin   </th><th>GarageFinish_RFn   </th><th>GarageFinish_Unavailable  </th><th>GarageFinish_Unf  </th><th>GarageQual_Ex       </th><th>GarageQual_Fa      </th><th>GarageQual_Gd       </th><th>GarageQual_Po       </th><th>GarageQual_TA      </th><th>GarageQual_Unavailable  </th><th>GarageCond_Ex        </th><th>GarageCond_Fa       </th><th>GarageCond_Gd        </th><th>GarageCond_Po       </th><th>GarageCond_TA      </th><th>GarageCond_Unavailable  </th><th>PavedDrive_N       </th><th>PavedDrive_P       </th><th>PavedDrive_Y       </th><th>PoolQC_Ex            </th><th>PoolQC_Fa            </th><th>PoolQC_Gd           </th><th>PoolQC_Unavailable  </th><th>Fence_GdPrv        </th><th>Fence_GdWo          </th><th>Fence_MnPrv        </th><th>Fence_MnWw          </th><th>Fence_Unavailable  </th><th>MiscFeature_Gar2     </th><th>MiscFeature_Othr     </th><th>MiscFeature_Shed   </th><th>MiscFeature_TenC     </th><th>MiscFeature_Unavailable  </th><th>SaleType_COD       </th><th>SaleType_CWD         </th><th>SaleType_Con         </th><th>SaleType_ConLD       </th><th>SaleType_ConLI      </th><th>SaleType_ConLw      </th><th>SaleType_New       </th><th>SaleType_Oth        </th><th>SaleType_WD       </th><th>SaleCondition_Abnorml  </th><th>SaleCondition_AdjLand  </th><th>SaleCondition_Alloca  </th><th>SaleCondition_Family  </th><th>SaleCondition_Normal  </th><th>SaleCondition_Partial  </th></tr>
</thead>
<tbody>
<tr><td>type   </td><td>real                  </td><td>real                  </td><td>real                 </td><td>real                   </td><td>real                  </td><td>real                  </td><td>real                  </td><td>real                 </td><td>real                 </td><td>real                  </td><td>real                  </td><td>real                  </td><td>real                 </td><td>real                 </td><td>real                  </td><td>real                  </td><td>real                   </td><td>real                  </td><td>real                 </td><td>real                  </td><td>real                 </td><td>real                  </td><td>real                  </td><td>real                  </td><td>real                 </td><td>real                  </td><td>real                 </td><td>real                   </td><td>real                </td><td>real                 </td><td>real                 </td><td>real                  </td><td>real                  </td><td>real                 </td><td>real                  </td><td>real                 </td><td>real                 </td><td>real               </td><td>int                </td><td>int                </td><td>int                 </td><td>int               </td><td>int                </td><td>int                </td><td>int                </td><td>int                </td><td>int                 </td><td>int                </td><td>int               </td><td>int                 </td><td>int                </td><td>int                </td><td>int                </td><td>int                </td><td>int                 </td><td>int                </td><td>int                 </td><td>int                  </td><td>int                </td><td>int                </td><td>int                </td><td>int                  </td><td>int                </td><td>int                </td><td>int                </td><td>int                 </td><td>int                   </td><td>int                   </td><td>int                  </td><td>int                   </td><td>int                   </td><td>int                   </td><td>int                   </td><td>int                   </td><td>int                   </td><td>int                  </td><td>int                   </td><td>int                   </td><td>int                 </td><td>int                   </td><td>int                  </td><td>int                   </td><td>int                   </td><td>int                   </td><td>int                 </td><td>int                  </td><td>int                   </td><td>int                   </td><td>int                   </td><td>int                  </td><td>int                   </td><td>int                </td><td>int                </td><td>int              </td><td>int                 </td><td>int                 </td><td>int                 </td><td>int                </td><td>int                  </td><td>int                 </td><td>int                  </td><td>int                </td><td>int                </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int               </td><td>int                 </td><td>int                </td><td>int                </td><td>int                </td><td>int                </td><td>int                 </td><td>int                </td><td>int                 </td><td>int                 </td><td>int                </td><td>int                 </td><td>int                </td><td>int                 </td><td>int                </td><td>int                 </td><td>int               </td><td>int                 </td><td>int                  </td><td>int                  </td><td>int               </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                 </td><td>int                 </td><td>int                </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                 </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                 </td><td>int                 </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                 </td><td>int                 </td><td>int               </td><td>int                </td><td>int                     </td><td>int                </td><td>int                 </td><td>int                </td><td>int                </td><td>int                 </td><td>int                 </td><td>int              </td><td>int                  </td><td>int                </td><td>int                </td><td>int                </td><td>int                </td><td>int                </td><td>int                </td><td>int                 </td><td>int                </td><td>int                 </td><td>int                </td><td>int                </td><td>int                   </td><td>int                 </td><td>int                </td><td>int                  </td><td>int                </td><td>int                   </td><td>int                </td><td>int                </td><td>int                </td><td>int                </td><td>int                       </td><td>int               </td><td>int                </td><td>int               </td><td>int                 </td><td>int                </td><td>int                       </td><td>int               </td><td>int                 </td><td>int                 </td><td>int                 </td><td>int                 </td><td>int                 </td><td>int                       </td><td>int                </td><td>int                  </td><td>int                </td><td>int                 </td><td>int                 </td><td>int                  </td><td>int                  </td><td>int               </td><td>int                </td><td>int                </td><td>int                  </td><td>int                </td><td>int                </td><td>int                </td><td>int                </td><td>int                 </td><td>int                 </td><td>int                  </td><td>int               </td><td>int                     </td><td>int               </td><td>int                </td><td>int                </td><td>int               </td><td>int                 </td><td>int                 </td><td>int                 </td><td>int                 </td><td>int                 </td><td>int                  </td><td>int               </td><td>int                </td><td>int                 </td><td>int               </td><td>int               </td><td>int                </td><td>int                      </td><td>int                </td><td>int                </td><td>int                 </td><td>int                 </td><td>int                  </td><td>int                </td><td>int                     </td><td>int                </td><td>int                </td><td>int                       </td><td>int               </td><td>int                 </td><td>int                </td><td>int                 </td><td>int                 </td><td>int                </td><td>int                     </td><td>int                  </td><td>int                 </td><td>int                  </td><td>int                 </td><td>int                </td><td>int                     </td><td>int                </td><td>int                </td><td>int                </td><td>int                  </td><td>int                  </td><td>int                 </td><td>int                 </td><td>int                </td><td>int                 </td><td>int                </td><td>int                 </td><td>int                </td><td>int                  </td><td>int                  </td><td>int                </td><td>int                  </td><td>int                      </td><td>int                </td><td>int                  </td><td>int                  </td><td>int                  </td><td>int                 </td><td>int                 </td><td>int                </td><td>int                 </td><td>int               </td><td>int                    </td><td>int                    </td><td>int                   </td><td>int                   </td><td>int                   </td><td>int                    </td></tr>
<tr><td>mins   </td><td>-1.730864877400689    </td><td>-0.8725627562389217   </td><td>-1.6628930854346002  </td><td>-0.9237292282108068    </td><td>-3.6884129023269017   </td><td>-4.112969893342749    </td><td>-3.287823627081604    </td><td>-1.689368497954768   </td><td>-0.5707501337786359  </td><td>-0.9730181828073764   </td><td>-0.2886528311122454   </td><td>-1.28417561989079     </td><td>-2.411166932760057   </td><td>-2.1441720802090702  </td><td>-0.7951632272572445   </td><td>-0.12024172373467248  </td><td>-2.2491201474608564    </td><td>-0.8199643654586394   </td><td>-0.24106103579929675 </td><td>-2.841822193737443    </td><td>-0.7616206719986545  </td><td>-3.514951815977729    </td><td>-4.751486354069578    </td><td>-2.780469339451636    </td><td>-0.9512264882332891  </td><td>-4.120324395421982    </td><td>-2.365439947036252   </td><td>-2.2129629830885067    </td><td>-0.7521758378613592 </td><td>-0.704483250333654   </td><td>-0.3593249004055313  </td><td>-0.11633928614822256  </td><td>-0.2702083542015609   </td><td>-0.06869174753820707 </td><td>-0.08768781151769862  </td><td>-1.969111448594116   </td><td>-1.3676547331070554  </td><td>10.46024210819052  </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>0.0              </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0               </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0               </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0              </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                   </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                       </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0               </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0               </td><td>0.0                     </td><td>0.0               </td><td>0.0                </td><td>0.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0               </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0               </td><td>0.0                </td><td>0.0                      </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                     </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                 </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                </td><td>0.0                  </td><td>0.0                      </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                    </td><td>0.0                    </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                    </td></tr>
<tr><td>mean   </td><td>1.2627269022069676e-14</td><td>-8.771195575407731e-17</td><td>9.513874063560301e-17</td><td>-1.0102053742133688e-16</td><td>1.2067170179763664e-16</td><td>3.4092737313806687e-16</td><td>1.0424332837905004e-15</td><td>4.480899158665341e-15</td><td>-8.61940727125976e-18</td><td>1.8214596497756474e-17</td><td>-8.673617379884035e-19</td><td>-7.752045533271357e-17</td><td>2.436744382661171e-16</td><td>5.797771117366235e-17</td><td>-5.204170427930421e-18</td><td>-5.346471963069144e-17</td><td>-1.4322310698533514e-16</td><td>-7.079840186330344e-17</td><td>3.778444571111983e-17</td><td>1.0657707355532509e-16</td><td>5.963111948670274e-17</td><td>2.7990034335428904e-16</td><td>4.1356891869459567e-16</td><td>-9.698188432882837e-17</td><td>6.429318882839041e-17</td><td>1.5650458359828257e-16</td><td>7.817097663620487e-17</td><td>-1.5504091066542713e-17</td><td>2.42861286636753e-17</td><td>2.916503843986007e-17</td><td>1.100465205072787e-17</td><td>1.2332799712022613e-18</td><td>2.7321894746634712e-17</td><td>2.382534274036896e-17</td><td>-6.701724678676024e-18</td><td>9.951620690701324e-17</td><td>3.568060560552033e-14</td><td>12.02405090110935  </td><td>0.00684931506849315</td><td>0.04452054794520548</td><td>0.010958904109589041</td><td>0.7883561643835616</td><td>0.14931506849315068</td><td>0.00410958904109589</td><td>0.9958904109589041 </td><td>0.03424657534246575</td><td>0.028082191780821917</td><td>0.9376712328767123 </td><td>0.3315068493150685</td><td>0.028082191780821917</td><td>0.00684931506849315</td><td>0.6335616438356164 </td><td>0.04315068493150685</td><td>0.03424657534246575</td><td>0.024657534246575342</td><td>0.897945205479452  </td><td>0.9993150684931507  </td><td>0.0006849315068493151</td><td>0.18013698630136987</td><td>0.06438356164383562</td><td>0.03219178082191781</td><td>0.0027397260273972603</td><td>0.7205479452054795 </td><td>0.9465753424657535 </td><td>0.04452054794520548</td><td>0.008904109589041096</td><td>0.011643835616438357  </td><td>0.0013698630136986301 </td><td>0.010958904109589041 </td><td>0.03972602739726028   </td><td>0.019178082191780823  </td><td>0.10273972602739725   </td><td>0.03493150684931507   </td><td>0.0684931506849315    </td><td>0.05410958904109589   </td><td>0.025342465753424658 </td><td>0.011643835616438357  </td><td>0.03356164383561644   </td><td>0.1541095890410959  </td><td>0.0061643835616438354 </td><td>0.05                 </td><td>0.028082191780821917  </td><td>0.05273972602739726   </td><td>0.0773972602739726    </td><td>0.017123287671232876</td><td>0.050684931506849315 </td><td>0.04041095890410959   </td><td>0.0589041095890411    </td><td>0.017123287671232876  </td><td>0.026027397260273973 </td><td>0.007534246575342466  </td><td>0.03287671232876712</td><td>0.05547945205479452</td><td>0.863013698630137</td><td>0.005479452054794521</td><td>0.013013698630136987</td><td>0.007534246575342466</td><td>0.01780821917808219</td><td>0.0013698630136986301</td><td>0.003424657534246575</td><td>0.0013698630136986301</td><td>0.00410958904109589</td><td>0.9897260273972602 </td><td>0.0006849315068493151</td><td>0.0013698630136986301</td><td>0.0006849315068493151</td><td>0.0006849315068493151</td><td>0.0013698630136986301</td><td>0.8356164383561644</td><td>0.021232876712328767</td><td>0.03561643835616438</td><td>0.02945205479452055</td><td>0.07808219178082192</td><td>0.10547945205479452</td><td>0.009589041095890411</td><td>0.49726027397260275</td><td>0.005479452054794521</td><td>0.007534246575342466</td><td>0.3047945205479452 </td><td>0.025342465753424658</td><td>0.04452054794520548</td><td>0.008904109589041096</td><td>0.7815068493150685 </td><td>0.007534246575342466</td><td>0.1958904109589041</td><td>0.004794520547945206</td><td>0.0013698630136986301</td><td>0.0006849315068493151</td><td>0.9821917808219178</td><td>0.0006849315068493151</td><td>0.0006849315068493151</td><td>0.0006849315068493151</td><td>0.007534246575342466</td><td>0.003424657534246575</td><td>0.00410958904109589</td><td>0.0136986301369863   </td><td>0.0006849315068493151</td><td>0.0013698630136986301</td><td>0.03424657534246575  </td><td>0.0006849315068493151</td><td>0.04178082191780822  </td><td>0.15205479452054796  </td><td>0.0006849315068493151</td><td>0.1506849315068493   </td><td>0.07397260273972603  </td><td>0.0013698630136986301</td><td>0.017123287671232876</td><td>0.3527397260273973   </td><td>0.1410958904109589   </td><td>0.01780821917808219  </td><td>0.0136986301369863   </td><td>0.002054794520547945 </td><td>0.004794520547945206 </td><td>0.017123287671232876 </td><td>0.0006849315068493151</td><td>0.0410958904109589   </td><td>0.14178082191780822  </td><td>0.00684931506849315  </td><td>0.14657534246575343  </td><td>0.0006849315068493151</td><td>0.09726027397260274  </td><td>0.003424657534246575</td><td>0.01780821917808219 </td><td>0.3452054794520548   </td><td>0.13493150684931507  </td><td>0.026027397260273973 </td><td>0.010273972602739725</td><td>0.3047945205479452  </td><td>0.5917808219178082</td><td>0.08767123287671233</td><td>0.005479452054794521    </td><td>0.03561643835616438</td><td>0.009589041095890411</td><td>0.33424657534246577</td><td>0.6205479452054794 </td><td>0.002054794520547945</td><td>0.019178082191780823</td><td>0.1              </td><td>0.0006849315068493151</td><td>0.8780821917808219 </td><td>0.1                </td><td>0.43424657534246575</td><td>0.44315068493150683</td><td>0.01643835616438356</td><td>0.00410958904109589</td><td>0.002054794520547945</td><td>0.08287671232876713</td><td>0.023972602739726026</td><td>0.4232876712328767 </td><td>0.44452054794520546</td><td>0.025342465753424658  </td><td>0.030821917808219176</td><td>0.04452054794520548</td><td>0.0013698630136986301</td><td>0.897945205479452  </td><td>0.025342465753424658  </td><td>0.15136986301369862</td><td>0.09178082191780822</td><td>0.07808219178082192</td><td>0.6527397260273973 </td><td>0.026027397260273973      </td><td>0.1506849315068493</td><td>0.10136986301369863</td><td>0.2863013698630137</td><td>0.050684931506849315</td><td>0.0910958904109589 </td><td>0.025342465753424658      </td><td>0.2945205479452055</td><td>0.013013698630136987</td><td>0.022602739726027398</td><td>0.009589041095890411</td><td>0.031506849315068496</td><td>0.036986301369863014</td><td>0.026027397260273973      </td><td>0.8602739726027397 </td><td>0.0006849315068493151</td><td>0.9780821917808219 </td><td>0.012328767123287671</td><td>0.004794520547945206</td><td>0.0013698630136986301</td><td>0.0027397260273972603</td><td>0.5075342465753425</td><td>0.03356164383561644</td><td>0.16506849315068492</td><td>0.0006849315068493151</td><td>0.29315068493150687</td><td>0.06506849315068493</td><td>0.934931506849315  </td><td>0.06438356164383562</td><td>0.018493150684931507</td><td>0.002054794520547945</td><td>0.0006849315068493151</td><td>0.9136986301369863</td><td>0.0006849315068493151   </td><td>0.0684931506849315</td><td>0.02671232876712329</td><td>0.40136986301369865</td><td>0.5034246575342466</td><td>0.009589041095890411</td><td>0.003424657534246575</td><td>0.021232876712328767</td><td>0.023287671232876714</td><td>0.010273972602739725</td><td>0.0006849315068493151</td><td>0.9315068493150684</td><td>0.01643835616438356</td><td>0.022602739726027398</td><td>0.2602739726027397</td><td>0.0136986301369863</td><td>0.21438356164383562</td><td>0.4726027397260274       </td><td>0.00410958904109589</td><td>0.5958904109589042 </td><td>0.013013698630136987</td><td>0.06027397260273973 </td><td>0.0061643835616438354</td><td>0.2650684931506849 </td><td>0.05547945205479452     </td><td>0.2410958904109589 </td><td>0.28904109589041094</td><td>0.05547945205479452       </td><td>0.4143835616438356</td><td>0.002054794520547945</td><td>0.03287671232876712</td><td>0.009589041095890411</td><td>0.002054794520547945</td><td>0.897945205479452  </td><td>0.05547945205479452     </td><td>0.0013698630136986301</td><td>0.023972602739726026</td><td>0.0061643835616438354</td><td>0.004794520547945206</td><td>0.9082191780821918 </td><td>0.05547945205479452     </td><td>0.06164383561643835</td><td>0.02054794520547945</td><td>0.9178082191780822 </td><td>0.0013698630136986301</td><td>0.0013698630136986301</td><td>0.002054794520547945</td><td>0.9952054794520548  </td><td>0.04041095890410959</td><td>0.036986301369863014</td><td>0.10753424657534247</td><td>0.007534246575342466</td><td>0.8075342465753425 </td><td>0.0013698630136986301</td><td>0.0013698630136986301</td><td>0.03356164383561644</td><td>0.0006849315068493151</td><td>0.963013698630137        </td><td>0.02945205479452055</td><td>0.0027397260273972603</td><td>0.0013698630136986301</td><td>0.0061643835616438354</td><td>0.003424657534246575</td><td>0.003424657534246575</td><td>0.08356164383561644</td><td>0.002054794520547945</td><td>0.8678082191780822</td><td>0.06917808219178082    </td><td>0.0027397260273972603  </td><td>0.00821917808219178   </td><td>0.0136986301369863    </td><td>0.8205479452054795    </td><td>0.08561643835616438    </td></tr>
<tr><td>maxs   </td><td>1.730864877400689     </td><td>3.147672552810648     </td><td>7.369662271535608    </td><td>20.51827315777325      </td><td>2.821425316152009     </td><td>3.0785702794481655    </td><td>1.2828389943341163    </td><td>1.2178427314346332   </td><td>8.285201088776468    </td><td>11.405752927121783    </td><td>8.851637899984492     </td><td>4.004295000518199     </td><td>11.520949229255583   </td><td>9.132681280322796    </td><td>3.9369627591698944    </td><td>11.64774940778996     </td><td>7.855574356961303      </td><td>4.963359178355918     </td><td>8.138679732461972    </td><td>2.605521880034986     </td><td>3.2167914250247285   </td><td>6.294997338292252     </td><td>8.86861196453301      </td><td>4.604888881772236     </td><td>3.703937778204484    </td><td>0.3114602557257217    </td><td>2.988889235402396    </td><td>4.421526004152803      </td><td>6.087634999939538   </td><td>7.554198174442678    </td><td>8.67530910373841     </td><td>17.217232781030106    </td><td>8.341461781978069     </td><td>18.306180183312676   </td><td>31.16526797399778     </td><td>2.100892394960775    </td><td>1.6452097110066422   </td><td>13.534473028231162 </td><td>1.0                </td><td>1.0                </td><td>1.0                 </td><td>1.0               </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                 </td><td>1.0                </td><td>1.0               </td><td>1.0                 </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                 </td><td>1.0                </td><td>1.0                 </td><td>1.0                  </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                  </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                 </td><td>1.0                   </td><td>1.0                   </td><td>1.0                  </td><td>1.0                   </td><td>1.0                   </td><td>1.0                   </td><td>1.0                   </td><td>1.0                   </td><td>1.0                   </td><td>1.0                  </td><td>1.0                   </td><td>1.0                   </td><td>1.0                 </td><td>1.0                   </td><td>1.0                  </td><td>1.0                   </td><td>1.0                   </td><td>1.0                   </td><td>1.0                 </td><td>1.0                  </td><td>1.0                   </td><td>1.0                   </td><td>1.0                   </td><td>1.0                  </td><td>1.0                   </td><td>1.0                </td><td>1.0                </td><td>1.0              </td><td>1.0                 </td><td>1.0                 </td><td>1.0                 </td><td>1.0                </td><td>1.0                  </td><td>1.0                 </td><td>1.0                  </td><td>1.0                </td><td>1.0                </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0               </td><td>1.0                 </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                 </td><td>1.0                </td><td>1.0                 </td><td>1.0                 </td><td>1.0                </td><td>1.0                 </td><td>1.0                </td><td>1.0                 </td><td>1.0                </td><td>1.0                 </td><td>1.0               </td><td>1.0                 </td><td>1.0                  </td><td>1.0                  </td><td>1.0               </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                 </td><td>1.0                 </td><td>1.0                </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                 </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                 </td><td>1.0                 </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                 </td><td>1.0                 </td><td>1.0               </td><td>1.0                </td><td>1.0                     </td><td>1.0                </td><td>1.0                 </td><td>1.0                </td><td>1.0                </td><td>1.0                 </td><td>1.0                 </td><td>1.0              </td><td>1.0                  </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                 </td><td>1.0                </td><td>1.0                 </td><td>1.0                </td><td>1.0                </td><td>1.0                   </td><td>1.0                 </td><td>1.0                </td><td>1.0                  </td><td>1.0                </td><td>1.0                   </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                       </td><td>1.0               </td><td>1.0                </td><td>1.0               </td><td>1.0                 </td><td>1.0                </td><td>1.0                       </td><td>1.0               </td><td>1.0                 </td><td>1.0                 </td><td>1.0                 </td><td>1.0                 </td><td>1.0                 </td><td>1.0                       </td><td>1.0                </td><td>1.0                  </td><td>1.0                </td><td>1.0                 </td><td>1.0                 </td><td>1.0                  </td><td>1.0                  </td><td>1.0               </td><td>1.0                </td><td>1.0                </td><td>1.0                  </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                 </td><td>1.0                 </td><td>1.0                  </td><td>1.0               </td><td>1.0                     </td><td>1.0               </td><td>1.0                </td><td>1.0                </td><td>1.0               </td><td>1.0                 </td><td>1.0                 </td><td>1.0                 </td><td>1.0                 </td><td>1.0                 </td><td>1.0                  </td><td>1.0               </td><td>1.0                </td><td>1.0                 </td><td>1.0               </td><td>1.0               </td><td>1.0                </td><td>1.0                      </td><td>1.0                </td><td>1.0                </td><td>1.0                 </td><td>1.0                 </td><td>1.0                  </td><td>1.0                </td><td>1.0                     </td><td>1.0                </td><td>1.0                </td><td>1.0                       </td><td>1.0               </td><td>1.0                 </td><td>1.0                </td><td>1.0                 </td><td>1.0                 </td><td>1.0                </td><td>1.0                     </td><td>1.0                  </td><td>1.0                 </td><td>1.0                  </td><td>1.0                 </td><td>1.0                </td><td>1.0                     </td><td>1.0                </td><td>1.0                </td><td>1.0                </td><td>1.0                  </td><td>1.0                  </td><td>1.0                 </td><td>1.0                 </td><td>1.0                </td><td>1.0                 </td><td>1.0                </td><td>1.0                 </td><td>1.0                </td><td>1.0                  </td><td>1.0                  </td><td>1.0                </td><td>1.0                  </td><td>1.0                      </td><td>1.0                </td><td>1.0                  </td><td>1.0                  </td><td>1.0                  </td><td>1.0                 </td><td>1.0                 </td><td>1.0                </td><td>1.0                 </td><td>1.0               </td><td>1.0                    </td><td>1.0                    </td><td>1.0                   </td><td>1.0                   </td><td>1.0                   </td><td>1.0                    </td></tr>
<tr><td>sigma  </td><td>1.0003426417780727    </td><td>1.000342641778087     </td><td>1.000342641778087    </td><td>1.0003426417780867     </td><td>1.0003426417780865    </td><td>1.0003426417780863    </td><td>1.0003426417780872    </td><td>1.000342641778086    </td><td>1.0003426417780865   </td><td>1.000342641778087     </td><td>1.0003426417780859    </td><td>1.0003426417780874    </td><td>1.0003426417780874   </td><td>1.0003426417780856   </td><td>1.0003426417780856    </td><td>1.0003426417780859    </td><td>1.0003426417780867     </td><td>1.0003426417780856    </td><td>1.0003426417780867   </td><td>1.0003426417780872    </td><td>1.0003426417780874   </td><td>1.0003426417780863    </td><td>1.0003426417780876    </td><td>1.0003426417780872    </td><td>1.0003426417780867   </td><td>1.0003426417780867    </td><td>1.0003426417780867   </td><td>1.0003426417780865     </td><td>1.0003426417780874  </td><td>1.0003426417780863   </td><td>1.000342641778086    </td><td>1.0003426417780874    </td><td>1.0003426417780865    </td><td>1.000342641778087    </td><td>1.0003426417780865    </td><td>1.0003426417780867   </td><td>1.0003426417780863   </td><td>0.39945186826116397</td><td>0.08250493515184774</td><td>0.2063192299887911 </td><td>0.1041452613358008  </td><td>0.4086136097920373</td><td>0.3565208808722697 </td><td>0.06399613628755287</td><td>0.06399613628755287</td><td>0.18192420454133193</td><td>0.1652643014480002  </td><td>0.24183454966521173</td><td>0.4709160752392985</td><td>0.1652643014480002  </td><td>0.08250493515184774</td><td>0.48199627681654716</td><td>0.20326584228129488</td><td>0.18192420454133193</td><td>0.15513227848974456 </td><td>0.30282407981351067</td><td>0.026171196129510684</td><td>0.026171196129510684 </td><td>0.3844331901986958 </td><td>0.24551905432223395</td><td>0.17656960117334067</td><td>0.052288551278964694 </td><td>0.44888374357504446</td><td>0.22495582798808414</td><td>0.2063192299887911 </td><td>0.09397273522657877 </td><td>0.10731330056785437   </td><td>0.03699897442192923   </td><td>0.1041452613358008   </td><td>0.19538172056590825   </td><td>0.13719757994034523   </td><td>0.3037226661175691    </td><td>0.18366927460422675   </td><td>0.2526768070794214    </td><td>0.22631133738630374   </td><td>0.15721690348369968  </td><td>0.10731330056785437   </td><td>0.18015962660076157   </td><td>0.36117748040938086 </td><td>0.07829803935206678   </td><td>0.2180196242212594   </td><td>0.1652643014480002    </td><td>0.2235900014683294    </td><td>0.2673123019064821    </td><td>0.12977525204759144 </td><td>0.21942868535465537  </td><td>0.19698855734166057   </td><td>0.23552581655545554   </td><td>0.12977525204759144   </td><td>0.15927129919468277  </td><td>0.08650206227032596   </td><td>0.17837496183722434</td><td>0.22899213628605464</td><td>0.343950699128721</td><td>0.07384553272149193 </td><td>0.11337171517897796 </td><td>0.08650206227032596 </td><td>0.132299187130019  </td><td>0.03699897442192923  </td><td>0.05844029842502525 </td><td>0.03699897442192923  </td><td>0.06399613628755287</td><td>0.10087312592169323</td><td>0.026171196129510684 </td><td>0.03699897442192923  </td><td>0.026171196129510684 </td><td>0.026171196129510684 </td><td>0.03699897442192923  </td><td>0.3707502583706333</td><td>0.14420917339109338 </td><td>0.18539538765531705</td><td>0.16912783093446626</td><td>0.26839281276605215</td><td>0.30727545826828295</td><td>0.09748641291124244 </td><td>0.5001638121620734 </td><td>0.07384553272149193 </td><td>0.08650206227032596 </td><td>0.46047807081921965</td><td>0.15721690348369968 </td><td>0.2063192299887911 </td><td>0.09397273522657877 </td><td>0.41336536946308766</td><td>0.08650206227032596 </td><td>0.3970205541297276</td><td>0.06909995321300888 </td><td>0.03699897442192923  </td><td>0.026171196129510684 </td><td>0.132299187130019 </td><td>0.026171196129510684 </td><td>0.026171196129510684 </td><td>0.026171196129510684 </td><td>0.08650206227032596 </td><td>0.05844029842502525 </td><td>0.06399613628755287</td><td>0.1162765587141946   </td><td>0.026171196129510684 </td><td>0.03699897442192923  </td><td>0.18192420454133193  </td><td>0.026171196129510684 </td><td>0.20015650124786     </td><td>0.35919702888945876  </td><td>0.026171196129510684 </td><td>0.3578640801216643   </td><td>0.26181597964892594  </td><td>0.03699897442192923  </td><td>0.12977525204759144 </td><td>0.47798629543783094  </td><td>0.34823971970756445  </td><td>0.132299187130019    </td><td>0.1162765587141946   </td><td>0.04529876161967097  </td><td>0.06909995321300888  </td><td>0.12977525204759144  </td><td>0.026171196129510684 </td><td>0.19858002872992706  </td><td>0.34894472251561093  </td><td>0.08250493515184774  </td><td>0.3538032630517682   </td><td>0.026171196129510684 </td><td>0.2964133798808208   </td><td>0.05844029842502525 </td><td>0.132299187130019   </td><td>0.4755981323768334   </td><td>0.3417674629517527   </td><td>0.15927129919468277  </td><td>0.10087312592169323 </td><td>0.46047807081921965 </td><td>0.491672510257295 </td><td>0.2829130777994895 </td><td>0.07384553272149193     </td><td>0.18539538765531705</td><td>0.09748641291124244 </td><td>0.47188803927697087</td><td>0.48541691671397397</td><td>0.04529876161967097 </td><td>0.13719757994034523 </td><td>0.300102792533426</td><td>0.026171196129510684 </td><td>0.32730296526920427</td><td>0.300102792533426  </td><td>0.4958274639890749 </td><td>0.4969278524159491 </td><td>0.12719755605884858</td><td>0.06399613628755287</td><td>0.04529876161967097 </td><td>0.2757902444795971 </td><td>0.153016188726593   </td><td>0.49424946678798465</td><td>0.4970827607112045 </td><td>0.15721690348369968   </td><td>0.17289419145362608 </td><td>0.2063192299887911 </td><td>0.03699897442192923  </td><td>0.30282407981351067</td><td>0.15721690348369968   </td><td>0.3585318286496777 </td><td>0.28881522758337996</td><td>0.26839281276605215</td><td>0.47626246579321585</td><td>0.15927129919468277       </td><td>0.3578640801216643</td><td>0.3019212642574394 </td><td>0.4521868479292668</td><td>0.21942868535465537 </td><td>0.28784401782665164</td><td>0.15721690348369968       </td><td>0.4559831204741484</td><td>0.11337171517897796 </td><td>0.14868422129508393 </td><td>0.09748641291124244 </td><td>0.1747429031496555  </td><td>0.18879281688676075 </td><td>0.15927129919468277       </td><td>0.34682135413890475</td><td>0.026171196129510684 </td><td>0.14646539225648492</td><td>0.11038620653599346 </td><td>0.06909995321300888 </td><td>0.03699897442192923  </td><td>0.052288551278964694 </td><td>0.500114533343848 </td><td>0.18015962660076157</td><td>0.3713695575662699 </td><td>0.026171196129510684 </td><td>0.4553629158067877 </td><td>0.24673119063521753</td><td>0.24673119063521753</td><td>0.24551905432223395</td><td>0.13477238174450887 </td><td>0.04529876161967097 </td><td>0.026171196129510684 </td><td>0.2809047697148763</td><td>0.026171196129510684    </td><td>0.2526768070794214</td><td>0.1612966206958389 </td><td>0.49034353141779197</td><td>0.5001595884536151</td><td>0.09748641291124244 </td><td>0.05844029842502525 </td><td>0.14420917339109338 </td><td>0.1508673102099356  </td><td>0.10087312592169323 </td><td>0.026171196129510684 </td><td>0.2526768070794214</td><td>0.12719755605884858</td><td>0.14868422129508393 </td><td>0.4389343834976037</td><td>0.1162765587141946</td><td>0.4105346362951587 </td><td>0.49941988937230464      </td><td>0.06399613628755287</td><td>0.4908870308779153 </td><td>0.11337171517897796 </td><td>0.23807528775463221 </td><td>0.07829803935206678  </td><td>0.4415209033811093 </td><td>0.22899213628605464     </td><td>0.42789492744648916</td><td>0.45347236758818726</td><td>0.22899213628605464       </td><td>0.4927840821439425</td><td>0.04529876161967097 </td><td>0.17837496183722434</td><td>0.09748641291124244 </td><td>0.04529876161967097 </td><td>0.30282407981351067</td><td>0.22899213628605464     </td><td>0.03699897442192923  </td><td>0.153016188726593   </td><td>0.07829803935206678  </td><td>0.06909995321300888 </td><td>0.28881522758337996</td><td>0.22899213628605464     </td><td>0.24058994034059597</td><td>0.14191378138156566</td><td>0.27475078906920986</td><td>0.03699897442192923  </td><td>0.03699897442192923  </td><td>0.04529876161967097 </td><td>0.06909995321300888 </td><td>0.19698855734166057</td><td>0.18879281688676075 </td><td>0.3098974197252131 </td><td>0.08650206227032596 </td><td>0.39437192349642425</td><td>0.03699897442192923  </td><td>0.03699897442192923  </td><td>0.18015962660076157</td><td>0.026171196129510684 </td><td>0.18879281688676075      </td><td>0.16912783093446626</td><td>0.052288551278964694 </td><td>0.03699897442192923  </td><td>0.07829803935206678  </td><td>0.05844029842502525 </td><td>0.05844029842502525 </td><td>0.2768241010111916 </td><td>0.04529876161967097 </td><td>0.3388152020275898</td><td>0.2538436719721309     </td><td>0.052288551278964694   </td><td>0.09031727589007567   </td><td>0.1162765587141946    </td><td>0.38386187547163386   </td><td>0.27989269608141876    </td></tr>
<tr><td>zeros  </td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                      </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                    </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                    </td><td>0                     </td><td>0                     </td><td>0                      </td><td>0                     </td><td>0                    </td><td>0                     </td><td>0                    </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                     </td><td>0                    </td><td>0                      </td><td>0                   </td><td>0                    </td><td>0                    </td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                     </td><td>0                    </td><td>0                    </td><td>0                  </td><td>1450               </td><td>1395               </td><td>1444                </td><td>309               </td><td>1242               </td><td>1454               </td><td>6                  </td><td>1410               </td><td>1419                </td><td>91                 </td><td>976               </td><td>1419                </td><td>1450               </td><td>535                </td><td>1397               </td><td>1410               </td><td>1424                </td><td>149                </td><td>1                   </td><td>1459                 </td><td>1197               </td><td>1366               </td><td>1413               </td><td>1456                 </td><td>408                </td><td>78                 </td><td>1395               </td><td>1447                </td><td>1443                  </td><td>1458                  </td><td>1444                 </td><td>1402                  </td><td>1432                  </td><td>1310                  </td><td>1409                  </td><td>1360                  </td><td>1381                  </td><td>1423                 </td><td>1443                  </td><td>1411                  </td><td>1235                </td><td>1451                  </td><td>1387                 </td><td>1419                  </td><td>1383                  </td><td>1347                  </td><td>1435                </td><td>1386                 </td><td>1401                  </td><td>1374                  </td><td>1435                  </td><td>1422                 </td><td>1449                  </td><td>1412               </td><td>1379               </td><td>200              </td><td>1452                </td><td>1441                </td><td>1449                </td><td>1434               </td><td>1458                 </td><td>1455                </td><td>1458                 </td><td>1454               </td><td>15                 </td><td>1459                 </td><td>1458                 </td><td>1459                 </td><td>1459                 </td><td>1458                 </td><td>240               </td><td>1429                </td><td>1408               </td><td>1417               </td><td>1346               </td><td>1306               </td><td>1446                </td><td>734                </td><td>1452                </td><td>1449                </td><td>1015               </td><td>1423                </td><td>1395               </td><td>1447                </td><td>319                </td><td>1449                </td><td>1174              </td><td>1453                </td><td>1458                 </td><td>1459                 </td><td>26                </td><td>1459                 </td><td>1459                 </td><td>1459                 </td><td>1449                </td><td>1455                </td><td>1454               </td><td>1440                 </td><td>1459                 </td><td>1458                 </td><td>1410                 </td><td>1459                 </td><td>1399                 </td><td>1238                 </td><td>1459                 </td><td>1240                 </td><td>1352                 </td><td>1458                 </td><td>1435                </td><td>945                  </td><td>1254                 </td><td>1434                 </td><td>1440                 </td><td>1457                 </td><td>1453                 </td><td>1435                 </td><td>1459                 </td><td>1400                 </td><td>1253                 </td><td>1450                 </td><td>1246                 </td><td>1459                 </td><td>1318                 </td><td>1455                </td><td>1434                </td><td>956                  </td><td>1263                 </td><td>1422                 </td><td>1445                </td><td>1015                </td><td>596               </td><td>1332               </td><td>1452                    </td><td>1408               </td><td>1446                </td><td>972                </td><td>554                </td><td>1457                </td><td>1432                </td><td>1314             </td><td>1459                 </td><td>178                </td><td>1314               </td><td>826                </td><td>813                </td><td>1436               </td><td>1454               </td><td>1457                </td><td>1339               </td><td>1425                </td><td>842                </td><td>811                </td><td>1423                  </td><td>1415                </td><td>1395               </td><td>1458                 </td><td>149                </td><td>1423                  </td><td>1239               </td><td>1326               </td><td>1346               </td><td>507                </td><td>1422                      </td><td>1240              </td><td>1312               </td><td>1042              </td><td>1386                </td><td>1327               </td><td>1423                      </td><td>1030              </td><td>1441                </td><td>1427                </td><td>1446                </td><td>1414                </td><td>1406                </td><td>1422                      </td><td>204                </td><td>1459                 </td><td>32                 </td><td>1442                </td><td>1453                </td><td>1458                 </td><td>1456                 </td><td>719               </td><td>1411               </td><td>1219               </td><td>1459                 </td><td>1032               </td><td>1365               </td><td>95                 </td><td>1366               </td><td>1433                </td><td>1457                </td><td>1459                 </td><td>126               </td><td>1459                    </td><td>1360              </td><td>1421               </td><td>874                </td><td>725               </td><td>1446                </td><td>1455                </td><td>1429                </td><td>1426                </td><td>1445                </td><td>1459                 </td><td>100               </td><td>1436               </td><td>1427                </td><td>1080              </td><td>1440              </td><td>1147               </td><td>770                      </td><td>1454               </td><td>590                </td><td>1441                </td><td>1372                </td><td>1451                 </td><td>1073               </td><td>1379                    </td><td>1108               </td><td>1038               </td><td>1379                      </td><td>855               </td><td>1457                </td><td>1412               </td><td>1446                </td><td>1457                </td><td>149                </td><td>1379                    </td><td>1458                 </td><td>1425                </td><td>1451                 </td><td>1453                </td><td>134                </td><td>1379                    </td><td>1370               </td><td>1430               </td><td>120                </td><td>1458                 </td><td>1458                 </td><td>1457                </td><td>7                   </td><td>1401               </td><td>1406                </td><td>1303               </td><td>1449                </td><td>281                </td><td>1458                 </td><td>1458                 </td><td>1411               </td><td>1459                 </td><td>54                       </td><td>1417               </td><td>1456                 </td><td>1458                 </td><td>1451                 </td><td>1455                </td><td>1455                </td><td>1338               </td><td>1457                </td><td>193               </td><td>1359                   </td><td>1456                   </td><td>1448                  </td><td>1440                  </td><td>262                   </td><td>1335                   </td></tr>
<tr><td>missing</td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                      </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                    </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                    </td><td>0                     </td><td>0                     </td><td>0                      </td><td>0                     </td><td>0                    </td><td>0                     </td><td>0                    </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                     </td><td>0                    </td><td>0                      </td><td>0                   </td><td>0                    </td><td>0                    </td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                     </td><td>0                    </td><td>0                    </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                   </td><td>0                 </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                   </td><td>0                  </td><td>0                 </td><td>0                   </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                   </td><td>0                  </td><td>0                   </td><td>0                    </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                    </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                   </td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                     </td><td>0                     </td><td>0                   </td><td>0                     </td><td>0                    </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                   </td><td>0                    </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                    </td><td>0                     </td><td>0                  </td><td>0                  </td><td>0                </td><td>0                   </td><td>0                   </td><td>0                   </td><td>0                  </td><td>0                    </td><td>0                   </td><td>0                    </td><td>0                  </td><td>0                  </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                 </td><td>0                   </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                   </td><td>0                  </td><td>0                   </td><td>0                   </td><td>0                  </td><td>0                   </td><td>0                  </td><td>0                   </td><td>0                  </td><td>0                   </td><td>0                 </td><td>0                   </td><td>0                    </td><td>0                    </td><td>0                 </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                   </td><td>0                   </td><td>0                  </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                   </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                   </td><td>0                   </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                   </td><td>0                   </td><td>0                 </td><td>0                  </td><td>0                       </td><td>0                  </td><td>0                   </td><td>0                  </td><td>0                  </td><td>0                   </td><td>0                   </td><td>0                </td><td>0                    </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                   </td><td>0                  </td><td>0                   </td><td>0                  </td><td>0                  </td><td>0                     </td><td>0                   </td><td>0                  </td><td>0                    </td><td>0                  </td><td>0                     </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                         </td><td>0                 </td><td>0                  </td><td>0                 </td><td>0                   </td><td>0                  </td><td>0                         </td><td>0                 </td><td>0                   </td><td>0                   </td><td>0                   </td><td>0                   </td><td>0                   </td><td>0                         </td><td>0                  </td><td>0                    </td><td>0                  </td><td>0                   </td><td>0                   </td><td>0                    </td><td>0                    </td><td>0                 </td><td>0                  </td><td>0                  </td><td>0                    </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                   </td><td>0                   </td><td>0                    </td><td>0                 </td><td>0                       </td><td>0                 </td><td>0                  </td><td>0                  </td><td>0                 </td><td>0                   </td><td>0                   </td><td>0                   </td><td>0                   </td><td>0                   </td><td>0                    </td><td>0                 </td><td>0                  </td><td>0                   </td><td>0                 </td><td>0                 </td><td>0                  </td><td>0                        </td><td>0                  </td><td>0                  </td><td>0                   </td><td>0                   </td><td>0                    </td><td>0                  </td><td>0                       </td><td>0                  </td><td>0                  </td><td>0                         </td><td>0                 </td><td>0                   </td><td>0                  </td><td>0                   </td><td>0                   </td><td>0                  </td><td>0                       </td><td>0                    </td><td>0                   </td><td>0                    </td><td>0                   </td><td>0                  </td><td>0                       </td><td>0                  </td><td>0                  </td><td>0                  </td><td>0                    </td><td>0                    </td><td>0                   </td><td>0                   </td><td>0                  </td><td>0                   </td><td>0                  </td><td>0                   </td><td>0                  </td><td>0                    </td><td>0                    </td><td>0                  </td><td>0                    </td><td>0                        </td><td>0                  </td><td>0                    </td><td>0                    </td><td>0                    </td><td>0                   </td><td>0                   </td><td>0                  </td><td>0                   </td><td>0                 </td><td>0                      </td><td>0                      </td><td>0                     </td><td>0                     </td><td>0                     </td><td>0                      </td></tr>
<tr><td>0      </td><td>-1.730864877400689    </td><td>0.07337496353744775   </td><td>0.2128771963643249   </td><td>-0.20714170777431132   </td><td>0.6514792433257054    </td><td>-0.5171998069472914   </td><td>1.0509937888999856    </td><td>0.8786680880058696   </td><td>0.5141038909843643   </td><td>0.5754248369676035    </td><td>-0.2886528311122454   </td><td>-0.9445906057378156   </td><td>-0.4593025408311876  </td><td>-0.7934337933349002  </td><td>1.161851587468555     </td><td>-0.12024172373467248  </td><td>0.3703334392167798     </td><td>1.1078101491462131    </td><td>-0.24106103579929675 </td><td>0.7897405221108432    </td><td>1.2275853765130371   </td><td>0.1637791168735145    </td><td>-0.2114535812020489   </td><td>0.9122097711603002    </td><td>-0.9512264882332891  </td><td>0.2960261798262024    </td><td>0.31172464418307205  </td><td>0.35100032086652694    </td><td>-0.7521758378613592 </td><td>0.2165031608388436   </td><td>-0.3593249004055313  </td><td>-0.11633928614822256  </td><td>-0.2702083542015609   </td><td>-0.06869174753820707 </td><td>-0.08768781151769862  </td><td>-1.599111099180035   </td><td>0.1387774889497933   </td><td>12.247694320220994 </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>1.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>1.0              </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0               </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0              </td><td>0.0                  </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                </td><td>0.0                   </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                       </td><td>1.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                     </td><td>0.0               </td><td>0.0                </td><td>1.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0               </td><td>0.0                </td><td>1.0                      </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>1.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                </td><td>0.0                  </td><td>1.0                      </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                    </td><td>0.0                    </td><td>0.0                   </td><td>0.0                   </td><td>1.0                   </td><td>0.0                    </td></tr>
<tr><td>1      </td><td>-1.728492204505006    </td><td>-0.8725627562389217   </td><td>0.6457472613948461   </td><td>-0.09188637231949036   </td><td>-0.07183611428306244  </td><td>2.179627757849301     </td><td>0.15673371079690998   </td><td>-0.4295769652193609  </td><td>-0.5707501337786359  </td><td>1.171992119373828     </td><td>-0.2886528311122454   </td><td>-0.6412279930944917   </td><td>0.4664649160883461   </td><td>0.25714042978945445  </td><td>-0.7951632272572445   </td><td>-0.12024172373467248  </td><td>-0.4825119145852412    </td><td>-0.8199643654586394   </td><td>3.948809348331338    </td><td>0.7897405221108432    </td><td>-0.7616206719986545  </td><td>0.1637791168735145    </td><td>-0.2114535812020489   </td><td>-0.31868326571034533  </td><td>0.6004949339126351   </td><td>0.2364947442137705    </td><td>0.31172464418307205  </td><td>-0.060731012615303295  </td><td>1.6261947918523714  </td><td>-0.704483250333654   </td><td>-0.3593249004055313  </td><td>-0.11633928614822256  </td><td>-0.2702083542015609   </td><td>-0.06869174753820707 </td><td>-0.08768781151769862  </td><td>-0.489110050937792   </td><td>-0.6144386220786311  </td><td>12.109010932687042 </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>1.0                   </td><td>0.0                </td><td>1.0                </td><td>0.0              </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0              </td><td>0.0                  </td><td>1.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                </td><td>0.0                   </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                   </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                       </td><td>1.0               </td><td>0.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                       </td><td>1.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                     </td><td>0.0               </td><td>0.0                </td><td>0.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0               </td><td>1.0                </td><td>0.0                      </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>1.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                </td><td>0.0                  </td><td>1.0                      </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                    </td><td>0.0                    </td><td>0.0                   </td><td>0.0                   </td><td>1.0                   </td><td>0.0                    </td></tr>
<tr><td>2      </td><td>-1.7261195316093232   </td><td>0.07337496353744775   </td><td>0.29945120937042913  </td><td>0.07347997855047012    </td><td>0.6514792433257054    </td><td>-0.5171998069472914   </td><td>0.9847523016330912    </td><td>0.8302145675160463   </td><td>0.32591492750506834  </td><td>0.09290718208021594   </td><td>-0.2886528311122454   </td><td>-0.3016429789415173   </td><td>-0.3133687545187488  </td><td>-0.6278260340246571  </td><td>1.1893506246778172    </td><td>-0.12024172373467248  </td><td>0.5150125617367655     </td><td>1.1078101491462131    </td><td>-0.24106103579929675 </td><td>0.7897405221108432    </td><td>1.2275853765130371   </td><td>0.1637791168735145    </td><td>-0.2114535812020489   </td><td>-0.31868326571034533  </td><td>0.6004949339126351   </td><td>0.29161644385491114   </td><td>0.31172464418307205  </td><td>0.6317262300586839     </td><td>-0.7521758378613592 </td><td>-0.07036145903455729 </td><td>-0.3593249004055313  </td><td>-0.11633928614822256  </td><td>-0.2702083542015609   </td><td>-0.06869174753820707 </td><td>-0.08768781151769862  </td><td>0.9908913467185322   </td><td>0.1387774889497933   </td><td>12.31716669303576  </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>1.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>1.0              </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0               </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0              </td><td>0.0                  </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                </td><td>0.0                   </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                       </td><td>1.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                     </td><td>0.0               </td><td>0.0                </td><td>1.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0               </td><td>1.0                </td><td>0.0                      </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>1.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                </td><td>0.0                  </td><td>1.0                      </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                    </td><td>0.0                    </td><td>0.0                   </td><td>0.0                   </td><td>1.0                   </td><td>0.0                    </td></tr>
<tr><td>3      </td><td>-1.7237468587136402   </td><td>0.3098593934815401    </td><td>0.0685871746874845   </td><td>-0.09689747386100432   </td><td>0.6514792433257054    </td><td>-0.5171998069472914   </td><td>-1.863631650843372    </td><td>-0.720298088158301   </td><td>-0.5707501337786359  </td><td>-0.4992735761906688   </td><td>-0.2886528311122454   </td><td>-0.06166956894008201  </td><td>-0.6873240819443732  </td><td>-0.5217335632165325  </td><td>0.9372761169262503    </td><td>-0.12024172373467248  </td><td>0.3836591478699364     </td><td>1.1078101491462131    </td><td>-0.24106103579929675 </td><td>-1.0260408358133      </td><td>-0.7616206719986545  </td><td>0.1637791168735145    </td><td>-0.2114535812020489   </td><td>0.29676325272497744   </td><td>0.6004949339126351   </td><td>0.28500183989797423   </td><td>1.650306939792734    </td><td>0.7908042452675729     </td><td>-0.7521758378613592 </td><td>-0.1760484242510734  </td><td>4.092523739317571    </td><td>-0.11633928614822256  </td><td>-0.2702083542015609   </td><td>-0.06869174753820707 </td><td>-0.08768781151769862  </td><td>-1.599111099180035   </td><td>-1.3676547331070554  </td><td>11.84939770159144  </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0                 </td><td>0.0                  </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>1.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>1.0              </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0                  </td><td>0.0                 </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0              </td><td>0.0                  </td><td>1.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                   </td><td>0.0                 </td><td>1.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                       </td><td>1.0               </td><td>0.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                       </td><td>1.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0               </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                     </td><td>0.0               </td><td>0.0                </td><td>1.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0               </td><td>0.0                </td><td>0.0                      </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>0.0                       </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                </td><td>0.0                  </td><td>1.0                      </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>1.0                    </td><td>0.0                    </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                    </td></tr>
<tr><td>4      </td><td>-1.7213741858179572   </td><td>0.07337496353744775   </td><td>0.7611792787363184   </td><td>0.37514829134961014    </td><td>1.3747946009344731    </td><td>-0.5171998069472914   </td><td>0.951631557999644     </td><td>0.7333075265363995   </td><td>1.366489196155293    </td><td>0.4635684715164364    </td><td>-0.2886528311122454   </td><td>-0.17486457365774016  </td><td>0.1996797129859189   </td><td>-0.045611255199583686</td><td>1.6178772878554806    </td><td>-0.12024172373467248  </td><td>1.299325699608267      </td><td>1.1078101491462131    </td><td>-0.24106103579929675 </td><td>0.7897405221108432    </td><td>1.2275853765130371   </td><td>1.3900227611572622    </td><td>-0.2114535812020489   </td><td>1.527656289595623     </td><td>0.6004949339126351   </td><td>0.28941157586926547   </td><td>1.650306939792734    </td><td>1.6984846849888804     </td><td>0.7801971853099035  </td><td>0.5637603322645395   </td><td>-0.3593249004055313  </td><td>-0.11633928614822256  </td><td>-0.2702083542015609   </td><td>-0.06869174753820707 </td><td>-0.08768781151769862  </td><td>2.100892394960775    </td><td>0.1387774889497933   </td><td>12.429216196844385 </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                   </td><td>0.0                  </td><td>1.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>1.0              </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0               </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0              </td><td>0.0                  </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                </td><td>0.0                   </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                   </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                       </td><td>1.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                     </td><td>0.0               </td><td>0.0                </td><td>1.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0               </td><td>1.0                </td><td>0.0                      </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>1.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                </td><td>0.0                  </td><td>1.0                      </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                    </td><td>0.0                    </td><td>0.0                   </td><td>0.0                   </td><td>1.0                   </td><td>0.0                    </td></tr>
<tr><td>5      </td><td>-1.7190015129222744   </td><td>-0.1631094664066446   </td><td>0.7900372830716864   </td><td>0.3606160968792197     </td><td>-0.7951514718918303   </td><td>-0.5171998069472914   </td><td>0.7197863525655132    </td><td>0.4910399240872828   </td><td>-0.5707501337786359  </td><td>0.632449650727022     </td><td>-0.2886528311122454   </td><td>-1.1392860138521876   </td><td>-0.596115465499099   </td><td>-0.9486910676882532  </td><td>0.5018746944462716    </td><td>-0.12024172373467248  </td><td>-0.2921446481115758    </td><td>1.1078101491462131    </td><td>-0.24106103579929675 </td><td>-1.0260408358133      </td><td>1.2275853765130371   </td><td>-2.288708171693981    </td><td>-0.2114535812020489   </td><td>-0.934129784145668    </td><td>-0.9512264882332891  </td><td>0.27397749996974613   </td><td>0.31172464418307205  </td><td>0.03284429044874903    </td><td>-0.4329314580340128 </td><td>-0.25153911369144205 </td><td>-0.3593249004055313  </td><td>10.802446267979857    </td><td>-0.2702083542015609   </td><td>-0.06869174753820707 </td><td>1.3237359981507422    </td><td>1.3608916961326132   </td><td>0.8919935999782177   </td><td>11.870599909242044 </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>1.0                   </td><td>0.0                 </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>1.0              </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0              </td><td>0.0                  </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                </td><td>0.0                   </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                       </td><td>1.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                     </td><td>0.0               </td><td>0.0                </td><td>0.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0               </td><td>0.0                </td><td>1.0                      </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>0.0                       </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>1.0                </td><td>0.0                  </td><td>0.0                      </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                    </td><td>0.0                    </td><td>0.0                   </td><td>0.0                   </td><td>1.0                   </td><td>0.0                    </td></tr>
<tr><td>6      </td><td>-1.7166288400265914   </td><td>-0.8725627562389217   </td><td>0.5014572397180057   </td><td>-0.04337890939763529   </td><td>1.3747946009344731    </td><td>-0.5171998069472914   </td><td>1.084114532533433     </td><td>0.9755751289855163   </td><td>0.4587541958433949   </td><td>2.029557587832776     </td><td>-0.2886528311122454   </td><td>-0.5665192899808374   </td><td>1.4332762504082532   </td><td>1.3749928051335951   </td><td>-0.7951632272572445   </td><td>-0.12024172373467248  </td><td>0.33987467658099335    </td><td>1.1078101491462131    </td><td>-0.24106103579929675 </td><td>0.7897405221108432    </td><td>-0.7616206719986545  </td><td>0.1637791168735145    </td><td>-0.2114535812020489   </td><td>0.29676325272497744   </td><td>0.6004949339126351   </td><td>0.298231047811848     </td><td>0.31172464418307205  </td><td>0.7627316543483571     </td><td>1.283007083537974   </td><td>0.15611060928654868  </td><td>-0.3593249004055313  </td><td>-0.11633928614822256  </td><td>-0.2702083542015609   </td><td>-0.06869174753820707 </td><td>-0.08768781151769862  </td><td>0.6208909973044512   </td><td>-0.6144386220786311  </td><td>12.634603026569334 </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                  </td><td>0.0                   </td><td>1.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>1.0              </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0               </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0              </td><td>0.0                  </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                   </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                   </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                       </td><td>1.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                     </td><td>0.0               </td><td>0.0                </td><td>1.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0               </td><td>0.0                </td><td>0.0                      </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>1.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                </td><td>0.0                  </td><td>1.0                      </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                    </td><td>0.0                    </td><td>0.0                   </td><td>0.0                   </td><td>1.0                   </td><td>0.0                    </td></tr>
<tr><td>7      </td><td>-1.7142561671309084   </td><td>0.07337496353744775   </td><td>-1.6628930854346002  </td><td>-0.013512744210212127  </td><td>0.6514792433257054    </td><td>0.3817427146515728    </td><td>0.05737147989656824   </td><td>-0.5749375266888309  </td><td>0.7576425496046296   </td><td>0.9109939333211048    </td><td>-0.09022046788626462  </td><td>-0.7951731995105069   </td><td>0.11303152736290836  </td><td>-0.14394086229004052 </td><td>1.45746623746812      </td><td>-0.12024172373467248  </td><td>1.0937290518167084     </td><td>1.1078101491462131    </td><td>-0.24106103579929675 </td><td>0.7897405221108432    </td><td>1.2275853765130371   </td><td>0.1637791168735145    </td><td>-0.2114535812020489   </td><td>0.29676325272497744   </td><td>2.1522163560585597   </td><td>0.22988014025683365   </td><td>0.31172464418307205  </td><td>0.05155935106155949    </td><td>1.1233848936243007  </td><td>2.3755368788333873   </td><td>3.37237175348001     </td><td>-0.11633928614822256  </td><td>-0.2702083542015609   </td><td>-0.06869174753820707 </td><td>0.6180240933165219    </td><td>1.7308920455466943   </td><td>0.8919935999782177   </td><td>12.206072645530174 </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0                 </td><td>0.0                  </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                   </td><td>1.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>0.0              </td><td>0.0                 </td><td>1.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0               </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0              </td><td>0.0                  </td><td>1.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                </td><td>0.0                   </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                       </td><td>1.0               </td><td>0.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>1.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                       </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                     </td><td>0.0               </td><td>0.0                </td><td>0.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0               </td><td>1.0                </td><td>0.0                      </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>1.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>1.0                </td><td>0.0                  </td><td>0.0                      </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                    </td><td>0.0                    </td><td>0.0                   </td><td>0.0                   </td><td>1.0                   </td><td>0.0                    </td></tr>
<tr><td>8      </td><td>-1.7118834942352257   </td><td>-0.1631094664066446   </td><td>-0.1911348643308282  </td><td>-0.44065903960886155   </td><td>0.6514792433257054    </td><td>-0.5171998069472914   </td><td>-1.3336997527082162   </td><td>-1.689368497954768   </td><td>-0.5707501337786359  </td><td>-0.9730181828073764   </td><td>-0.2886528311122454   </td><td>0.8710572699334213    </td><td>-0.2404018613625294  </td><td>-0.3638886676239571  </td><td>0.9281097711898298    </td><td>-0.12024172373467248  </td><td>0.49216848975992566    </td><td>-0.8199643654586394   </td><td>-0.24106103579929675 </td><td>0.7897405221108432    </td><td>-0.7616206719986545  </td><td>-1.0624645274102331   </td><td>4.32857919166548      </td><td>0.9122097711603002    </td><td>2.1522163560585597   </td><td>0.13727568485971745   </td><td>0.31172464418307205  </td><td>-0.023300891389682367  </td><td>-0.03387598324982984</td><td>-0.704483250333654   </td><td>2.9959286699740124   </td><td>-0.11633928614822256  </td><td>-0.2702083542015609   </td><td>-0.06869174753820707 </td><td>-0.08768781151769862  </td><td>-0.859110400351873   </td><td>0.1387774889497933   </td><td>11.77452020265869  </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>1.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>1.0                   </td><td>0.0                 </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>1.0                </td><td>0.0                </td><td>0.0              </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0                  </td><td>0.0                 </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0              </td><td>0.0                  </td><td>1.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                   </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                       </td><td>1.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                       </td><td>1.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0               </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>1.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0               </td><td>0.0                     </td><td>0.0               </td><td>0.0                </td><td>0.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                 </td><td>1.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0               </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0               </td><td>1.0                </td><td>0.0                      </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>0.0                       </td><td>1.0               </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                     </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                </td><td>0.0                  </td><td>1.0                      </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>1.0                    </td><td>0.0                    </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                    </td></tr>
<tr><td>9      </td><td>-1.709510821339543    </td><td>3.147672552810648     </td><td>-0.21999286866619627 </td><td>-0.31037039952949874   </td><td>-0.7951514718918303   </td><td>0.3817427146515728    </td><td>-1.0687338036406382   </td><td>-1.689368497954768   </td><td>-0.5707501337786359  </td><td>0.8934478367797453    </td><td>-0.2886528311122454   </td><td>-0.9672296066813472   </td><td>-0.151473460328387   </td><td>-0.22156949946671697 </td><td>-0.7951632272572445   </td><td>-0.12024172373467248  </td><td>-0.8346913575615222    </td><td>1.1078101491462131    </td><td>-0.24106103579929675 </td><td>-1.0260408358133      </td><td>-0.7616206719986545  </td><td>-1.0624645274102331   </td><td>4.32857919166548      </td><td>-0.934129784145668    </td><td>2.1522163560585597   </td><td>0.15491462874488243   </td><td>-1.02685765142659    </td><td>-1.2538161266819705    </td><td>-0.7521758378613592 </td><td>-0.644090698781359   </td><td>-0.3593249004055313  </td><td>-0.11633928614822256  </td><td>-0.2702083542015609   </td><td>-0.06869174753820707 </td><td>-0.08768781151769862  </td><td>-1.969111448594116   </td><td>0.1387774889497933   </td><td>11.6784399034478   </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>1.0                 </td><td>0.0                  </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>1.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                 </td><td>0.0                  </td><td>0.0                   </td><td>0.0                   </td><td>0.0                   </td><td>0.0                  </td><td>0.0                   </td><td>1.0                </td><td>0.0                </td><td>0.0              </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                 </td><td>1.0                  </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0               </td><td>1.0                 </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>1.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>1.0               </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0              </td><td>0.0                  </td><td>1.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>1.0                </td><td>0.0                   </td><td>0.0                 </td><td>0.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                   </td><td>0.0                </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                       </td><td>1.0                </td><td>0.0                  </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                </td><td>0.0                  </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                     </td><td>0.0               </td><td>0.0                </td><td>0.0                </td><td>1.0               </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>1.0               </td><td>0.0                </td><td>0.0                 </td><td>0.0               </td><td>0.0               </td><td>1.0                </td><td>0.0                      </td><td>0.0                </td><td>1.0                </td><td>0.0                 </td><td>0.0                 </td><td>0.0                  </td><td>0.0                </td><td>0.0                     </td><td>0.0                </td><td>1.0                </td><td>0.0                       </td><td>0.0               </td><td>0.0                 </td><td>0.0                </td><td>1.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                     </td><td>0.0                  </td><td>0.0                 </td><td>0.0                  </td><td>0.0                 </td><td>1.0                </td><td>0.0                     </td><td>0.0                </td><td>0.0                </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>1.0                 </td><td>0.0                </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                </td><td>0.0                  </td><td>1.0                      </td><td>0.0                </td><td>0.0                  </td><td>0.0                  </td><td>0.0                  </td><td>0.0                 </td><td>0.0                 </td><td>0.0                </td><td>0.0                 </td><td>1.0               </td><td>0.0                    </td><td>0.0                    </td><td>0.0                   </td><td>0.0                   </td><td>1.0                   </td><td>0.0                    </td></tr>
</tbody>
</table>



```python
y = "SalePrice"
splits = df.split_frame(ratios = [0.8], seed = 1)
train_aml = splits[0]
val = splits[1]
```


```python
aml = H2OAutoML(max_runtime_secs = 60, seed = 1, project_name = "Housing Data Analysis")
aml.train(y = y, training_frame = train_aml, leaderboard_frame = val)
```

    AutoML progress: |████████████████████████████████████████████████████████| 100%



```python
aml.leaderboard.head()
```


<table>
<thead>
<tr><th>model_id                                           </th><th style="text-align: right;">  mean_residual_deviance</th><th style="text-align: right;">    rmse</th><th style="text-align: right;">      mse</th><th style="text-align: right;">      mae</th><th style="text-align: right;">     rmsle</th></tr>
</thead>
<tbody>
<tr><td>StackedEnsemble_BestOfFamily_AutoML_20190104_134403</td><td style="text-align: right;">               0.0113005</td><td style="text-align: right;">0.106304</td><td style="text-align: right;">0.0113005</td><td style="text-align: right;">0.0782194</td><td style="text-align: right;">0.00819334</td></tr>
<tr><td>StackedEnsemble_AllModels_AutoML_20190104_134403   </td><td style="text-align: right;">               0.0113005</td><td style="text-align: right;">0.106304</td><td style="text-align: right;">0.0113005</td><td style="text-align: right;">0.0782194</td><td style="text-align: right;">0.00819334</td></tr>
<tr><td>XGBoost_1_AutoML_20190104_134403                   </td><td style="text-align: right;">               0.013727 </td><td style="text-align: right;">0.117162</td><td style="text-align: right;">0.013727 </td><td style="text-align: right;">0.0844481</td><td style="text-align: right;">0.00902482</td></tr>
<tr><td>GLM_grid_1_AutoML_20190104_134403_model_1          </td><td style="text-align: right;">               0.0148006</td><td style="text-align: right;">0.121658</td><td style="text-align: right;">0.0148006</td><td style="text-align: right;">0.086818 </td><td style="text-align: right;">0.00934338</td></tr>
<tr><td>DRF_1_AutoML_20190104_134403                       </td><td style="text-align: right;">               0.017177 </td><td style="text-align: right;">0.131061</td><td style="text-align: right;">0.017177 </td><td style="text-align: right;">0.0939359</td><td style="text-align: right;">0.0101283 </td></tr>
<tr><td>XRT_1_AutoML_20190104_134403                       </td><td style="text-align: right;">               0.0174549</td><td style="text-align: right;">0.132117</td><td style="text-align: right;">0.0174549</td><td style="text-align: right;">0.0979803</td><td style="text-align: right;">0.010192  </td></tr>
</tbody>
</table>





    




```python
pred = aml.predict(val)
pred.head()
```

    stackedensemble prediction progress: |████████████████████████████████████| 100%



<table>
<thead>
<tr><th style="text-align: right;">  predict</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">  12.277 </td></tr>
<tr><td style="text-align: right;">  11.6499</td></tr>
<tr><td style="text-align: right;">  12.7037</td></tr>
<tr><td style="text-align: right;">  11.9552</td></tr>
<tr><td style="text-align: right;">  11.3838</td></tr>
<tr><td style="text-align: right;">  11.7319</td></tr>
<tr><td style="text-align: right;">  12.038 </td></tr>
<tr><td style="text-align: right;">  11.7838</td></tr>
<tr><td style="text-align: right;">  12.3429</td></tr>
<tr><td style="text-align: right;">  12.1643</td></tr>
</tbody>
</table>





    




```python
perf = aml.leader.model_performance(val)
perf
```

    
    ModelMetricsRegressionGLM: stackedensemble
    ** Reported on test data. **
    
    MSE: 0.01130053526428233
    RMSE: 0.10630397576893505
    MAE: 0.07821943720832428
    RMSLE: 0.008193340141979944
    R^2: 0.9324793927504783
    Mean Residual Deviance: 0.01130053526428233
    Null degrees of freedom: 294
    Residual degrees of freedom: 291
    Null deviance: 49.58251098775174
    Residual deviance: 3.333657902963288
    AIC: -475.2832952394578





    


The values in our dependent variable SalePrice are continuous and are too big to predict using regression.  So we will convert our dependent to a multiclass variable and apply classification models to predict the ranges of SalePrice.

```python
#house_test = train_X[pd.DataFrame([set(X.columns).intersection(set(train_X.columns))]).T[0].tolist()]
#X = X[pd.DataFrame([set(X.columns).intersection(set(train_X.columns))]).T[0].tolist()]
```


```python
np.exp(train_X['SalePrice']).describe()
```




    count      1460.000000
    mean     180921.195890
    std       79442.502883
    min       34900.000000
    25%      129975.000000
    50%      163000.000000
    75%      214000.000000
    max      755000.000000
    Name: SalePrice, dtype: float64




```python
bins = [30900, 130000, 163000, 214000, 800000]
names = [1, 2, 3, 4]
```


```python
names1 = ['Cheap','Lower Range', 'Mid-Range', 'Expensive'] 
train_X['SalePrice'] = np.exp(train_X['SalePrice'])
train_X['SalePriceRange'] = train_X['SalePrice']
train_X.SalePriceRange = pd.cut(train_X['SalePriceRange'], bins, labels = names)
```


```python
X_c = train_X.drop(['SalePrice','SalePriceRange'], axis = 1)
y_c = train_X.SalePriceRange
```


```python
X_c = X_c[pd.DataFrame([set(X_c.columns).intersection(set(test_X.columns))]).T[0]]
test_X = test_X[pd.DataFrame([set(test_X.columns).intersection(set(X_c.columns))]).T[0]]
```


```python
X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(X_c, y_c, test_size = 0.2, random_state = 1)
```


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from xgboost import XGBClassifier

from yellowbrick.classifier import ConfusionMatrix
```


```python
def model_fit(x):
    x.fit(X_train_c, y_train_c)
    y_pred = x.predict(X_val_c)
    model_fit.accuracy = accuracy_score(y_pred, y_val_c)
    print('Accuracy Score',accuracy_score(y_pred, y_val_c))
    print(classification_report(y_pred, y_val_c))
    #print('Confusion Matrix \n',confusion_matrix(y_pred, y_val_c))
    
    classes = names1
    
    model_cm = ConfusionMatrix(
    x, classes = classes,
    label_encoder = {1 : 'Cheap', 2 : 'Lower Range', 3 : 'Mid-Range', 4 : 'Expensive'})
    
    model_cm.fit(X_train_c, y_train_c)
    model_cm.score(X_val_c, y_val_c)
    
    model_cm.poof()    
```


```python
model_fit(KNeighborsClassifier(n_neighbors = 4))
KNN = model_fit.accuracy
```

    Accuracy Score 0.6986301369863014
                  precision    recall  f1-score   support
    
               1       0.93      0.71      0.81       112
               2       0.49      0.55      0.52        60
               3       0.62      0.66      0.64        71
               4       0.70      0.90      0.79        49
    
       micro avg       0.70      0.70      0.70       292
       macro avg       0.68      0.71      0.69       292
    weighted avg       0.73      0.70      0.70       292
    



![png](output_100_1.png)



```python
from sklearn.linear_model import LogisticRegression
model_fit(LogisticRegression())
Logistic = model_fit.accuracy
```

    Accuracy Score 0.7191780821917808
                  precision    recall  f1-score   support
    
               1       0.87      0.78      0.82        96
               2       0.51      0.59      0.54        58
               3       0.62      0.67      0.64        70
               4       0.86      0.79      0.82        68
    
       micro avg       0.72      0.72      0.72       292
       macro avg       0.71      0.71      0.71       292
    weighted avg       0.74      0.72      0.73       292
    



![png](output_101_1.png)



```python
from sklearn.naive_bayes import GaussianNB
model_fit(GaussianNB())
Gaussian = model_fit.accuracy
```

    Accuracy Score 0.4520547945205479
                  precision    recall  f1-score   support
    
               1       0.60      0.72      0.66        72
               2       0.22      0.37      0.28        41
               3       0.08      0.38      0.13        16
               4       0.94      0.36      0.52       163
    
       micro avg       0.45      0.45      0.45       292
       macro avg       0.46      0.46      0.40       292
    weighted avg       0.71      0.45      0.50       292
    



![png](output_102_1.png)



```python
from sklearn import tree
model_fit(tree.DecisionTreeClassifier())
Tree = model_fit.accuracy
```

    Accuracy Score 0.6712328767123288
                  precision    recall  f1-score   support
    
               1       0.76      0.74      0.75        88
               2       0.57      0.54      0.55        71
               3       0.57      0.65      0.61        66
               4       0.79      0.75      0.77        67
    
       micro avg       0.67      0.67      0.67       292
       macro avg       0.67      0.67      0.67       292
    weighted avg       0.68      0.67      0.67       292
    



![png](output_103_1.png)



```python
from sklearn.ensemble import RandomForestClassifier
model_fit(RandomForestClassifier(n_estimators = 100, max_depth =10, random_state = 1))
RandomForest = model_fit.accuracy
```

    Accuracy Score 0.773972602739726
                  precision    recall  f1-score   support
    
               1       0.86      0.80      0.83        93
               2       0.66      0.61      0.63        72
               3       0.74      0.79      0.76        71
               4       0.83      0.93      0.87        56
    
       micro avg       0.77      0.77      0.77       292
       macro avg       0.77      0.78      0.77       292
    weighted avg       0.77      0.77      0.77       292
    



![png](output_104_1.png)



```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {'max_depth' : [1, 5, 10],
              'learning_rate' : [0.01, 0.1], 
              'n_estimators' :[5, 10, 15]}

xgb = XGBClassifier()
xgb_cv = RandomizedSearchCV(xgb, param_grid, cv = 5)
xgb_cv.fit(X_train_c, y_train_c)
print(xgb_cv.best_params_)
print(xgb_cv.best_score_)
```

    /home/m/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)


    {'n_estimators': 10, 'max_depth': 10, 'learning_rate': 0.1}
    0.7148972602739726

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
param_grid = {'max_depth' : np.arange(20, 40),
              'learning_rate' : [0.001, 0.01, 0.1], 
              'n_estimators' : np.arange(500, 1800)}
xgb = XGBClassifier()
xgb_cv = GridSearchCV(xgb, param_grid, cv = 5)
xgb_cv.fit(X_train_c, y_train_c)
print(xgb.best_params_)
print(xgb.best_score_)

```python
model_fit(XGBClassifier(max_depth=25, learning_rate=0.1, n_estimators=1800, silent=True, 
                        objective='multi:softprop', booster='gbtree', n_jobs=2, 
                        nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                        subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, 
                        reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=1, 
                        seed=None, missing=None))
XGBClf = model_fit.accuracy
```

    Accuracy Score 0.7636986301369864
                  precision    recall  f1-score   support
    
               1       0.86      0.82      0.84        90
               2       0.63      0.58      0.60        72
               3       0.70      0.78      0.74        68
               4       0.86      0.87      0.86        62
    
       micro avg       0.76      0.76      0.76       292
       macro avg       0.76      0.76      0.76       292
    weighted avg       0.76      0.76      0.76       292
    



![png](output_107_1.png)



```python
scores_list_1 = ['KNN','Logistic','Gaussian','Tree','RandomForest','XGBClassifier']
scores_1 = [KNN, Logistic, Gaussian, Tree, RandomForest, XGBClf]
```


```python
score_df_classification = pd.DataFrame([scores_list_1, scores_1]).T
```


```python
score_df_classification.index = score_df_classification[0]
del score_df_classification[0]
score_df_classification
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KNN</th>
      <td>0.69863</td>
    </tr>
    <tr>
      <th>Logistic</th>
      <td>0.719178</td>
    </tr>
    <tr>
      <th>Gaussian</th>
      <td>0.452055</td>
    </tr>
    <tr>
      <th>Tree</th>
      <td>0.671233</td>
    </tr>
    <tr>
      <th>RandomForest</th>
      <td>0.773973</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.763699</td>
    </tr>
  </tbody>
</table>
</div>


It appears that XGBClassifier is the best choice at 76.7123%

```python
test_prediction = XGBClassifier(max_depth=25, learning_rate=0.1, n_estimators=1800, silent=True, 
                        objective='multi:softprop', booster='gbtree', n_jobs=2, 
                        nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                        subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, 
                        reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=1, 
                        seed=None, missing=None).fit(X_train_c,y_train_c).predict(test_X)
```


```python
test_prediction
```




    array([2, 2, 3, ..., 3, 1, 4])




```python
plt.subplot(1,2,1)
pd.DataFrame(test_prediction)[0].value_counts().plot.bar(figsize = (15,5))
plt.xticks([0, 1, 2, 3], names1)
plt.title('Prediction on Class Distribution on Sale Price Range')

plt.subplot(1,2,2)
train_X.SalePriceRange.value_counts().plot.bar()
plt.xticks([0, 1, 2, 3], names1)
plt.title('Class Distribution on Sale Price Range')
plt.show()
```


![png](output_114_0.png)

The test data set sale price range prediction is evenly distributed.