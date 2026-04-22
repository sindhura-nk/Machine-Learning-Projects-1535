from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

## Code for data cleaning and preprocessing
## this code deals with categorical and continuous data
def cleaning_preprocessing_cat_con(data):
    # first create cat and con list
    cat = list(data.select_dtypes(include='str').columns)
    con = list(data.select_dtypes(include='number').columns)

    # create categorical pipeline
    cat_pipe = make_pipeline(
        SimpleImputer(strategy='most_frequent'), # data cleaning - replaced with mode
        OneHotEncoder(handle_unknown='ignore',sparse_output=False) # encoding of categorical data
    )

    num_pipe = make_pipeline(
        SimpleImputer(strategy='median'), # data cleaning- replacing with median
        StandardScaler() # numeric data scaling
    )

    pre = ColumnTransformer([
        ("cat",cat_pipe,cat),
        ("con",num_pipe,con)
    ]).set_output(transform='pandas')

    return pre # returning the preprocessor
