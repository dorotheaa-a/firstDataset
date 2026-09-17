# firstDataset

#Set the path to the file you'd like to load
file_path = "/kaggle/input/datasets/abigailwiryokasa/testing/Drug.csv"

#df accesses the actual csv file
df = pd.read_csv(file_path)

#the function .head() prints the first 5 records
print("First 5 records:", df.head())
#the function .info() provides a summary of the dataframe which includes number of rows, column names and its data types, and how much memory the data frame consuming
print(df.info())

#this drops the EaseOfUse column and converts it into a NumPy array
X = df.drop(columns=['EaseOfUse']).to_numpy()
y = df['EaseOfUse'].to_numpy()


#for pandas dataframe, the iloc property gets or sets the values of specified indexes
X = df.iloc[:, :-1] # every column except the last
y = df.iloc[:, -1] # only the last column

#split the training data into 2 sets, one for training and the other testing
def train_test_split(X, y, test_ratio=0.2, seed=42):
    #this creates a random number generator with a fixed seed to make sure that the same random split can be reproduced
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)  # shuffle row indices
    #calculates how many samples to assign to test set, so if test_ratio=0.2 means only 20% of the data is used for testing
    n_test = int(n * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    #select rows from X and Y using the generated indices
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    #return the separated training and testing data
    return X_train, X_test, y_train, y_test

#applies the train-test split function, one for training and the other for testing
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

#this standardizes the numerical values 
#standardized value = (value - mean) / standard deviation
#mean and std are calculated from the training set
def standardize(X_train, X_test):
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    mu = X_train[numeric_cols].mean()
    sigma = X_train[numeric_cols].std()
    X_train_scaled[numeric_cols] = (
        X_train[numeric_cols] - mu
    ) / sigma
    X_test_scaled[numeric_cols] = (
        X_test[numeric_cols] - mu
    ) / sigma
    return X_train_scaled, X_test_scaled

#apply standardization to training and testing sets
X_train_s, X_test_s = standardize(X_train, X_test)

#display standardized training set
print(X_train_s)
print("------------------------------------------------------")
#displays standardized testing set
print(X_test_s)
print(X_train.shape)

#select the numerical columns for validation
numeric_cols = X_train_s.select_dtypes(include=["float64", "int64"]).columns
#check the mean and standard deviation
print(X_train_s[numeric_cols].mean())
print(X_train_s[numeric_cols].std())

#check the size of each split
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
