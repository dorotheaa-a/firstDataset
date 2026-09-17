# firstDataset

# Set the path to the file you'd like to load
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


def train_test_split(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)  # shuffle row indices
    n_test = int(n * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
