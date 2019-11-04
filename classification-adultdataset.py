df = pd.read_csv('adult.data', sep=",", skipinitialspace=True, header=None)

#information from data.names 
#objective is to classify if someone makes more than or less than 50k
df.columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"
]

columns = ["age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]

salary = {'<=50K': 0, '>50K': 1}

df['salary'] = df['salary'].map(salary)
le = preprocessing.LabelEncoder()

"""
df['workclass'] = le.fit_transform(df['workclass']) 
df['education'] = le.fit_transform(df['education']) 
df['marital-status'] = le.fit_transform(df['marital-status']) 
df['occupation'] = le.fit_transform(df['occupation']) 
df['relationship'] = le.fit_transform(df['relationship']) 
df['race'] = le.fit_transform(df['race']) 
df['sex'] = le.fit_transform(df['sex']) 
df['native-country'] = le.fit_transform(df['native-country']) 
"""
for x in columns:
    df[x] = le.fit_transform(df[x])

X = df.loc[:, 'age':'native-country']
Y = df['salary']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=42)

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
#    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier()]

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name)
    print(score)

