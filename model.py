import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

def train_model():
    df = pd.read_csv('consumer_electronics_sales_data.csv')

    # Drop kolom yang tidak diperlukan
    df = df.drop(['ProductID'], axis=1)

    # Encoding kolom kategorikal
    le = LabelEncoder()
    df['ProductCategory'] = le.fit_transform(df['ProductCategory'])
    df['ProductBrand'] = le.fit_transform(df['ProductBrand'])

    # Menentukan fitur dan target
    X = df.loc[:, 'ProductCategory':'CustomerSatisfaction']
    y = df['PurchaseIntent']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Normalisasi data
    scaler = MinMaxScaler(feature_range=(0, 10))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Inisialisasi dan latih model KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    return knn, scaler

model, scaler = train_model()

def predict(data):
    data = scaler.transform(data)
    predictions = model.predict(data)
    return predictions
