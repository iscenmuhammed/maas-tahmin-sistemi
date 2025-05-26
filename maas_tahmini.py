import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Veri setini oku
df = pd.read_csv("ornek_veri.csv")

# Eksik veriyi doldur (ortalama ile)
df['maas'] = df['maas'].fillna(df['maas'].mean())

# Cinsiyeti sayıya çevir (LabelEncoder)
le = LabelEncoder()
df['cinsiyet_kod'] = le.fit_transform(df['cinsiyet'])

# Giriş (X) ve çıkış (y) verileri
X = df[['yas', 'cinsiyet_kod']]
y = df['maas']

# Modeli kur ve eğit
model = LinearRegression()
model.fit(X, y)

# Örnek tahmin: 40 yaşında kadın
tahmin = model.predict([[40, 1]])
print(f"40 yaşında bir kadın için tahmini maaş: {tahmin[0]:.2f} TL")
