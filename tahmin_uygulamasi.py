import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Veri setini oku
df = pd.read_csv("ornek_veri.csv")
df['maas'] = df['maas'].fillna(df['maas'].mean())

# Cinsiyeti sayıya çevir
le = LabelEncoder()
df['cinsiyet_kod'] = le.fit_transform(df['cinsiyet'])  # Erkek: 0, Kadın: 1

# Modeli eğit
X = df[['yas', 'cinsiyet_kod']]
y = df['maas']
model = LinearRegression()
model.fit(X, y)

# Kullanıcıdan giriş al
print("Maaş Tahmin Sistemi\n-------------------")
yas = int(input("Yaşınızı giriniz: "))
cinsiyet = input("Cinsiyetinizi giriniz (Erkek/Kadın): ")

# Tahmin yap
kod = 0 if cinsiyet.lower() == 'erkek' else 1
tahmin = model.predict([[yas, kod]])[0]
print(f"\nTahmini maaş: {tahmin:.2f} TL")

input("\nKapatmak için ENTER'a bas...")
