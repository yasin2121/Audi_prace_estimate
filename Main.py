import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('audi.csv')  



# df.head(3)

df=df.drop(columns=['index','href','MileageRank','PriceRank','PPYRank','Score'])

# df.head(3)


# df.info()

# df.head(3)

df.columns=["yil","kasa","mil","motor","ps","vites","yakit","sahip","fiyat","ppy"]

# df.head(3)

df['motor']=df['motor'].str.replace("L","")

df['motor']=pd.to_numeric(df['motor'])

# df.head(3)

df=pd.get_dummies(df,columns=['kasa','vites','yakit'],drop_first=True)

# df.head(3)

y=df[['fiyat']]
x=df.drop("fiyat",axis=1)

lm=LinearRegression()
model=lm.fit(x,y)

model.predict([[2017,30000,1.6,110,1,2600,0,1]])

print(model.score(x,y))

