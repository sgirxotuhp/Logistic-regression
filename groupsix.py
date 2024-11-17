import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# 匯入資料
data = pd.read_csv('test.csv', delimiter=';')  # 請確保 bank.csv 檔案位於工作目錄中
print("資料集前五筆：\n", data.head())
print("\n資料集欄位資訊：\n", data.info())

# 檢查缺失值
print("\n檢查缺失值：\n", data.isnull().sum())


# Step 1:預測顧客是否會認購定期存款
# 分割資料集
data_ = pd.get_dummies(data, drop_first=True)
X = data_.drop('y_yes', axis=1)
y = data_['y_yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 訓練模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 評估模型
y_pred = model.predict(X_test)
print("分類報告：\n", classification_report(y_test, y_pred))
print("混淆矩陣：\n", confusion_matrix(y_test, y_pred))

# 顯示每個顧客的預測結果
predictions = pd.DataFrame({'實際值': y_test, '預測值': y_pred})
print("\n每個顧客的預測結果：\n", predictions)


# Step 2: 資料分析與視覺化
bins = [0, 20, 40, 60, 80, 100]
labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels)

# 繪製年齡區間的長條圖
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='age_group', hue='y')
plt.title('Age Group vs Subscription')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Subscription (y)', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

features = ['job', 'marital', 'education', 'loan']
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x=feature, hue='y')
    plt.title(f'{feature.capitalize()} vs Subscription')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Count')
    plt.legend(title='Subscription (y)', labels=['No', 'Yes'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Step 3: 預測模型
# 篩選所需的欄位
selected_features = ['age', 'balance', 'loan']
X = data[selected_features]
y = data['y'].apply(lambda x: 1 if x == 'yes' else 0)  # 將 y 轉換為 0 和 1

# 將 loan 轉換為數值型
X['loan'] = X['loan'].apply(lambda x: 1 if x == 'yes' else 0)

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立羅吉斯回歸模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測並計算準確度
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nLogistic Regression 模型的測試準確度：{accuracy:.2f}")
