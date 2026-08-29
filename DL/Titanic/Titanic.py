import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# -------------------- 1. 数据加载 --------------------
train_df = pd.read_csv(r"E:\VS_Code_Program\LLM-MultiModal\DL\Titanic\train.csv")
test_df = pd.read_csv(r"E:\VS_Code_Program\LLM-MultiModal\DL\Titanic\test.csv")
gender_submission = pd.read_csv(r"E:\VS_Code_Program\LLM-MultiModal\DL\Titanic\gender_submission.csv")

# -------------------- 2. 数据预处理 --------------------
def preprocess(df, is_train=True):
    """
    对 Titanic 数据集进行特征工程和清洗。
    返回特征矩阵 X 和目标 y（如果是训练集）。
    """
    df_copy = df.copy()
    
    # 提取称呼（Title）以丰富特征（可选）
    df_copy['Title'] = df_copy['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Col': 7,
                 'Major': 8, 'Mlle': 9, 'Countess': 10, 'Ms': 11, 'Lady': 12, 'Jonkheer': 13,
                 'Don': 14, 'Dona': 15, 'Mme': 16, 'Capt': 17, 'Sir': 18}
    df_copy['Title'] = df_copy['Title'].map(title_map).fillna(0).astype(int)
    
    # 年龄缺失填充中位数
    df_copy['Age'].fillna(df_copy['Age'].median(), inplace=True)
    
    # 船舱缺失填充 'U'（未知）
    df_copy['Cabin'] = df_copy['Cabin'].fillna('U').apply(lambda x: x[0])
    cabin_map = {'U': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
    df_copy['Cabin'] = df_copy['Cabin'].map(cabin_map).fillna(0).astype(int)
    
    # 登船港口填充众数（S）
    df_copy['Embarked'].fillna('S', inplace=True)
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    df_copy['Embarked'] = df_copy['Embarked'].map(embarked_map).astype(int)
    
    # 票价缺失填充0（test中有缺失）
    df_copy['Fare'].fillna(df_copy['Fare'].median(), inplace=True)
    
    # 性别编码：male=0, female=1
    df_copy['Sex'] = df_copy['Sex'].map({'male': 0, 'female': 1}).astype(int)
    
    # 选择特征
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin', 'Title']
    X = df_copy[features].values.astype(np.float32)
    
    if is_train:
        y = df_copy['Survived'].values.astype(np.int64)
        return X, y
    else:
        return X

# 处理训练集
X_train_raw, y_train_raw = preprocess(train_df, is_train=True)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train_raw, y_train_raw, 
                                                  test_size=0.2, random_state=42)

# 标准化特征（基于训练集统计量）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 测试集预处理
X_test = preprocess(test_df, is_train=False)
X_test = scaler.transform(X_test)

# -------------------- 3. 自定义 Dataset 和 DataLoader --------------------
class TitanicDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

batch_size = 64
train_dataset = TitanicDataset(X_train, y_train)
val_dataset = TitanicDataset(X_val, y_val)
test_dataset = TitanicDataset(X_test)  # 无标签

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------- 4. 定义神经网络模型 --------------------
class TitanicNet(nn.Module):
    def __init__(self, input_dim):
        super(TitanicNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

input_dim = X_train.shape[1]
model = TitanicNet(input_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------- 5. 训练循环 --------------------
epochs = 100
best_val_acc = 0.0

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)
    
    train_acc = correct_train / total_train
    avg_train_loss = train_loss / total_train
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == y_batch).sum().item()
            total_val += y_batch.size(0)
    
    val_acc = correct_val / total_val
    avg_val_loss = val_loss / total_val
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}')

print(f'Best Validation Accuracy: {best_val_acc:.4f}')

# -------------------- 6. 在测试集上预测 --------------------
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_preds = []
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())

# 生成提交文件
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")

# 如果提供了 gender_submission.csv，可以计算测试集准确率（仅用于验证）
try:
    true_labels = gender_submission['Survived'].values
    test_acc = accuracy_score(true_labels, test_preds)
    print(f'Test Accuracy (against gender_submission): {test_acc:.4f}')
except:
    print("gender_submission.csv not used for evaluation.")