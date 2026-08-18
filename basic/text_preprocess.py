import pandas as pd
import string

# 简单英文停用词表：高频无意义词汇，大模型预处理要删掉
stop_words = {"i", "me", "my", "are", "is", "they", "also", "but", "for", "about", "have", "can"}

# 1.读取csv文件
df = pd.read_csv("text_data.csv")
print("====原始文本数据====")
print(df)

def clean_text(raw_text):
    # 1）全部转小写
    text = raw_text.lower()
    # 2）去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3）分词，按空格切分
    tokens = text.split()
    # 4）过滤停用词
    tokens_filtered = [word for word in tokens if word not in stop_words]
    return tokens_filtered

# 对全部文本做预处理，新增一列保存处理之后分词结果
df["processed_tokens"] = df["text"].apply(clean_text)

print("\n====预处理之后结果====")
print(df[["text", "processed_tokens"]])

# 将处理完的数据保存为新csv，模拟给大模型用的数据集
df.to_csv("processed_text_out.csv", index=False)
print("\n已输出处理后的数据集 processed_text_out.csv")