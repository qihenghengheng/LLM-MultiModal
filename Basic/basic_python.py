# 1. 循环打印数字
print("====循环打印数字====")
for i in range(1, 11):
    print(i)

# 2. 字典存储文本数据
print("\n====字典存储文本数据====")
student_info = {
    "name": "张三",
    "major": "软件工程",
    "hobby": "编程",
    "grade": "大三"
}
print(student_info)
print("姓名：", student_info["name"])

# 3. 简单求和函数
print("\n====简单求和函数====")
def sum_func(a, b):
    return a + b

result = sum_func(12, 28)
print(f"12 + 28 = {result}")
