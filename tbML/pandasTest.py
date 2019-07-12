import pandas

food_info = pandas.read_csv("food_info.csv")
# print(type(food_info))
# print(food_info.dtypes)
# print(help(pandas.read_csv))
print(food_info.head())
print(food_info.tail())

