import pandas as pd
data = pd.read_csv(r"C:\Users\ajay_\PycharmProjects\Microsoft\car_sales.csv")
#print(data)
modified_data= data.drop(['Model', 'Latest_Launch'], axis = 1)
#print(modified_data)
new_data = modified_data[modified_data['Price_in_thousands'].notna()]
new_data['__year_resale_value'].fillna(value = new_data['__year_resale_value'].median(), inplace = True)
new_data['Fuel_efficiency'].fillna(value = new_data['Fuel_efficiency'].median(), inplace = True)
#new_data.info()
new_data['Curb_weight'].fillna(value = new_data['Curb_weight'].mean(), inplace = True)
### Drop the columns - _year_resale_value, Engine_size, Wheelbase, Length, Fuel_capacity, Power_perf_factor
final_data = new_data.drop(['__year_resale_value', 'Engine_size', 'Wheelbase', 'Length', 'Fuel_capacity', 'Power_perf_factor'], axis = 1)
#Encoding the variable - Manufacturer such that if the average sales of a manufacturer are greater than 75, they belong
#to class 2, else class 1
manufacturer_count = dict()
for each_manufacturer in list(data['Manufacturer']):
    if each_manufacturer not in manufacturer_count:
        manufacturer_count[each_manufacturer] = 1
    else:
        manufacturer_count[each_manufacturer] += 1
sorted_manufacturers = dict(sorted(manufacturer_count.items(), key=lambda item: item[1], reverse=True))
manufacturers = []
for each_manufacturer in final_data['Manufacturer']:
    if sorted_manufacturers[each_manufacturer] > 75:
        manufacturers.append(2)
    else:
        manufacturers.append(1)
final_data['Manufacturer'] = manufacturers
#all manufacuters belong to class:1 hence we drop column:manufacturer
final_data = final_data.drop(['Manufacturer'], axis = 1)
print(final_data.columns)
final_data.to_csv(r'C:\Users\ajay_\PycharmProjects\Microsoft\pre_processed_file.csv',index=False)

