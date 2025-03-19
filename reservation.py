
import pandas as pd 

df_booking = pd.read_excel(r'C:\Users\Admin\Documents\Python\DB_KHACH-SAN.xlsx',sheet_name='bookings')
df_payment = pd.read_excel(r'C:\Users\Admin\Documents\Python\DB_KHACH-SAN.xlsx',sheet_name='payments_senior')
df_service_usage = pd.read_excel(r'C:\Users\Admin\Documents\Python\DB_KHACH-SAN.xlsx',sheet_name='service_usage_senior')
df_service = pd.read_excel(r'C:\Users\Admin\Documents\Python\DB_KHACH-SAN.xlsx',sheet_name='services_senior')
df_customer = pd.read_excel(r'C:\Users\Admin\Documents\Python\DB_KHACH-SAN.xlsx',sheet_name='customers_senior')
df_room = pd.read_excel(r'C:\Users\Admin\Documents\Python\DB_KHACH-SAN.xlsx',sheet_name='rooms')
from pandasql import sqldf

df_merge1 = pd.merge(left=df_booking, right=df_customer, how='inner',on='customer_id')
from pandasql import sqldf

query_1 = """
    SELECT COUNT(booking_id) as Reserve_Num, customer_id, full_name 
    FROM df_merge1 
    WHERE status NOT IN('Cancelled','Pending') 
    GROUP BY customer_id, full_name 
    ORDER BY Reserve_Num DESC  
    LIMIT 1
"""
result_1 = sqldf(query_1, globals()) 
print(result_1)
# Có thể thấy khách hàng số 130 tên Customer 130 là khách hàng Vip khi đặt 10 phòng

#  dự đoán nhóm khách hàng có nguy cơ rời bỏ coi pending là confirm luôn 
query_reserve = """
    SELECT COALESCE(COUNT(b.booking_id), 0) AS Reserve_Num, c.customer_id, c.full_name, COALESCE(SUM(su.total_price),0) as total_service_price
    FROM df_customer c  
    LEFT JOIN df_booking b  
    ON c.customer_id = b.customer_id 
    LEFT JOIN df_service_usage su
    ON su.booking_id = b.booking_id
    AND b.status != 'Cancelled' 
    GROUP BY c.customer_id, c.full_name  
    ORDER BY Reserve_Num DESC  
"""
result_reserve = sqldf(query_reserve, globals())
df_reserve = pd.DataFrame(result_reserve)
print(df_reserve)

query_cancel = """
    SELECT COALESCE(COUNT(b.booking_id), 0) AS Cancel_Num, c.customer_id, c.full_name 
    FROM df_customer c  
    LEFT JOIN df_booking b  
    ON c.customer_id = b.customer_id 
    AND b.status NOT IN ('Pending', 'Confirmed')  
    GROUP BY c.customer_id, c.full_name  
    ORDER BY Cancel_Num DESC   
"""
result_cancel = sqldf(query_cancel,globals())
df_cancel = pd.DataFrame(result_cancel)
print(df_cancel)

df_cluster = pd.merge(df_reserve, df_cancel, on='customer_id', how='inner')
print(df_cluster)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X = df_cluster[['Reserve_Num','Cancel_Num']]
interia = []
k_value = range(1,20)
for k in k_value:
    k_model = KMeans(n_clusters=k, init='k-means++',random_state=42, n_init=100)
    k_model.fit(X)
    interia.append(k_model.inertia_)

plt.figure(figsize=(10,8))
plt.plot(k_value,interia, marker = 'o',color = 'blue')
plt.grid(True)
plt.show()

# Chia làm 2 nhóm

k_model =KMeans(n_clusters=3,init='k-means++',n_init=100,random_state=42)

df_cluster['Thuộc Cụm'] =k_model.fit_predict(X)

import seaborn as sns
sns.scatterplot(data=df_cluster,x='Reserve_Num',y='Cancel_Num',hue='Thuộc Cụm',palette='tab10')
plt.scatter(k_model.cluster_centers_[:,0],k_model.cluster_centers_[:,1],c='red',marker='x',s=200)
plt.legend()
plt.title('Biểu đồ phân cụm')
plt.xlabel('Số lượng đặt theo nhóm')
plt.ylabel('Số lượng hủy theo nhóm')
plt.grid(True)
plt.show()

grouped = df_cluster.groupby('Thuộc Cụm').agg(soluong=('customer_id', 'count'),Avg_reserve=('Reserve_Num', 'mean'), Avg_cancel=('Cancel_Num', 'mean'), Avg_usage = ('total_price','mean'))
print(grouped.sort_values('Avg_cancel',ascending=False))
# Nhóm 1 là những người có số lần hủy phòng nhiều nhất (trung bình gần 5 lần)
# tuy nhiên nhóm này lại là nhóm sử dụng nhiều dịch vụ nhất
# Nên bằng mọi giá phải giữ lại nhóm khách hàng này

# Nhóm 0 là nhóm có tỷ lệ hủy phòng nhiều nhất (30%) (Avg_Cancel/Avg_reserve)
df_cluster.columns
from sklearn.metrics import silhouette_score
X = df_cluster.select_dtypes(include=['number']).drop(columns=['Thuộc Cụm','full_name_x','full_name_y', 'customer_id'], errors='ignore')

labels = df_cluster['Thuộc Cụm'].astype(int)

print(f"Shape của X: {X.shape}, Số Cluster: {labels.nunique()}")

silhouette = silhouette_score(X, labels)
print(f"Chỉ số Silhouette: {silhouette:.4f}")
# Chỉ số silhouette là 0.1843 tạm chấp nhận đuọc
# 1 là mất 0 là còn
df_cluster.drop(columns='full_name_x', inplace=True)
df_cluster['Churn'] = df_cluster.apply(lambda x: 1 if x['Cancel_Num'] > 0.3 * x['Reserve_Num'] else 0, axis=1)
df_cluster.drop(columns=['Reserve_Num','Cancel_Num'])

# Gộp bảng rooms với bảng booking để xem khách hàng ở những hạng phòng nào
df_room['room_type'].value_counts()
df_room['room_type']=df_room['room_type'].map({
    'Standard' : 1,
    'Deluxe' : 2,
    'Suite' :3,
    'Executive' :4,
    'Presidential' :5
}).astype('int')

query_room = """
    SELECT COALESCE(SUM(r.room_type), 0) AS Total_room_value, b.customer_id, SUM(price_per_night) as total_room_value_per_night
    FROM df_room r 
    RIGHT JOIN df_booking b  
    ON r.room_id = b.room_id
    AND b.status != 'Cancelled' 
    GROUP BY b.customer_id 
    ORDER BY Total_room_value DESC  
"""
result_room = sqldf(query_room,globals())
df_room = pd.DataFrame(result_room)
print(result_room)

# gộp df_cluster với df_room

df_cluster.drop(columns=['Reserve_Num','Cancel_Num','full_name_y'], inplace=True)

df_classify = pd.merge(df_cluster,df_room,on='customer_id',how='inner')
print(df_classify)

from sklearn.model_selection import KFold, cross_validate

X1 = df_classify.drop(columns=['Churn', 'customer_id'])
Y1 = df_classify['Churn']

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(model, X1, Y1, cv=k_fold, scoring='accuracy')
print(cv_results['test_score'].mean())
# Chỉ số accuracy sấp xỉ 0.71 -> mô hình tốt
model.fit(X1, Y1)

feature_importance = pd.DataFrame({
    'Feature': X1.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)
# Tổng giá trị chi tiêu cho dịch vụ phòng có ảnh hưởng lớn đến sự rời đi của khách hàng
# Hai giá trị còn lại ảnh hưởng ít hơn nhưng chênh lệch ko nhiều


