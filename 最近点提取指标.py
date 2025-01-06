import ee
import pandas as pd
import os

# 使用服务账号凭据进行身份验证并初始化 Earth Engine
ee.Initialize(credentials=ee.ServiceAccountCredentials(
    'your-service-account1@ee-a809277116.iam.gserviceaccount.com',
    'F:/ee-a809277116-3189e8d7d0f3.json'
))

# 加载 Excel 文件，获取点的经纬度和名称
df_sites = pd.read_excel("D:/Dosktop/site2.xlsx", header=0)

# 提取坐标和名称
coordinates = [
    {"geometry": ee.Geometry.Point([lon, lat]), "name": name} 
    for lat, lon, name in zip(df_sites['Nearest_Lat'], df_sites['Nearest_Lon'], df_sites['Site'])
]

# 定义更严格的云掩膜函数
def mask_clouds(image):
    qa = image.select('QA_PIXEL')
    cloud_mask = qa.bitwiseAnd(1 << 5).eq(0)  # 云
    cloud_confidence_mask = qa.bitwiseAnd(1 << 4).eq(0)  # 可能的云
    shadow_mask = qa.bitwiseAnd(1 << 3).eq(0)  # 云阴影
    # 合并所有掩膜条件
    combined_mask = cloud_mask.And(cloud_confidence_mask).And(shadow_mask)
    return image.updateMask(combined_mask)

# 定义计算 NDVI、NIRv 和 EVI 的函数
def add_indices(image, dataset_name):
    if dataset_name == "Landsat 8":
        nir = 'SR_B5'
        red = 'SR_B4'
        blue = 'SR_B2'
    else:  # Landsat 7 和 Landsat 5
        nir = 'SR_B4'
        red = 'SR_B3'
        blue = 'SR_B1'
    
    # 计算 NDVI
    ndvi = image.expression(
        '(NIR - RED) / (NIR + RED)', 
        {'NIR': image.select(nir), 'RED': image.select(red)}
    ).rename('NDVI')
    
    # 计算 NIRv
    nirv = image.expression(
        'NDVI * NIR',
        {'NDVI': ndvi, 'NIR': image.select(nir)}
    ).rename('NIRv')
    
    # 计算 EVI
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', 
        {'NIR': image.select(nir), 'RED': image.select(red), 'BLUE': image.select(blue)}
    ).rename('EVI')
    
    return image.addBands([ndvi, nirv, evi])

# 定义数据集和时间范围
datasets = [
    {"name": "Landsat 5", "collection": "LANDSAT/LT05/C02/T1_L2", "date_range": ['1980-06-01', '2011-11-01']},
    {"name": "Landsat 7", "collection": "LANDSAT/LE07/C02/T1_L2", "date_range": ['1980-06-01', '2017-01-01']},
    {"name": "Landsat 8", "collection": "LANDSAT/LC08/C02/T1_L2", "date_range": ['1980-06-01', '2024-12-31']}
]

# 创建保存结果的目录
os.makedirs('D:/Dosktop/最近点1.1/', exist_ok=True)

# 提取 NDVI、NIRv、EVI 数据并保存到文件
for i, point_data in enumerate(coordinates):
    point = point_data['geometry']
    name = point_data['name']
    print(f"Processing point {i+1}/{len(coordinates)}: {name}")

    point_indices_data = []
    for dataset in datasets:
        dataset_name = dataset["name"]
        collection = ee.ImageCollection(dataset["collection"]) \
            .filterBounds(point).filterDate(*dataset["date_range"]) \
            .map(mask_clouds).map(lambda img: add_indices(img, dataset_name))

        # 提取 NDVI、NIRv 和 EVI 值
        indices_features = collection.map(
            lambda img: ee.Feature(
                point, {
                    'Date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss'),
                    'Dataset': dataset_name,
                    'NDVI': img.reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=point, scale=30
                    ).get('NDVI'),
                    'NIRv': img.reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=point, scale=30
                    ).get('NIRv'),
                    'EVI': img.reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=point, scale=30
                    ).get('EVI')
                }
            )
        ).getInfo()

        # 收集结果
        point_indices_data.extend([
            {"Name": name, "Date": f['properties']['Date'], "Dataset": f['properties']['Dataset'], 
             "NDVI": f['properties']['NDVI'], "NIRv": f['properties']['NIRv'], "EVI": f['properties']['EVI']}
            for f in indices_features['features']
        ])

    # 调整变量顺序并保存数据到 Excel
    pd.DataFrame(
        point_indices_data, columns=["Name", "Date", "Dataset", "NDVI", "NIRv", "EVI"]
    ).to_excel(f'D:/Dosktop/最近点1.1/point_{i+1}_indices.xlsx', index=False)
    print(f"Point {i+1} ({name}) data saved to D:/Dosktop/最近点1.1/point_{i+1}_indices.xlsx")

print("所有点的 NDVI、NIRv 和 EVI 数据已保存到 D:/Dosktop/最近点1.1/")
