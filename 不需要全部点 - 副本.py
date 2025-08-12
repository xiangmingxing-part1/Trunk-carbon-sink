import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# --- 1. 定义文件路径 ---
path_attributes = r"D:\北美基本属性\北美基础属性表3.csv"
path_tree_rings = r"D:\北美四张树轮标准年表\AgeDepSpline.csv"
path_npp_nc = r"D:\Model data\NPP-TRENDY\npp\year\5trend\0.5\LPJ-GUESS_S3_npp.nc"
path_map_shapefile = r"D:\map2\世界底图\World_countries.shp"
output_figure_path = r"D:\correlation_map_refined.png"

# --- 2. 加载数据 ---
print("正在加载数据...")
sites_df = pd.read_csv(path_attributes)
tree_rings_df = pd.read_csv(path_tree_rings)
npp_ds = xr.open_dataset(path_npp_nc)
world_map = gpd.read_file(path_map_shapefile)
print("数据加载完成。")

# --- 3. 数据预处理 ---
print("正在预处理数据...")
if 'year' in tree_rings_df.columns:
    tree_rings_df = tree_rings_df.set_index('year')
else:
    raise ValueError("树轮年表CSV中未找到'year'列")

start_year, end_year = 1980, 2015
tree_rings_period = tree_rings_df.loc[start_year:end_year]

if 'year' not in npp_ds.coords:
    npp_ds['year'] = npp_ds['time'].dt.year
    
npp_period = npp_ds.where((npp_ds.year >= start_year) & (npp_ds.year <= end_year), drop=True)
    
print(f"数据已筛选至 {start_year}-{end_year} 期间。")

# --- 4. 计算每个站点的相关性 ---
print("开始计算相关性...")
results = []

for index, site in sites_df.iterrows():
    site_name = site['name']
    site_lon = site['longitude']
    site_lat = site['latitude']

    if site_name not in tree_rings_period.columns:
        print(f"警告：在树轮年表中未找到站点 '{site_name}'，已跳过。")
        continue

    site_trw = tree_rings_period[site_name].dropna()

    if site_trw.empty:
        print(f"警告：站点 '{site_name}' 在指定时间段内无有效树轮数据，已跳过。")
        continue

    try:
        site_npp_ts = npp_period['npp'].sel(longitude=site_lon, latitude=site_lat, method='nearest').to_pandas().dropna()
    except Exception as e:
        print(f"错误：提取站点 '{site_name}' 的npp数据时出错：{e}")
        continue

    aligned_trw, aligned_npp = site_trw.align(site_npp_ts, join='inner')

    if len(aligned_trw) < 20:
        print(f"警告：站点 '{site_name}' 的可用重叠数据不足（少于20个），已跳过。")
        continue
    
    correlation, p_value = pearsonr(aligned_trw, aligned_npp)
    
    results.append({
        'name': site_name,
        'longitude': site_lon,
        'latitude': site_lat,
        'correlation': correlation,
        'p_value': p_value
    })

print(f"相关性计算完成，共处理了 {len(results)} 个有效站点。")

# --- 5. 创建地理数据框 (GeoDataFrame) 用于绘图 ---
if not results:
    print("没有可供绘图的数据，程序终止。")
else:
    results_df = pd.DataFrame(results)
    gdf_results = gpd.GeoDataFrame(
        results_df,
        geometry=gpd.points_from_xy(results_df.longitude, results_df.latitude),
        crs="EPSG:4326"
    )
    
    # --- 6. 绘制地图 ---
    print("正在绘制地图...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    world_map.plot(ax=ax, color='lightgray', edgecolor='white')
    
    plot = gdf_results.plot(column='correlation', ax=ax, legend=True,
                            cmap='coolwarm', vmin=-1, vmax=1,
                            legend_kwds={'label': "Correlation Coefficient (TRW vs. npp)",
                                         'orientation': "horizontal"})
    
    ax.set_xlim(-167.27, -55.00)
    ax.set_ylim(9.58, 74.99)
    
    ax.set_title(f'Correlation between Tree Rings and npp ({start_year}-{end_year})', fontsize=16)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_figure_path, dpi=300, bbox_inches='tight')
    print(f"地图已成功保存到: {output_figure_path}")
    
    plt.show()