import os, glob
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np

PATH_ATTRIBUTES = r"D:\北美基本属性\北美基础属性表3.csv"
PATH_TREE_RINGS = r"D:\北美四张树轮标准年表\AgeDepSpline.csv"
PATH_MAP_SHP    = r"D:\map2\世界底图\World_countries.shp"
DIR_NC = r"D:\Model data\NPP-TRENDY\npp\year\5trend\0.5"
OUTPUT_DIR = r"D:\1\斯皮尔曼"
START_YEAR, END_YEAR = 1982, 2001
MIN_OVERLAP = 20
TARGET_VAR = "npp"

def detect_lon_lat_dims(ds):
    lon_candidates = ["lon","longitude","Lon","Longitude"]
    lat_candidates = ["lat","latitude","Lat","Latitude"]
    lon_name = next((n for n in lon_candidates if n in ds.dims or n in ds.coords), None)
    lat_name = next((n for n in lat_candidates if n in ds.dims or n in ds.coords), None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"未识别经纬度名。dims:{list(ds.dims)} coords:{list(ds.coords)}")
    return lon_name, lat_name

def ensure_year_coord(ds):
    if "year" not in ds.coords:
        if "time" not in ds.coords and "time" not in ds.dims:
            raise ValueError("无 year 与 time 坐标。")
        ds["year"] = ds["time"].dt.year
    return ds

def pick_target_var(ds, preferred="npp"):
    if preferred in ds.data_vars: return preferred
    if len(ds.data_vars)==0: raise ValueError("无数据变量。")
    return list(ds.data_vars.keys())[0]

def compute_and_plot_for_nc(nc_path, sites_df, tree_rings_period, world_map):
    print(f"\n>>> 处理文件: {nc_path}")
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        print(f"打开 NC 出错：{e}"); return
    try:
        ds = ensure_year_coord(ds)
        ds_period = ds.where((ds.year>=START_YEAR)&(ds.year<=END_YEAR), drop=True)
    except Exception as e:
        print(f"年份处理出错：{e}"); return
    try:
        lon_name, lat_name = detect_lon_lat_dims(ds_period)
    except Exception as e:
        print(f"经纬度识别失败：{e}"); return
    try:
        var_name = pick_target_var(ds_period, preferred=TARGET_VAR)
    except Exception as e:
        print(f"变量选择失败：{e}"); return
    print(f"使用变量:{var_name}; 经度:{lon_name}; 纬度:{lat_name}")

    results = []
    total_sites = len(sites_df)
    for _, site in sites_df.iterrows():
        site_name = site["name"]; site_lon = site["longitude"]; site_lat = site["latitude"]
        if site_name not in tree_rings_period.columns: continue
        site_trw = tree_rings_period[site_name].dropna()
        if site_trw.empty: continue
        try:
            site_var_ts = ds_period[var_name].sel({lon_name:site_lon, lat_name:site_lat}, method="nearest").to_pandas().dropna()
        except Exception as e:
            print(f"提取站点 {site_name} 出错：{e}"); continue
        aligned_trw, aligned_var = site_trw.align(site_var_ts, join="inner")
        if len(aligned_trw) < MIN_OVERLAP: continue
        corr, pval = spearmanr(aligned_trw, aligned_var)
        results.append({"name":site_name,"longitude":site_lon,"latitude":site_lat,"correlation":corr,"p_value":pval})

    print(f"有效站点数：{len(results)} / {total_sites}")
    if not results:
        print("无可绘制数据，跳过。"); return

    results_df = pd.DataFrame(results)
    gdf_results = gpd.GeoDataFrame(results_df, geometry=gpd.points_from_xy(results_df.longitude, results_df.latitude), crs="EPSG:4326")
    avg_corr = results_df["correlation"].mean()

    base = os.path.splitext(os.path.basename(nc_path))[0]
    out_name = f"correlation_map_{base}_{START_YEAR}-{END_YEAR}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    print("保存：", out_path)

    fig, ax = plt.subplots(1,1,figsize=(12,12))
    world_map.plot(ax=ax, color="lightgray", edgecolor="white")
    gdf_results.plot(column="correlation", ax=ax, legend=True, cmap="coolwarm", vmin=-1, vmax=1,
                     legend_kwds={"label": f"Spearman ρ (TRW vs. {var_name})", "orientation":"horizontal"})
    ax.set_xlim(-167.27, -55.00); ax.set_ylim(9.58, 74.99)
    ax.set_title(f"Spearman ρ: Tree Rings vs {var_name} ({START_YEAR}-{END_YEAR})\nAverage ρ: {avg_corr:.2f}", fontsize=16)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)

def main():
    print("加载基础数据...")
    sites_df = pd.read_csv(PATH_ATTRIBUTES)
    tree_rings_df = pd.read_csv(PATH_TREE_RINGS)
    world_map = gpd.read_file(PATH_MAP_SHP)
    if "year" not in tree_rings_df.columns: raise ValueError("树轮年表无 year 列。")
    tree_rings_period = tree_rings_df.set_index("year").loc[START_YEAR:END_YEAR]
    nc_files = sorted(glob.glob(os.path.join(DIR_NC, "*.nc")))
    if not nc_files:
        print("未找到 .nc 文件：", DIR_NC); return
    print(f"发现 {len(nc_files)} 个 NC，开始处理...")
    for nc in nc_files:
        compute_and_plot_for_nc(nc, sites_df, tree_rings_period, world_map)
    print("\n完成。")

if __name__ == "__main__":
    main()
