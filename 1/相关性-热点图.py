import os
import glob
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ---------------- 用户可修改区 ----------------
PATH_ATTRIBUTES = r"D:\北美基本属性\北美基础属性表3.csv"
PATH_TREE_RINGS = r"D:\北美四张树轮标准年表\AgeDepSpline.csv"
DIR_NC = r"D:\Model data\NPP-TRENDY\npp\year\5trend\0.5"
OUTPUT_DIR = r"D:\1"

START_YEAR, END_YEAR = 1982, 2001
MIN_OVERLAP = 20
TARGET_VAR = "npp"
# --------------------------------------------


def detect_lon_lat_dims(ds):
    """自动探测经纬度维度名"""
    lon_candidates = ["lon", "longitude", "Lon", "Longitude"]
    lat_candidates = ["lat", "latitude", "Lat", "Latitude"]
    lon_name = next((n for n in lon_candidates if n in ds.dims or n in ds.coords), None)
    lat_name = next((n for n in lat_candidates if n in ds.dims or n in ds.coords), None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"未识别经纬度，dims: {list(ds.dims)}, coords: {list(ds.coords)}")
    return lon_name, lat_name


def ensure_year_coord(ds):
    """确保存在 year 坐标"""
    if "year" not in ds.coords:
        if "time" not in ds.coords and "time" not in ds.dims:
            raise ValueError("无 year 或 time 坐标")
        ds["year"] = ds["time"].dt.year
    return ds


def pick_target_var(ds, preferred="npp"):
    """选择要分析的变量"""
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 0:
        raise ValueError("无数据变量")
    return list(ds.data_vars.keys())[0]


def compute_avg_corr_for_nc(nc_path, sites_df, tree_rings_period):
    try:
        ds = xr.open_dataset(nc_path)
        ds = ensure_year_coord(ds)
        ds_period = ds.where((ds.year >= START_YEAR) & (ds.year <= END_YEAR), drop=True)
        lon_name, lat_name = detect_lon_lat_dims(ds_period)
        var_name = pick_target_var(ds_period, preferred=TARGET_VAR)
    except Exception as e:
        print(f"{os.path.basename(nc_path)} 打开失败: {e}")
        return None

    results = []
    for _, site in sites_df.iterrows():
        site_name = site["name"]
        if site_name not in tree_rings_period.columns:
            continue
        site_trw = tree_rings_period[site_name].dropna()
        if site_trw.empty:
            continue
        try:
            site_var_ts = (
                ds_period[var_name]
                .sel({lon_name: site["longitude"], lat_name: site["latitude"]}, method="nearest")
                .to_pandas()
                .dropna()
            )
        except:
            continue

        aligned_trw, aligned_var = site_trw.align(site_var_ts, join="inner")
        if len(aligned_trw) < MIN_OVERLAP:
            continue

        corr, _ = pearsonr(aligned_trw, aligned_var)
        results.append(corr)

    if not results:
        return None
    return sum(results) / len(results)  # 平均相关系数


def main():
    # 读取数据
    sites_df = pd.read_csv(PATH_ATTRIBUTES)
    tree_rings_df = pd.read_csv(PATH_TREE_RINGS).set_index("year")
    tree_rings_period = tree_rings_df.loc[START_YEAR:END_YEAR]

    # 遍历 NC 文件
    avg_corr_list = []
    nc_files = sorted(glob.glob(os.path.join(DIR_NC, "*.nc")))
    for nc in nc_files:
        avg_corr = compute_avg_corr_for_nc(nc, sites_df, tree_rings_period)
        if avg_corr is not None:
            avg_corr_list.append({"file": os.path.basename(nc), "avg_corr": avg_corr})

    if not avg_corr_list:
        print("没有计算到任何平均相关值")
        return

    df = pd.DataFrame(avg_corr_list)

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(df["file"], df["avg_corr"], color="skyblue")
    plt.xticks(rotation=90)
    plt.ylabel("Average Correlation")
    plt.title(f"Average Correlation ({START_YEAR}-{END_YEAR})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "average_correlation_bar.png"), dpi=300)
    plt.close()

    print("已保存平均相关系数柱状图到：", os.path.join(OUTPUT_DIR, "average_correlation_bar.png"))


if __name__ == "__main__":
    main()
