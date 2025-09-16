import os
import glob
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# ---------------- 用户可修改区 ----------------
# 基础数据路径
PATH_ATTRIBUTES = r"D:\北美基本属性\北美基础属性表3.csv"
PATH_TREE_RINGS = r"D:\北美四张树轮标准年表\AgeDepSpline.csv"
PATH_MAP_SHP    = r"D:\map2\世界底图\World_countries.shp"

# NC 文件所在目录（将批量遍历此目录下的所有 .nc）
DIR_NC = r"D:\Model data\NPP-TRENDY\npp\year\5trend\0.5"

# 输出目录（建议保持为 D:\ 或某个专门文件夹）
OUTPUT_DIR = r"D:\1"

# 相关性计算的年份窗口（沿用你近期示例 1981–2000）
START_YEAR, END_YEAR = 1982, 2001

# 至少多少个重叠年份才计算相关（与你脚本一致为 20）
MIN_OVERLAP = 20

# 目标变量名称（若 NC 中不是 'npp'，会自动尝试寻找第一个数据变量）
TARGET_VAR = "npp"
# --------------------------------------------


def detect_lon_lat_dims(ds):
    """自动探测经纬度维度名，兼容 ('lon','lat') / ('longitude','latitude') / 大小写变体"""
    lon_candidates = ["lon", "longitude", "Lon", "Longitude"]
    lat_candidates = ["lat", "latitude", "Lat", "Latitude"]
    lon_name = next((n for n in lon_candidates if n in ds.dims or n in ds.coords), None)
    lat_name = next((n for n in lat_candidates if n in ds.dims or n in ds.coords), None)
    if lon_name is None or lat_name is None:
        raise ValueError("未能在 NC 中识别经纬度维度/坐标名，请检查文件。"
                         f" 现有 dims: {list(ds.dims)}; coords: {list(ds.coords)}")
    return lon_name, lat_name


def ensure_year_coord(ds):
    """确保存在 year 坐标；若无，则由 time->year 生成（与你的做法一致）"""
    if "year" not in ds.coords:
        if "time" not in ds.coords and "time" not in ds.dims:
            raise ValueError("数据集中既无 year 也无 time 坐标，无法构造年份。")
        ds["year"] = ds["time"].dt.year
    return ds


def pick_target_var(ds, preferred="npp"):
    """选择要分析的数据变量：优先选择 'npp'，否则选第一个 DataArray"""
    if preferred in ds.data_vars:
        return preferred
    # 选第一个 data_var
    if len(ds.data_vars) == 0:
        raise ValueError("NC 文件中没有可用的数据变量。")
    return list(ds.data_vars.keys())[0]


def compute_and_plot_for_nc(nc_path, sites_df, tree_rings_period, world_map):
    print(f"\n>>> 处理文件: {nc_path}")
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        print(f"打开 NC 出错：{e}")
        return

    # 确保 year 坐标，并按年份窗口筛选（与你代码一致的 where 方式）
    try:
        ds = ensure_year_coord(ds)
        ds_period = ds.where((ds.year >= START_YEAR) & (ds.year <= END_YEAR), drop=True)
    except Exception as e:
        print(f"年份坐标处理出错：{e}")
        return

    # 识别经纬度维度名
    try:
        lon_name, lat_name = detect_lon_lat_dims(ds_period)
    except Exception as e:
        print(f"经纬度识别失败：{e}")
        return

    # 选择变量
    try:
        var_name = pick_target_var(ds_period, preferred=TARGET_VAR)
    except Exception as e:
        print(f"变量选择失败：{e}")
        return

    print(f"使用变量: {var_name}; 经度维: {lon_name}; 纬度维: {lat_name}")

    # 逐站点计算 Pearson 相关
    results = []
    total_sites = len(sites_df)
    for idx, site in sites_df.iterrows():
        site_name = site["name"]
        site_lon  = site["longitude"]
        site_lat  = site["latitude"]

        if site_name not in tree_rings_period.columns:
            # 站点不在年表中
            continue

        site_trw = tree_rings_period[site_name].dropna()
        if site_trw.empty:
            # 指定期间无有效树轮数据
            continue

        try:
            # 最近邻抽取该变量的时间序列（与你脚本相同方法：nearest）【参考：单文件实现】 
            # 注：此处不使用 .sel(year=slice(...))，因为前面已经 where 了年份
            site_var_ts = (
                ds_period[var_name]
                .sel({lon_name: site_lon, lat_name: site_lat}, method="nearest")
                .to_pandas()
                .dropna()
            )
        except Exception as e:
            print(f"提取站点 {site_name} 的 {var_name} 时出错：{e}")
            continue

        # 与树轮按年份 inner 对齐（方法与单文件版一致）【参考：单文件实现】 
        aligned_trw, aligned_var = site_trw.align(site_var_ts, join="inner")

        if len(aligned_trw) < MIN_OVERLAP:
            continue

        corr, pval = pearsonr(aligned_trw, aligned_var)
        results.append({
            "name": site_name,
            "longitude": site_lon,
            "latitude": site_lat,
            "correlation": corr,
            "p_value": pval
        })

    print(f"有效站点数：{len(results)} / {total_sites}")

    if not results:
        print("没有可绘制的数据，本文件跳过。")
        return

    # 生成 GeoDataFrame 并绘图
    results_df = pd.DataFrame(results)
    gdf_results = gpd.GeoDataFrame(
        results_df,
        geometry=gpd.points_from_xy(results_df.longitude, results_df.latitude),
        crs="EPSG:4326"
    )

    # 平均相关（标题显示）
    avg_corr = results_df["correlation"].mean()

    # 组织输出文件名：模型名_起止年.png
    base = os.path.splitext(os.path.basename(nc_path))[0]
    out_name = f"correlation_map_{base}_{START_YEAR}-{END_YEAR}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    print("正在绘制地图并保存：", out_path)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    world_map.plot(ax=ax, color="lightgray", edgecolor="white")

    plot = gdf_results.plot(
        column="correlation",
        ax=ax,
        legend=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        legend_kwds={
            "label": "Correlation Coefficient (TRW vs. {})".format(var_name),
            "orientation": "horizontal"
        }
    )

    # 北美视域
    ax.set_xlim(-167.27, -55.00)
    ax.set_ylim(9.58, 74.99)

    # 标题加入平均相关
    ax.set_title(
        f"Correlation between Tree Rings and {var_name} "
        f"({START_YEAR}-{END_YEAR})\nAverage correlation: {avg_corr:.2f}",
        fontsize=16
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("已保存：", out_path)


def main():
    # 读取基础数据（一次读取，复用）
    print("正在加载基础数据...")
    sites_df = pd.read_csv(PATH_ATTRIBUTES)
    tree_rings_df = pd.read_csv(PATH_TREE_RINGS)
    world_map = gpd.read_file(PATH_MAP_SHP)
    print("基础数据加载完成。")

    # 树轮设置索引，并切片到指定时间窗（做法与单文件一致）【参考：单文件实现】 
    if "year" not in tree_rings_df.columns:
        raise ValueError("树轮年表 CSV 中未找到 'year' 列。")
    tree_rings_df = tree_rings_df.set_index("year")
    tree_rings_period = tree_rings_df.loc[START_YEAR:END_YEAR]

    # 遍历目录下所有 .nc
    nc_files = sorted(glob.glob(os.path.join(DIR_NC, "*.nc")))
    if not nc_files:
        print("未在目录中找到 .nc 文件：", DIR_NC)
        return

    print(f"共发现 {len(nc_files)} 个 NC 文件，开始批量处理...")
    for nc in nc_files:
        compute_and_plot_for_nc(nc, sites_df, tree_rings_period, world_map)

    print("\n全部处理完成。")


if __name__ == "__main__":
    main()
