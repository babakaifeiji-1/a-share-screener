"""
A股"低位 + 出清 + 基本面好转"初筛工具
思路：先找跌得深、处于低位的申万二级行业，再在这些行业里找估值低、现金流好、业绩开始好转的公司。
输出的是"研究候选清单"，不是买入建议，每个标的都需要人工做产业调查。

数据源：行业指数/成分股来自申万官网，财报来自东方财富业绩报表，行情快照来自东方财富（只请求一次）。
依赖：Python 3.8+，pip install akshare pandas
用法：python a_share_screener.py
"""
import time
import warnings
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import akshare as ak
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------- 参数（可按需调整） ----------------
REPORT_DATE = "20260930"       # 最新财报期（三季报）
LOOKBACK_YEARS = 4             # 行业回看年数
MIN_DRAWDOWN = 0.30            # 行业指数距区间高点回撤至少 30%
MAX_POSITION = 0.35            # 行业指数处于区间价格分位 35% 以下（低位）
MIN_MKT_CAP = 50e8             # 总市值 >= 50 亿，过滤小票
MAX_PB = 2.0                   # 个股市净率上限
MIN_CFO_PER_SHARE = 0          # 每股经营现金流 > 0
TOP_INDUSTRIES = 15            # 最多保留多少个低位行业
SLEEP = 0.5                    # 请求间隔，避免被限流
WORKERS = 6                    # 申万行业数据并发数


def retry(func, *args, tries=3, **kwargs):
    """简单重试：网络抖动或临时限流时等待后再试"""
    for i in range(tries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if i == tries - 1:
                raise
            wait = 5 * (i + 1)
            print(f"  请求失败（{e.__class__.__name__}），{wait}s 后重试")
            time.sleep(wait)


def _industry_stats(r, start):
    """单个行业：用周线计算回撤和区间分位。申万接口单次响应慢（约 10s），所以并发调用"""
    code = r["行业代码"].split(".")[0]
    try:
        h = retry(ak.index_hist_sw, symbol=code, period="week")
    except Exception as e:  # 单个行业失败不影响整体
        print(f"[跳过行业] {r['行业名称']}: {e.__class__.__name__}")
        return None
    h["日期"] = pd.to_datetime(h["日期"])
    close = h.loc[h["日期"] >= start, "收盘"].astype(float)
    if len(close) < 52:  # 上市不足一年的新指数不参与
        return None
    hi, lo, last = close.max(), close.min(), close.iloc[-1]
    return {
        "行业代码": code,
        "行业": r["行业名称"],
        "一级行业": r["上级行业"],
        "距高点回撤": 1 - last / hi,
        "区间分位": (last - lo) / (hi - lo) if hi > lo else 0,
        "距低点涨幅": last / lo - 1,   # 观察是否已开始止跌
        "行业PB": r["市净率"],
        "行业股息率": r["静态股息率"],
    }


def industry_layer() -> pd.DataFrame:
    """第一层：找处于低位、回撤深的申万二级行业"""
    info = retry(ak.sw_index_second_info)
    start = pd.Timestamp(dt.date.today() - dt.timedelta(days=365 * LOOKBACK_YEARS))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda r: _industry_stats(r, start), [r for _, r in info.iterrows()]))
    df = pd.DataFrame([x for x in results if x])
    print(f"共获取 {len(df)} 个行业的指数数据")
    if df.empty:
        return df
    df = df[(df["距高点回撤"] >= MIN_DRAWDOWN) & (df["区间分位"] <= MAX_POSITION)]
    return df.sort_values("距高点回撤", ascending=False).head(TOP_INDUSTRIES)


def stock_layer(industries: pd.DataFrame) -> pd.DataFrame:
    """第二层：在低位行业里筛估值低、现金流好、业绩开始改善的公司"""
    cons_list = []
    for _, r in industries.iterrows():
        try:
            c = retry(ak.index_component_sw, symbol=r["行业代码"])
            c = c.rename(columns={"证券代码": "代码"})[["代码"]]
            c["行业"] = r["行业"]
            cons_list.append(c)
        except Exception as e:
            print(f"[跳过成分股] {r['行业']}: {e}")
        time.sleep(SLEEP)
    if not cons_list:
        return pd.DataFrame()
    cons = pd.concat(cons_list)

    fin = retry(ak.stock_yjbb_em, date=REPORT_DATE).rename(columns={"股票代码": "代码"})
    fin = fin[["代码", "营业总收入-同比增长", "净利润-同比增长", "净利润-季度环比增长",
               "净资产收益率", "每股经营现金流量", "销售毛利率"]]

    spot = retry(ak.stock_zh_a_spot_em)[["代码", "名称", "最新价", "总市值", "市净率", "市盈率-动态"]]

    df = cons.merge(spot, on="代码").merge(fin, on="代码", how="left")
    num_cols = [c for c in df.columns if c not in ("代码", "名称", "行业")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    df = df[
        (df["总市值"] >= MIN_MKT_CAP)
        & (df["市净率"] > 0) & (df["市净率"] <= MAX_PB)
        & (df["每股经营现金流量"] > MIN_CFO_PER_SHARE)
        & ~df["名称"].str.contains("ST|退", na=False)
    ].copy()
    # "边际改善"信号：营收同比转正，且净利润环比改善（行业出清后的早期特征）
    df["边际改善"] = (df["营业总收入-同比增长"] > 0) & (df["净利润-季度环比增长"] > 0)
    df["总市值(亿)"] = df.pop("总市值") / 1e8
    return df.sort_values(["边际改善", "市净率"], ascending=[False, True])


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)

    ind = industry_layer()
    print("\n=== 低位行业 ===")
    print(ind.to_string(index=False, float_format="%.2f") if not ind.empty else "无符合条件的行业")

    stocks = stock_layer(ind) if not ind.empty else pd.DataFrame()
    print(f"\n=== 候选公司（{len(stocks)} 家）===")
    if not stocks.empty:
        print(stocks.head(50).to_string(index=False, float_format="%.2f"))
        stocks.to_csv("候选清单.csv", index=False, encoding="utf-8-sig")
        print("\n完整结果已保存到 候选清单.csv")
