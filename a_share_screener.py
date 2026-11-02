"""
A股"低位 + 出清 + 基本面好转"初筛工具
思路：先找跌得深、处于低位的行业，再在这些行业里找估值低、财务扛得住、业绩开始好转的公司。
输出的是"研究候选清单"，不是买入建议，每个标的都需要人工做产业调查。

依赖：Python 3.8+，pip install akshare pandas
用法：python a_share_screener.py
"""
import time
import datetime as dt

import akshare as ak
import pandas as pd

# ---------------- 参数（可按需调整） ----------------
REPORT_DATE = "20260930"       # 最新财报期（三季报）
LOOKBACK_YEARS = 4             # 行业回看年数
MIN_DRAWDOWN = 0.40            # 行业指数距区间高点回撤至少 40%
MAX_POSITION = 0.35            # 行业指数处于区间价格分位 35% 以下（低位）
MIN_MKT_CAP = 50e8             # 总市值 >= 50 亿，过滤小票和垃圾股
MAX_PB = 2.0                   # 市净率上限
MIN_CFO_PER_SHARE = 0          # 每股经营现金流 > 0（真金白银在流入）
TOP_INDUSTRIES = 15            # 最多保留多少个低位行业


def industry_layer() -> pd.DataFrame:
    """第一层：找处于低位、回撤深的行业"""
    boards = ak.stock_board_industry_name_em()
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * LOOKBACK_YEARS)
    rows = []
    for name in boards["板块名称"]:
        try:
            h = ak.stock_board_industry_hist_em(
                symbol=name, start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"), period="日k", adjust="")
            close = h["收盘"].astype(float)
            hi, lo, last = close.max(), close.min(), close.iloc[-1]
            rows.append({
                "行业": name,
                "距高点回撤": 1 - last / hi,
                "区间分位": (last - lo) / (hi - lo) if hi > lo else 0,
                # 距区间最低点的反弹幅度：用来观察是否已开始止跌
                "距低点涨幅": last / lo - 1,
            })
        except Exception as e:  # 单个板块失败不影响整体
            print(f"[跳过行业] {name}: {e}")
        time.sleep(0.3)  # 避免请求过快被限流
    df = pd.DataFrame(rows)
    df = df[(df["距高点回撤"] >= MIN_DRAWDOWN) & (df["区间分位"] <= MAX_POSITION)]
    return df.sort_values("距高点回撤", ascending=False).head(TOP_INDUSTRIES)


def stock_layer(industries: pd.DataFrame) -> pd.DataFrame:
    """第二层：在低位行业里筛估值低、现金流好、业绩开始改善的公司"""
    fin = ak.stock_yjbb_em(date=REPORT_DATE)
    fin = fin.rename(columns={"股票代码": "代码"})
    keep = ["代码", "营业总收入-同比增长", "净利润-同比增长",
            "净利润-季度环比增长", "净资产收益率", "每股经营现金流量", "销售毛利率"]
    fin = fin[[c for c in keep if c in fin.columns]]

    spot = ak.stock_zh_a_spot_em()[["代码", "名称", "最新价", "总市值", "市净率", "市盈率-动态"]]

    out = []
    for name in industries["行业"]:
        try:
            cons = ak.stock_board_industry_cons_em(symbol=name)[["代码"]]
            cons["行业"] = name
            out.append(cons)
        except Exception as e:
            print(f"[跳过成分股] {name}: {e}")
        time.sleep(0.3)
    if not out:
        return pd.DataFrame()

    df = pd.concat(out).merge(spot, on="代码").merge(fin, on="代码", how="left")
    num_cols = [c for c in df.columns if c not in ("代码", "名称", "行业")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    df = df[
        (df["总市值"] >= MIN_MKT_CAP)
        & (df["市净率"] > 0) & (df["市净率"] <= MAX_PB)
        & (df["每股经营现金流量"] > MIN_CFO_PER_SHARE)
        & ~df["名称"].str.contains("ST|退", na=False)
    ]
    # "边际改善"信号：营收同比转正，且净利润环比改善（行业出清后的早期特征）
    df["边际改善"] = (df["营业总收入-同比增长"] > 0) & (df["净利润-季度环比增长"] > 0)
    return df.sort_values(["边际改善", "市净率"], ascending=[False, True])


if __name__ == "__main__":
    ind = industry_layer()
    print("\n=== 低位行业 ===")
    print(ind.to_string(index=False, float_format="%.2f"))

    stocks = stock_layer(ind)
    print(f"\n=== 候选公司（{len(stocks)} 家）===")
    print(stocks.head(50).to_string(index=False, float_format="%.2f"))

    stocks.to_csv("候选清单.csv", index=False, encoding="utf-8-sig")
    print("\n完整结果已保存到 候选清单.csv")
