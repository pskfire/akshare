import akshare as ak

stock_code = "600519"
stock_name = "贵州茅台"

print(f"正在查询 {stock_name}（{stock_code}）的股市行情...\n")

try:
    df = ak.stock_zh_a_spot_em()
    stock_data = df[df['代码'] == stock_code]

    if not stock_data.empty:
        data = stock_data.iloc[0]
        print(f"股票名称: {data['名称']}")
        print(f"代码: {data['代码']}")
        print(f"最新价: {data['最新价']}")
        print(f"涨跌幅: {data['涨跌幅']}%")
        print(f"涨跌额: {data['涨跌额']}")
        print(f"成交量: {data['成交量']}")
        print(f"成交额: {data['成交额']}")
        print(f"开盘: {data['开盘']}")
        print(f"最高: {data['最高']}")
        print(f"最低: {data['最低']}")
        print(f"收盘价: {data['收盘']}")
        print(f"时间: {data['时间']}")
    else:
        print(f"未找到股票代码 {stock_code} 的数据")

except Exception as e:
    print(f"获取数据时出错: {e}")
    print("\n尝试获取历史K线数据...")

    try:
        df_hist = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                                     start_date="20230501", end_date="20260504",
                                     adjust="qfq")
        print("\n最近5个交易日的历史数据:")
        print(df_hist.tail())
    except Exception as e2:
        print(f"获取历史数据时出错: {e2}")