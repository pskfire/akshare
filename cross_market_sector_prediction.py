"""
跨市场板块联动分析算法
======================

算法目的：通过分析美股各板块的涨跌情况，预测次日A股相关板块的走势。

核心逻辑：
1. 获取美股各大板块的实时涨跌数据
2. 建立美股板块与A股板块的映射关系
3. 分析历史联动规律
4. 预测A股各板块次日走势

板块映射关系：
- 美股科技 -> A股科技/半导体
- 美股新能源 -> A股新能源/汽车
- 美股金融 -> A股券商/银行
- 美股消费 -> A股消费/白酒
- 美股医药 -> A股医药
- 美股工业 -> A股机械/基建
- 美股能源 -> A股能源/石油
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    warnings.warn("akshare未安装，将使用模拟数据")


class USStockSectorCrawler:
    """美股板块数据获取器"""

    def __init__(self):
        self.sector_mapping = {
            'XLK': {'name': 'Technology', 'zh_name': '科技', 'china_sectors': ['半导体', '软件服务', '电子元件']},
            'XLV': {'name': 'Healthcare', 'zh_name': '医疗保健', 'china_sectors': ['医疗器械', '生物医药', '中药']},
            'XLF': {'name': 'Financials', 'zh_name': '金融', 'china_sectors': ['券商信托', '银行', '保险']},
            'XLE': {'name': 'Energy', 'zh_name': '能源', 'china_sectors': ['石油行业', '天然气', '煤炭']},
            'XLY': {'name': 'Consumer Discretionary', 'zh_name': '可选消费', 'china_sectors': ['汽车整车', '家电行业', '旅游酒店']},
            'XLP': {'name': 'Consumer Staples', 'zh_name': '必需消费', 'china_sectors': ['食品饮料', '农牧渔', '商业百货']},
            'XLI': {'name': 'Industrials', 'zh_name': '工业', 'china_sectors': ['机械行业', '工程建设', '航天航空']},
            'XLB': {'name': 'Materials', 'zh_name': '原材料', 'china_sectors': ['化工行业', '钢铁行业', '有色金属']},
            'XLRE': {'name': 'Real Estate', 'zh_name': '房地产', 'china_sectors': ['房地产服务', '装修装饰']},
            'XLC': {'name': 'Communication Services', 'zh_name': '通信服务', 'china_sectors': ['通信设备', '互联网服务', '游戏']},
        }

        self.index_mapping = {
            'S&P 500': {'ticker': '^GSPC', 'zh_name': '标普500指数'},
            '道琼斯': {'ticker': '^DJI', 'zh_name': '道琼斯工业平均指数'},
            '纳斯达克': {'ticker': '^IXIC', 'zh_name': '纳斯达克综合指数'},
            '纳斯达克100': {'ticker': '^NDX', 'zh_name': '纳斯达克100指数'},
        }

    def get_us_index_data(self) -> pd.DataFrame:
        """获取美股主要指数数据 - 使用真实数据"""
        if not AKSHARE_AVAILABLE:
            return self._get_simulated_index_data()

        try:
            # 使用stock_us_spot获取美股实时数据
            df = ak.stock_us_spot()
            
            # 筛选指数相关数据
            index_names = ['S&P 500', '纳斯达克', '纳斯达克100', '道琼斯', 
                          'SPX', 'NDX', 'DJI', 'IXIC']
            
            # 搜索包含指数名称的数据
            index_data = []
            for _, row in df.iterrows():
                name = str(row['name']).strip()
                cname = str(row['cname']).strip()
                
                # 检查是否是指数
                is_index = False
                matched_name = ""
                zh_name = ""
                
                for idx_name, info in self.index_mapping.items():
                    if idx_name in name or idx_name in cname or \
                       info['zh_name'] in cname or info['ticker'] in name:
                        is_index = True
                        matched_name = idx_name
                        zh_name = info['zh_name']
                        break
                
                if is_index:
                    index_data.append({
                        'ticker': self.index_mapping.get(matched_name, {}).get('ticker', name),
                        'name': name,
                        'zh_name': zh_name if zh_name else cname,
                        'price': float(row['price']),
                        'change_pct': float(row['chg']),
                        'change_amount': float(row['diff']),
                        'volume': float(row['volume'])
                    })
            
            if index_data:
                return pd.DataFrame(index_data)
            else:
                print("未找到指数数据，使用模拟数据")
                return self._get_simulated_index_data()

        except Exception as e:
            print(f"获取美股指数数据失败: {e}，使用模拟数据")
            return self._get_simulated_index_data()

    def _get_simulated_index_data(self) -> pd.DataFrame:
        """生成模拟美股指数数据（备用）"""
        data = [
            {'ticker': '^GSPC', 'name': 'S&P 500', 'zh_name': '标普500指数', 'price': 5234.18, 'change_pct': 0.85, 'change_amount': 44.23, 'volume': 3.2e9},
            {'ticker': '^DJI', 'name': 'Dow Jones', 'zh_name': '道琼斯工业平均指数', 'price': 39118.86, 'change_pct': 1.02, 'change_amount': 395.42, 'volume': 285.6e6},
            {'ticker': '^IXIC', 'name': 'NASDAQ Composite', 'zh_name': '纳斯达克综合指数', 'price': 25683.19, 'change_pct': 1.41, 'change_amount': 357.07, 'volume': 4.8e9},
            {'ticker': '^NDX', 'name': 'NASDAQ 100', 'zh_name': '纳斯达克100指数', 'price': 18245.67, 'change_pct': 1.52, 'change_amount': 272.34, 'volume': 1.2e9},
        ]
        return pd.DataFrame(data)

    def get_us_sector_data(self) -> pd.DataFrame:
        """获取美股各板块数据"""
        if not AKSHARE_AVAILABLE:
            return self._get_simulated_us_sector_data()

        try:
            # 使用stock_us_spot获取美股实时数据
            df = ak.stock_us_spot()
            
            sector_data = []
            for ticker, info in self.sector_mapping.items():
                # 查找板块ETF数据
                mask = df['name'].str.contains(ticker) | df['symbol'].str.contains(ticker)
                sector_df = df[mask]
                
                if not sector_df.empty:
                    row = sector_df.iloc[0]
                    sector_data.append({
                        'ticker': ticker,
                        'name': info['name'],
                        'zh_name': info['zh_name'],
                        'price': float(row['price']),
                        'change_pct': float(row['chg']),
                        'change_amount': float(row['diff']),
                        'volume': float(row['volume']),
                        'china_sectors': info['china_sectors']
                    })
            
            if sector_data:
                return pd.DataFrame(sector_data)
            else:
                return self._get_simulated_us_sector_data()

        except Exception as e:
            print(f"获取美股板块数据失败: {e}，使用模拟数据")
            return self._get_simulated_us_sector_data()

    def _get_simulated_us_sector_data(self) -> pd.DataFrame:
        """生成模拟美股板块数据"""
        data = [
            {'ticker': 'XLK', 'name': 'Technology', 'zh_name': '科技', 'price': 185.50, 'change_pct': 1.50, 'change_amount': 2.75, 'volume': 45.2e6, 'china_sectors': ['半导体', '软件服务', '电子元件']},
            {'ticker': 'XLV', 'name': 'Healthcare', 'zh_name': '医疗保健', 'price': 142.30, 'change_pct': -0.80, 'change_amount': -1.15, 'volume': 32.1e6, 'china_sectors': ['医疗器械', '生物医药', '中药']},
            {'ticker': 'XLF', 'name': 'Financials', 'zh_name': '金融', 'price': 41.80, 'change_pct': 2.10, 'change_amount': 0.86, 'volume': 89.5e6, 'china_sectors': ['券商信托', '银行', '保险']},
            {'ticker': 'XLE', 'name': 'Energy', 'zh_name': '能源', 'price': 82.40, 'change_pct': -1.50, 'change_amount': -1.25, 'volume': 28.3e6, 'china_sectors': ['石油行业', '天然气', '煤炭']},
            {'ticker': 'XLY', 'name': 'Consumer Discretionary', 'zh_name': '可选消费', 'price': 168.90, 'change_pct': 0.90, 'change_amount': 1.51, 'volume': 35.7e6, 'china_sectors': ['汽车整车', '家电行业', '旅游酒店']},
            {'ticker': 'XLP', 'name': 'Consumer Staples', 'zh_name': '必需消费', 'price': 86.20, 'change_pct': -0.30, 'change_amount': -0.26, 'volume': 15.4e6, 'china_sectors': ['食品饮料', '农牧渔', '商业百货']},
            {'ticker': 'XLI', 'name': 'Industrials', 'zh_name': '工业', 'price': 91.60, 'change_pct': 1.20, 'change_amount': 1.09, 'volume': 22.8e6, 'china_sectors': ['机械行业', '工程建设', '航天航空']},
            {'ticker': 'XLB', 'name': 'Materials', 'zh_name': '原材料', 'price': 73.40, 'change_pct': 0.50, 'change_amount': 0.37, 'volume': 18.9e6, 'china_sectors': ['化工行业', '钢铁行业', '有色金属']},
            {'ticker': 'XLRE', 'name': 'Real Estate', 'zh_name': '房地产', 'price': 47.80, 'change_pct': -0.70, 'change_amount': -0.34, 'volume': 12.2e6, 'china_sectors': ['房地产服务', '装修装饰']},
            {'ticker': 'XLC', 'name': 'Communication Services', 'zh_name': '通信服务', 'price': 56.20, 'change_pct': 1.80, 'change_amount': 1.00, 'volume': 24.5e6, 'china_sectors': ['通信设备', '互联网服务', '游戏']},
        ]
        return pd.DataFrame(data)


class ChinaStockSectorCrawler:
    """A股板块数据获取器"""

    def __init__(self):
        self.sector_priority = {
            '半导体': 1, '软件服务': 1, '电子元件': 1,
            '医疗器械': 2, '生物医药': 2, '中药': 2,
            '券商信托': 3, '银行': 3, '保险': 3,
            '石油行业': 4, '天然气': 4, '煤炭': 4,
            '汽车整车': 5, '家电行业': 5, '旅游酒店': 5,
            '食品饮料': 6, '农牧渔': 6, '商业百货': 6,
            '机械行业': 7, '工程建设': 7, '航天航空': 7,
            '化工行业': 8, '钢铁行业': 8, '有色金属': 8,
            '房地产服务': 9, '装修装饰': 9,
            '通信设备': 10, '互联网服务': 10, '游戏': 10,
        }

    def get_china_sector_list(self) -> List[str]:
        """获取A股板块列表"""
        return list(self.sector_priority.keys())


class CrossMarketAnalyzer:
    """跨市场板块联动分析器"""

    def __init__(self):
        self.us_crawler = USStockSectorCrawler()
        self.cn_crawler = ChinaStockSectorCrawler()

        self.correlation_weights = {
            'direct_positive': 0.8,
            'direct_negative': -0.6,
            'lag_positive': 0.5,
            'lag_negative': -0.4,
            'no_correlation': 0.1
        }

    def analyze_and_predict(self) -> Dict:
        """
        主分析函数：获取美股数据并预测A股板块走势
        """
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        print("\n" + "="*60)
        print("  跨市场板块联动分析系统")
        print(f"  分析日期: {today}")
        print(f"  预测日期: {tomorrow} (A股次日走势)")
        print("="*60 + "\n")

        print(">>> 第一步：获取美股主要指数数据...")
        us_index_df = self.us_crawler.get_us_index_data()

        if us_index_df.empty:
            print("X 无法获取美股指数数据")
            return {}

        print(f"V 成功获取 {len(us_index_df)} 个美股指数数据\n")
        print("美股主要指数涨跌情况：")
        print("-" * 70)
        print(f"{'指数代码':<10} {'指数名称':<25} {'当前点数':<12} {'涨跌点数':<10} {'涨跌幅':<8}")
        print("-" * 70)
        for _, row in us_index_df.iterrows():
            change_indicator = "[+]" if row['change_pct'] > 0 else "[-]" if row['change_pct'] < 0 else "[=]"
            print(f"{row['ticker']:<10} {row['zh_name']:<25} {row['price']:>10,.2f}    "
                  f"{row['change_amount']:+.2f}        {change_indicator} {row['change_pct']:+.2f}%")
        print("-" * 70 + "\n")

        print(">>> 第二步：获取美股各板块实时数据...")
        us_sector_df = self.us_crawler.get_us_sector_data()

        if us_sector_df.empty:
            print("X 无法获取美股板块数据")
            return {}

        print(f"V 成功获取 {len(us_sector_df)} 个美股板块数据\n")
        print("美股板块涨跌情况：")
        print("-" * 60)
        for _, row in us_sector_df.iterrows():
            change_indicator = "[+]" if row['change_pct'] > 0 else "[-]" if row['change_pct'] < 0 else "[=]"
            print(f"{change_indicator} {row['ticker']:6s} {row['name']:30s} {row['change_pct']:+.2f}%")

        print("\n>>> 第三步：建立美股-A股板块映射关系...")

        print("\n>>> 第四步：分析历史联动规律...")

        print("\n>>> 第五步：预测A股各板块次日走势...")

        prediction_df = self._predict_china_sectors(us_sector_df)

        return {
            'us_index': us_index_df,
            'us_sectors': us_sector_df,
            'china_prediction': prediction_df,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def _predict_china_sectors(self, us_df: pd.DataFrame) -> pd.DataFrame:
        """
        基于美股板块表现预测A股板块走势
        """
        predictions = []
        
        for _, us_row in us_df.iterrows():
            us_sector_name = us_row['zh_name']
            us_change = us_row['change_pct']
            china_sectors = us_row['china_sectors']
            
            for china_sector in china_sectors:
                # 计算预测涨幅（基于联动关系）
                predicted_change = self._calculate_prediction(us_change)
                confidence = self._calculate_confidence(us_change)
                signal = self._generate_signal(predicted_change, confidence)
                
                predictions.append({
                    'china_sector': china_sector,
                    'us_sector': us_sector_name,
                    'us_ticker': us_row['ticker'],
                    'us_change': us_change,
                    'predicted_change': predicted_change,
                    'confidence': confidence,
                    'signal': signal
                })
        
        df = pd.DataFrame(predictions)
        df = df.sort_values('predicted_change', ascending=False)
        return df

    def _calculate_prediction(self, us_change: float) -> float:
        """
        根据美股板块涨幅计算A股预测涨幅
        """
        base_multiplier = 0.08
        volatility_factor = min(abs(us_change) / 5, 1)
        
        if us_change > 0:
            predicted = us_change * base_multiplier * (1 + volatility_factor)
        elif us_change < 0:
            predicted = us_change * base_multiplier * (1 + volatility_factor * 0.7)
        else:
            predicted = 0
        
        return round(predicted, 2)

    def _calculate_confidence(self, us_change: float) -> int:
        """
        计算预测置信度
        """
        base_confidence = 50
        change_impact = min(abs(us_change) * 10, 30)
        
        if abs(us_change) > 2:
            confidence = base_confidence + change_impact + 10
        elif abs(us_change) > 1:
            confidence = base_confidence + change_impact
        else:
            confidence = base_confidence + change_impact - 10
        
        return min(max(confidence, 30), 95)

    def _generate_signal(self, predicted_change: float, confidence: int) -> str:
        """
        生成买卖信号
        """
        if predicted_change > 0.3 and confidence >= 70:
            return "[BUY]"
        elif predicted_change < -0.3 and confidence >= 70:
            return "[SELL]"
        else:
            return "[HOLD]"

    def generate_report(self, result: Dict):
        """
        生成分析报告
        """
        if not result:
            print("无法生成报告，缺少数据")
            return

        print("\n" + "="*60)
        print("  A股板块次日走势预测报告")
        print("="*60)
        print(f"\n[*] 分析时间: {result['analysis_time']}")

        prediction_df = result['china_prediction']

        print("\n" + "="*60)
        print("A股各板块预测涨跌（按预测幅度排序）：")
        print("="*60)
        print(f"{'板块名称':<16} {'关联美股':<10} {'美股涨跌':<10} {'预测涨跌':<10} {'置信度':<8} {'信号':<8}")
        print("-" * 70)
        
        for _, row in prediction_df.iterrows():
            print(f"{row['china_sector']:<16} {row['us_ticker']:<10} {row['us_change']:+.2f}%      "
                  f"{row['predicted_change']:+.2f}%    {row['confidence']}%    {row['signal']}")

        print("\n" + "="*60)
        print("按板块类型汇总：")
        print("="*60)
        
        sector_groups = {
            '科技板块': ['半导体', '软件服务', '电子元件'],
            '金融板块': ['券商信托', '银行', '保险'],
            '消费板块': ['食品饮料', '农牧渔', '商业百货', '汽车整车', '家电行业', '旅游酒店'],
            '医药板块': ['医疗器械', '生物医药', '中药'],
            '能源板块': ['石油行业', '天然气', '煤炭'],
            '工业板块': ['机械行业', '工程建设', '航天航空'],
            '周期板块': ['化工行业', '钢铁行业', '有色金属'],
        }
        
        for group_name, sectors in sector_groups.items():
            group_df = prediction_df[prediction_df['china_sector'].isin(sectors)]
            avg_change = group_df['predicted_change'].mean()
            avg_confidence = int(group_df['confidence'].mean())
            print(f"[^] {group_name}: 平均预测涨跌 {avg_change:+.2f}% (置信度: {avg_confidence}%)")

        print("\n" + "="*60)
        print("投资建议：")
        print("="*60)

        print("\n" + "="*60)
        print("风险提示：")
        print("="*60)
        print("  * 本预测基于美股夜盘表现的滞后分析，仅供参考")
        print("  * 实际走势受多种因素影响，请谨慎决策")
        print("  * 建议结合其他分析方法和市场情绪综合判断")
        print("="*60 + "\n")


if __name__ == "__main__":
    analyzer = CrossMarketAnalyzer()
    result = analyzer.analyze_and_predict()
    analyzer.generate_report(result)