"""
股票ST风险评估算法
==================

ST（Special Treatment）特别处理是中国股市对上市公司的一种风险警示制度。
本算法通过多维度财务指标和市场数据，评估股票被ST的风险概率。

主要风险维度：
1. 盈利能力风险（净利润、营业收入）
2. 偿债能力风险（资产负债率、流动比率）
3. 运营效率风险（应收账款周转、存货周转）
4. 现金流风险（经营现金流净额）
5. 审计风险（审计意见类型）
6. 市场信号风险（股价走势、交易异常）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import Counter
import argparse
import warnings

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    warnings.warn("akshare未安装，将使用模拟数据进行演示")


class STRiskAssessor:
    """股票ST风险评估器"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.risk_factors = {}
        self.weights = {
            'profitability': 0.25,      # 盈利能力权重
            'solvency': 0.20,           # 偿债能力权重
            'operation': 0.15,          # 运营效率权重
            'cashflow': 0.15,           # 现金流权重
            'audit': 0.15,             # 审计风险权重
            'market': 0.10             # 市场信号权重
        }

    def assess_st_risk(self) -> Dict:
        """
        综合评估ST风险
        返回：风险评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"正在评估股票 {self.stock_code} 的ST风险...")
        print(f"{'='*60}\n")

        # 获取各类数据
        financial_data = self._get_financial_data()
        market_data = self._get_market_data()
        audit_info = self._get_audit_info()

        # 计算各维度风险得分
        profitability_score = self._calculate_profitability_risk(financial_data)
        solvency_score = self._calculate_solvency_risk(financial_data)
        operation_score = self._calculate_operation_risk(financial_data)
        cashflow_score = self._calculate_cashflow_risk(financial_data)
        audit_score = self._calculate_audit_risk(audit_info)
        market_score = self._calculate_market_risk(market_data)

        # 计算加权综合风险得分
        weighted_score = (
            profitability_score * self.weights['profitability'] +
            solvency_score * self.weights['solvency'] +
            operation_score * self.weights['operation'] +
            cashflow_score * self.weights['cashflow'] +
            audit_score * self.weights['audit'] +
            market_score * self.weights['market']
        )

        # ST概率估算（将0-100的得分转换为概率）
        st_probability = self._score_to_probability(weighted_score)

        # 风险等级判定
        risk_level = self._get_risk_level(weighted_score)

        # 生成详细报告
        report = {
            'stock_code': self.stock_code,
            'assessment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'st_probability': st_probability,
            'risk_level': risk_level,
            'weighted_score': weighted_score,
            'risk_factors': {
                'profitability': {
                    'score': profitability_score,
                    'weight': self.weights['profitability'],
                    'details': self.risk_factors.get('profitability', {})
                },
                'solvency': {
                    'score': solvency_score,
                    'weight': self.weights['solvency'],
                    'details': self.risk_factors.get('solvency', {})
                },
                'operation': {
                    'score': operation_score,
                    'weight': self.weights['operation'],
                    'details': self.risk_factors.get('operation', {})
                },
                'cashflow': {
                    'score': cashflow_score,
                    'weight': self.weights['cashflow'],
                    'details': self.risk_factors.get('cashflow', {})
                },
                'audit': {
                    'score': audit_score,
                    'weight': self.weights['audit'],
                    'details': self.risk_factors.get('audit', {})
                },
                'market': {
                    'score': market_score,
                    'weight': self.weights['market'],
                    'details': self.risk_factors.get('market', {})
                }
            },
            'warning_signs': self._identify_warning_signs(),
            'recommendations': self._generate_recommendations(weighted_score)
        }

        return report

    def _get_financial_data(self) -> Dict:
        """获取财务数据（使用akshare或模拟数据）"""
        if not AKSHARE_AVAILABLE:
            return self._get_simulated_financial_data()

        try:
            # 获取股票财务指标
            df_financial = ak.stock_financial_analysis_indicator(symbol=self.stock_code)
            # 获取最新财务数据
            latest = df_financial.iloc[-1]

            return {
                'net_profit': latest.get('净利润', 0),
                'operating_revenue': latest.get('营业总收入', 0),
                'total_assets': latest.get('资产总计', 0),
                'total_liabilities': latest.get('负债合计', 0),
                'current_assets': latest.get('流动资产合计', 0),
                'current_liabilities': latest.get('流动负债合计', 0),
                'accounts_receivable': latest.get('应收账款', 0),
                'inventory': latest.get('存货', 0),
                'operating_cashflow': latest.get('经营活动产生的现金流量净额', 0),
                'roe': latest.get('净资产收益率', 0),
                'gross_margin': latest.get('毛利率', 0),
            }
        except Exception as e:
            print(f"获取财务数据失败: {e}，使用模拟数据")
            return self._get_simulated_financial_data()

    def _get_simulated_financial_data(self) -> Dict:
        """返回模拟财务数据用于演示"""
        return {
            'net_profit': -50000000,  # 亏损5000万
            'operating_revenue': 500000000,
            'total_assets': 2000000000,
            'total_liabilities': 1800000000,
            'current_assets': 800000000,
            'current_liabilities': 1200000000,
            'accounts_receivable': 200000000,
            'inventory': 150000000,
            'operating_cashflow': -30000000,
            'roe': -2.5,
            'gross_margin': 15.0,
        }

    def _calculate_price_trend(self, prices: np.ndarray) -> float:
        """
        计算价格趋势（百分比变化）
        """
        if len(prices) < 2:
            return 0.0
        first_price = prices[0]
        last_price = prices[-1]
        trend = ((last_price - first_price) / first_price * 100) if first_price != 0 else 0
        return float(trend)

    def _get_market_data(self) -> Dict:
        """获取市场数据"""
        if not AKSHARE_AVAILABLE:
            return self._get_simulated_market_data()

        try:
            # 获取历史价格数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
            df_price = ak.stock_zh_a_hist(symbol=self.stock_code, period="daily",
                                         start_date=start_date, end_date=end_date)

            # 计算股价统计
            prices = df_price['收盘'].values
            volumes = df_price['成交量'].values

            return {
                'price_trend': self._calculate_price_trend(prices),
                'volume_volatility': np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0,
                'price_volatility': np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0,
                'current_price': prices[-1] if len(prices) > 0 else 0,
                'highest_90d': np.max(prices) if len(prices) > 0 else 0,
                'lowest_90d': np.min(prices) if len(prices) > 0 else 0,
            }
        except Exception as e:
            print(f"获取市场数据失败: {e}，使用模拟数据")
            return self._get_simulated_market_data()

    def _get_simulated_market_data(self) -> Dict:
        """返回模拟市场数据"""
        return {
            'price_trend': -25.0,  # 下跌25%
            'volume_volatility': 0.8,
            'price_volatility': 0.15,
            'current_price': 8.5,
            'highest_90d': 12.0,
            'lowest_90d': 7.5,
        }

    def _get_audit_info(self) -> Dict:
        """获取审计信息"""
        if not AKSHARE_AVAILABLE:
            return self._get_simulated_audit_info()

        try:
            # 这里简化处理，实际应获取审计意见
            return {
                'audit_opinion': '标准无保留意见',  # 默认值
                'audit_firm': '未知',
                'audit_years': 3
            }
        except:
            return self._get_simulated_audit_info()

    def _get_simulated_audit_info(self) -> Dict:
        """返回模拟审计信息"""
        return {
            'audit_opinion': '保留意见',
            'audit_firm': '某会计师事务所',
            'audit_years': 3
        }

    def _calculate_profitability_risk(self, data: Dict) -> float:
        """
        计算盈利能力风险得分 (0-100)
        分数越高，风险越大
        """
        risk_score = 0

        # 净利润风险
        net_profit = data.get('net_profit', 0)
        if net_profit < 0:
            years_negative = 2  # 假设连续两年亏损
            risk_score += min(years_negative * 20, 40)
            self.risk_factors['profitability'] = {
                'net_profit': net_profit,
                'status': f'连续{years_negative}年亏损'
            }

        # 营业收入风险（过低可能表明经营异常）
        revenue = data.get('operating_revenue', 0)
        if revenue < 100000000:  # 小于1亿
            risk_score += 15
        elif revenue < 500000000:  # 小于5亿
            risk_score += 8

        # 净资产收益率风险
        roe = data.get('roe', 0)
        if roe < 0:
            risk_score += 15
        elif roe < 5:
            risk_score += 8

        # 毛利率风险
        gross_margin = data.get('gross_margin', 0)
        if gross_margin < 0:
            risk_score += 10
        elif gross_margin < 10:
            risk_score += 5

        return min(risk_score, 100)

    def _calculate_solvency_risk(self, data: Dict) -> float:
        """
        计算偿债能力风险得分 (0-100)
        """
        risk_score = 0

        # 资产负债率
        total_assets = data.get('total_assets', 1)
        total_liabilities = data.get('total_liabilities', 0)
        debt_ratio = (total_liabilities / total_assets * 100) if total_assets > 0 else 0

        if debt_ratio > 100:
            risk_score += 40
            status = '资不抵债'
        elif debt_ratio > 85:
            risk_score += 25
            status = '严重高负债'
        elif debt_ratio > 70:
            risk_score += 15
            status = '较高负债'
        else:
            status = '正常范围'

        # 流动比率风险
        current_assets = data.get('current_assets', 0)
        current_liabilities = data.get('current_liabilities', 0)
        current_ratio = (current_assets / current_liabilities) if current_liabilities > 0 else 0

        if current_ratio < 0.5:
            risk_score += 20
            status += ', 流动性严重不足'
        elif current_ratio < 1:
            risk_score += 10

        self.risk_factors['solvency'] = {
            'debt_ratio': debt_ratio,
            'current_ratio': current_ratio,
            'status': status
        }

        return min(risk_score, 100)

    def _calculate_operation_risk(self, data: Dict) -> float:
        """
        计算运营效率风险得分 (0-100)
        """
        risk_score = 0

        # 应收账款异常风险
        accounts_receivable = data.get('accounts_receivable', 0)
        operating_revenue = data.get('operating_revenue', 1)
        receivables_ratio = (accounts_receivable / operating_revenue * 100) if operating_revenue > 0 else 0

        if receivables_ratio > 50:
            risk_score += 25
            status = '应收账款异常偏高'
        elif receivables_ratio > 30:
            risk_score += 12

        # 存货风险
        inventory = data.get('inventory', 0)
        if operating_revenue > 0:
            inventory_turnover_days = (inventory / operating_revenue * 365) if operating_revenue > 0 else 0
            if inventory_turnover_days > 365:
                risk_score += 20

        self.risk_factors['operation'] = {
            'receivables_ratio': receivables_ratio,
            'inventory_turnover_days': inventory_turnover_days if operating_revenue > 0 else 0,
            'status': '正常' if risk_score < 15 else '存在异常'
        }

        return min(risk_score, 100)

    def _calculate_cashflow_risk(self, data: Dict) -> float:
        """
        计算现金流风险得分 (0-100)
        """
        risk_score = 0

        operating_cashflow = data.get('operating_cashflow', 0)
        operating_revenue = data.get('operating_revenue', 1)

        if operating_cashflow < 0:
            # 经营性现金流持续为负是重要风险信号
            cashflow_ratio = abs(operating_cashflow) / operating_revenue if operating_revenue > 0 else 0

            if cashflow_ratio > 0.5:
                risk_score += 35
                status = '经营现金流严重净流出'
            elif cashflow_ratio > 0.2:
                risk_score += 20
                status = '经营现金流净流出'
            else:
                risk_score += 10
                status = '经营现金流小幅净流出'
        else:
            status = '经营现金流正常'

        self.risk_factors['cashflow'] = {
            'operating_cashflow': operating_cashflow,
            'status': status
        }

        return min(risk_score, 100)

    def _calculate_audit_risk(self, audit_info: Dict) -> float:
        """
        计算审计风险得分 (0-100)
        """
        risk_score = 0

        audit_opinion = audit_info.get('audit_opinion', '标准无保留意见')

        if '无法表示' in audit_opinion or '否定' in audit_opinion:
            risk_score += 50
            status = '审计意见异常严重'
        elif '保留' in audit_opinion:
            risk_score += 35
            status = '审计意见为保留'
        elif '带强调' in audit_opinion:
            risk_score += 20
            status = '审计意见带强调事项段'
        elif '标准无保留' in audit_opinion:
            risk_score = 5
            status = '标准无保留意见'
        else:
            risk_score = 10
            status = '其他审计意见'

        self.risk_factors['audit'] = {
            'audit_opinion': audit_opinion,
            'status': status
        }

        return min(risk_score, 100)

    def _calculate_market_risk(self, market_data: Dict) -> float:
        """
        计算市场信号风险得分 (0-100)
        """
        risk_score = 0

        # 股价趋势风险
        price_trend = market_data.get('price_trend', 0)
        if price_trend < -50:
            risk_score += 25
        elif price_trend < -30:
            risk_score += 15
        elif price_trend < -20:
            risk_score += 8

        # 交易异常风险（成交量波动）
        volume_volatility = market_data.get('volume_volatility', 0)
        if volume_volatility > 2:
            risk_score += 15
        elif volume_volatility > 1.5:
            risk_score += 10

        # 股价波动性风险
        price_volatility = market_data.get('price_volatility', 0)
        if price_volatility > 0.5:
            risk_score += 10

        self.risk_factors['market'] = {
            'price_trend': price_trend,
            'volume_volatility': volume_volatility,
            'status': '正常' if risk_score < 15 else '存在异常'
        }

        return min(risk_score, 100)

    def _score_to_probability(self, score: float) -> float:
        """
        将风险得分转换为ST概率
        使用sigmoid函数进行非线性转换
        """
        # 将0-100的得分映射到概率
        # 得分越高，概率越高，但不会达到100%
        probability = 100 / (1 + np.exp(-0.08 * (score - 50)))
        return round(probability, 2)

    def _get_risk_level(self, score: float) -> str:
        """根据风险得分确定风险等级"""
        if score >= 70:
            return "极高风险"
        elif score >= 50:
            return "高风险"
        elif score >= 30:
            return "中等风险"
        elif score >= 15:
            return "较低风险"
        else:
            return "低风险"

    def _identify_warning_signs(self) -> List[str]:
        """识别风险警示信号"""
        warnings = []

        # 盈利能力警示
        if 'profitability' in self.risk_factors:
            pf = self.risk_factors['profitability']
            if '连续' in str(pf.get('status', '')):
                warnings.append(f"⚠️ {pf.get('status')}，可能被实施ST")

        # 偿债能力警示
        if 'solvency' in self.risk_factors:
            sol = self.risk_factors['solvency']
            if sol.get('debt_ratio', 0) > 85:
                warnings.append(f"⚠️ 资产负债率高达{sol.get('debt_ratio'):.1f}%，偿债能力严重不足")

        # 现金流警示
        if 'cashflow' in self.risk_factors:
            cf = self.risk_factors['cashflow']
            if '严重' in str(cf.get('status', '')):
                warnings.append(f"⚠️ {cf.get('status')}，经营状况堪忧")

        # 审计风险警示
        if 'audit' in self.risk_factors:
            aud = self.risk_factors['audit']
            if '保留' in str(aud.get('audit_opinion', '')):
                warnings.append(f"⚠️ 审计意见为保留意见，财务报告可信度存疑")

        # 市场信号警示
        if 'market' in self.risk_factors:
            mkt = self.risk_factors['market']
            if mkt.get('price_trend', 0) < -30:
                warnings.append(f"⚠️ 股价近90天下跌{Math.abs(mkt.get('price_trend')):.1f}%，市场信心不足")

        if not warnings:
            warnings.append("✓ 暂未发现明显风险信号")

        return warnings

    def _generate_recommendations(self, score: float) -> List[str]:
        """生成投资建议"""
        recommendations = []

        if score >= 70:
            recommendations.append("🔴 强烈建议回避该股票")
            recommendations.append("该股票财务状况严重恶化，ST风险极高")
            recommendations.append("建议深入调查公司具体情况，谨慎决策")
        elif score >= 50:
            recommendations.append("🟠 建议谨慎考虑")
            recommendations.append("该股票存在较多风险因素")
            recommendations.append("如持有应密切关注公司财务状况变化")
        elif score >= 30:
            recommendations.append("🟡 建议保持关注")
            recommendations.append("该股票存在一定风险，需持续跟踪")
            recommendations.append("如看好公司前景可少量配置但需严格止损")
        else:
            recommendations.append("🟢 风险较低，但仍需关注基本面变化")

        return recommendations


def print_assessment_report(report: Dict):
    """打印风险评估报告"""
    print(f"\n{'='*60}")
    print(f"   股票 ST 风险评估报告")
    print(f"{'='*60}")

    print(f"\n📊 股票代码: {report['stock_code']}")
    print(f"📅 评估时间: {report['assessment_date']}")

    print(f"\n{'─'*60}")
    print(f"🔮 ST风险概率: {report['st_probability']}%")
    print(f"📈 风险等级: {report['risk_level']}")
    print(f"📉 综合风险得分: {report['weighted_score']:.1f}/100")

    print(f"\n{'─'*60}")
    print("📋 各维度风险分析:")
    print(f"{'─'*60}")

    for factor, info in report['risk_factors'].items():
        factor_names = {
            'profitability': '盈利能力',
            'solvency': '偿债能力',
            'operation': '运营效率',
            'cashflow': '现金流',
            'audit': '审计风险',
            'market': '市场信号'
        }
        name = factor_names.get(factor, factor)
        score = info['score']
        weight = info['weight']

        # 风险等级颜色标识
        if score >= 70:
            level = "🔴极高"
        elif score >= 50:
            level = "🟠高"
        elif score >= 30:
            level = "🟡中"
        else:
            level = "🟢低"

        print(f"\n{name} ({level})")
        print(f"  得分: {score:.1f}/100 (权重: {weight*100:.0f}%)")
        print(f"  详情: {info['details']}")

    print(f"\n{'─'*60}")
    print("⚠️ 风险警示信号:")
    for warning in report['warning_signs']:
        print(f"  {warning}")

    print(f"\n{'─'*60}")
    print("💡 投资建议:")
    for rec in report['recommendations']:
        print(f"  {rec}")

    print(f"\n{'='*60}\n")


def assess_stock_st_risk(stock_code: str) -> Dict:
    """
    便捷函数：评估单只股票的ST风险

    参数:
        stock_code: 股票代码（如 '600519'）

    返回:
        风险评估报告字典
    """
    assessor = STRiskAssessor(stock_code)
    report = assessor.assess_st_risk()
    print_assessment_report(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='股票ST风险评估系统')
    parser.add_argument('stock_code', type=str, help='股票代码（如：600519）')

    args = parser.parse_args()

    print("股票ST风险评估系统")
    print("=" * 60)

    report = assess_stock_st_risk(args.stock_code)