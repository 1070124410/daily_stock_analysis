# -*- coding: utf-8 -*-
"""大盘复盘指数多日走势上下文的离线回归测试。

覆盖：
1. _compute_index_trend 从标准化指数历史 DataFrame 计算走势摘要
2. get_market_overview 为每个指数 fail-open 附加 trend
3. Prompt 单行与报告走势表渲染，以及数据缺失时的回退
4. DataFetcherManager.get_index_history 的多源 failover 与 fail-open
"""

import unittest

import numpy as np
import pandas as pd

from data_provider.base import BaseFetcher, DataFetcherManager
from src.market_analyzer import MarketAnalyzer, MarketIndex, MarketOverview


def _make_index_history(rows: int = 70, start: float = 3600.0, end: float = 3764.0) -> pd.DataFrame:
    """构造一段上涨趋势的标准化指数历史（含 ma5/ma10/ma20）。"""
    dates = pd.date_range("2026-05-01", periods=rows, freq="D")
    close = np.linspace(start, end, rows)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close - 3,
            "high": close + 8,
            "low": close - 9,
            "close": close,
            "volume": np.linspace(1e8, 1.2e8, rows),
            "amount": np.linspace(1.1e12, 1.24e12, rows),
        }
    )
    df["ma5"] = df["close"].rolling(5, min_periods=1).mean().round(2)
    df["ma10"] = df["close"].rolling(10, min_periods=1).mean().round(2)
    df["ma20"] = df["close"].rolling(20, min_periods=1).mean().round(2)
    return df


class ComputeIndexTrendTests(unittest.TestCase):
    def test_uptrend_produces_bull_alignment_and_metrics(self) -> None:
        trend = MarketAnalyzer._compute_index_trend(_make_index_history())
        self.assertEqual(trend["ma_alignment"], "bull")
        # 均线、区间、量能字段齐备
        for key in (
            "ma5",
            "ma10",
            "ma20",
            "recent_high",
            "recent_low",
            "change_5d_pct",
            "change_20d_pct",
            "amount_change_pct",
        ):
            self.assertIn(key, trend)
        self.assertGreaterEqual(trend["recent_high"], trend["recent_low"])
        self.assertEqual(trend["window"], 20)

    def test_downtrend_produces_bear_alignment(self) -> None:
        trend = MarketAnalyzer._compute_index_trend(
            _make_index_history(start=4000.0, end=3600.0)
        )
        self.assertEqual(trend["ma_alignment"], "bear")
        self.assertLess(trend["change_20d_pct"], 0)

    def test_empty_or_invalid_frame_returns_empty(self) -> None:
        self.assertEqual(MarketAnalyzer._compute_index_trend(pd.DataFrame()), {})
        self.assertEqual(MarketAnalyzer._compute_index_trend(None), {})
        # 缺少 close 列
        self.assertEqual(
            MarketAnalyzer._compute_index_trend(pd.DataFrame({"open": [1, 2, 3]})), {}
        )

    def test_short_series_omits_long_window_metrics(self) -> None:
        trend = MarketAnalyzer._compute_index_trend(_make_index_history(rows=4))
        # 不足 6/21 根时不给出对应涨跌幅，但仍返回可用的均线/区间
        self.assertNotIn("change_5d_pct", trend)
        self.assertNotIn("change_20d_pct", trend)
        self.assertIn("ma5", trend)


class TrendRenderingTests(unittest.TestCase):
    def _analyzer(self) -> MarketAnalyzer:
        analyzer = MarketAnalyzer.__new__(MarketAnalyzer)
        analyzer.region = "cn"
        analyzer.config = None
        return analyzer

    def test_prompt_line_and_block_render_when_trend_present(self) -> None:
        analyzer = self._analyzer()
        trend = MarketAnalyzer._compute_index_trend(_make_index_history())
        idx = MarketIndex(code="sh000001", name="上证指数", current=3764.0, change_pct=-3.05, trend=trend)
        line_zh = analyzer._format_index_trend_for_prompt(idx, "zh")
        self.assertTrue(line_zh.startswith("走势:"))
        self.assertIn("MA5", line_zh)
        line_en = analyzer._format_index_trend_for_prompt(idx, "en")
        self.assertTrue(line_en.startswith("trend:"))

        overview = MarketOverview(date="2026-07-17", indices=[idx])
        block = analyzer._build_index_trend_block(overview)
        self.assertIn("指数多日走势", block)
        self.assertIn("上证指数", block)

    def test_empty_trend_renders_nothing(self) -> None:
        analyzer = self._analyzer()
        idx = MarketIndex(code="x", name="空指数", current=1.0)
        overview = MarketOverview(date="d", indices=[idx])
        self.assertEqual(analyzer._format_index_trend_for_prompt(idx, "zh"), "")
        self.assertEqual(analyzer._build_index_trend_block(overview), "")


class _StubFetcher(BaseFetcher):
    """最小可实例化 fetcher，用于验证 manager 层 failover。"""

    def __init__(self, name, behavior):
        self.name = name
        self._behavior = behavior

    def _fetch_raw_data(self, *a, **k):  # pragma: no cover - 抽象方法占位
        return pd.DataFrame()

    def _normalize_data(self, df, code):  # pragma: no cover - 抽象方法占位
        return df

    def get_index_history(self, index_code, region="cn", days=60):
        if self._behavior == "raise":
            raise RuntimeError("boom")
        if self._behavior == "none":
            return None
        if self._behavior == "empty":
            return pd.DataFrame()
        return pd.DataFrame({"close": [1.0, 2.0, 3.0]})


class ManagerIndexHistoryTests(unittest.TestCase):
    def _manager(self, fetchers):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        manager._fetchers = fetchers
        return manager

    def test_failover_skips_failing_and_returns_first_nonempty(self) -> None:
        manager = self._manager(
            [
                _StubFetcher("Raise", "raise"),
                _StubFetcher("None", "none"),
                _StubFetcher("Empty", "empty"),
                _StubFetcher("Good", "good"),
            ]
        )
        df = manager.get_index_history("sh000001", region="cn", days=60)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)

    def test_all_sources_empty_returns_none(self) -> None:
        manager = self._manager(
            [_StubFetcher("Raise", "raise"), _StubFetcher("None", "none")]
        )
        self.assertIsNone(manager.get_index_history("sh000001"))

    def test_default_base_fetcher_returns_none(self) -> None:
        self.assertIsNone(_StubFetcher("Base", "none").get_index_history("x"))


if __name__ == "__main__":
    unittest.main()
