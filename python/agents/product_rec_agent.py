"""
商品推荐 Agent
- 候选筛选：根据用户偏好与库存状态完成规则召回
- 排序层：LLM 重排 + 用户画像特征交叉
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from models.schemas import Product, ProductRecResult, UserProfile

from .base_agent import BaseAgent

RERANK_PROMPT = """你是电商推荐排序专家。根据用户画像和候选商品,重新排序并选出最优的{num_items}个商品。

用户画像:
{user_profile}

候选商品:
{candidates}

排序原则:
1. 用户偏好类目优先
2. 价格在用户可接受范围内
3. 保证类目多样性(相邻商品尽量不同类目)
4. 新品适当加权

请输出商品ID列表(JSON数组),按推荐优先级排序:
["product_id_1", "product_id_2", ...]

只输出JSON数组,不要其他内容。"""

MOCK_PRODUCTS = [
    Product(product_id="P001", name="iPhone 16 Pro", category="手机", price=7999, brand="Apple", seller_id="S01", stock=500, tags=["旗舰", "新品"]),
    Product(product_id="P002", name="华为 Mate 70", category="手机", price=5999, brand="华为", seller_id="S02", stock=300, tags=["旗舰", "国产"]),
    Product(product_id="P003", name="AirPods Pro 3", category="耳机", price=1899, brand="Apple", seller_id="S01", stock=1000, tags=["降噪", "无线"]),
    Product(product_id="P004", name="Sony WH-1000XM6", category="耳机", price=2499, brand="Sony", seller_id="S03", stock=200, tags=["头戴", "降噪"]),
    Product(product_id="P005", name="iPad Air M3", category="平板", price=4799, brand="Apple", seller_id="S01", stock=400, tags=["学习", "办公"]),
    Product(product_id="P006", name="小米平板7 Pro", category="平板", price=2499, brand="小米", seller_id="S04", stock=600, tags=["性价比", "娱乐"]),
    Product(product_id="P007", name="Anker 140W充电器", category="配件", price=399, brand="Anker", seller_id="S05", stock=2000, tags=["快充", "便携"]),
    Product(product_id="P008", name="机械革命极光X", category="笔记本", price=6999, brand="机械革命", seller_id="S06", stock=150, tags=["游戏", "高性能"]),
    Product(product_id="P009", name="戴尔U2724D显示器", category="显示器", price=3299, brand="Dell", seller_id="S07", stock=80, tags=["4K", "办公"]),
    Product(product_id="P010", name="罗技MX Master 3S", category="配件", price=749, brand="罗技", seller_id="S08", stock=500, tags=["无线", "办公"]),
    Product(product_id="P011", name="三星980 Pro 2TB", category="存储", price=1199, brand="三星", seller_id="S09", stock=300, tags=["SSD", "高速"]),
    Product(product_id="P012", name="绿联氮化镓65W", category="配件", price=129, brand="绿联", seller_id="S10", stock=5000, tags=["快充", "性价比"]),
    Product(product_id="P013", name="Apple Watch Ultra 3", category="穿戴", price=5999, brand="Apple", seller_id="S01", stock=200, tags=["运动", "健康"]),
    Product(product_id="P014", name="大疆Mini 4 Pro", category="无人机", price=4788, brand="大疆", seller_id="S11", stock=100, tags=["航拍", "便携"]),
    Product(product_id="P015", name="Switch 2", category="游戏机", price=2499, brand="Nintendo", seller_id="S12", stock=50, tags=["新品", "游戏"]),
]


class ProductRecAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(
            name="product_rec",
            timeout=settings.agent_timeout_product_rec,
        )
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=512,
        )

    async def _execute(self, **kwargs: Any) -> ProductRecResult:
        operation: str = kwargs["operation"]

        if operation == "recall":
            limit = int(kwargs["limit"])
            candidates = self._load_candidates(limit)
            return ProductRecResult(
                success=True,
                products=candidates,
                recall_strategy="rule_recall",
                data={"candidate_count": len(candidates), "reranked": 0},
                confidence=0.8,
            )

        if operation != "rerank":
            raise ValueError(
                f"Unsupported ProductRecAgent operation: {operation}"
            )

        user_profile: UserProfile | None = kwargs.get("user_profile")
        candidates: list[Product] | None = kwargs.get("candidates")
        num_items: int = kwargs.get("num_items", 10)

        if candidates is None:
            raise ValueError("ProductRecAgent.rerank requires candidates")
        candidates = list(candidates)

        if not candidates:
            return ProductRecResult(
                success=True,
                products=[],
                recall_strategy="rule_recall",
                data={"candidate_count": 0, "reranked": 0},
                confidence=1.0,
            )

        ranked_ids = await self._rerank(
            user_profile,
            candidates,
            num_items,
        )

        id_to_product = {p.product_id: p for p in candidates}
        final_products = []
        seen_ids = set()

        for pid in ranked_ids:
            if pid in id_to_product and pid not in seen_ids:
                seen_ids.add(pid)
                final_products.append(id_to_product[pid])

        if len(final_products) < num_items:
            for product in candidates:
                if product.product_id not in seen_ids:
                    seen_ids.add(product.product_id)
                    final_products.append(product)
                    if len(final_products) >= num_items:
                        break

        return ProductRecResult(
            success=True,
            products=final_products[:num_items],
            recall_strategy="rule_recall+llm_rerank",
            data={
                "candidate_count": len(candidates),
                "reranked": len(ranked_ids),
            },
            confidence=0.8,
        )

    def _load_candidates(self, limit: int) -> list[Product]:
        """Load a deterministic candidate slice without ranking it."""
        return list(MOCK_PRODUCTS)[:limit]

    async def _rerank(
        self,
        user_profile: UserProfile | None,
        candidates: list[Product],
        num_items: int,
    ) -> list[str]:
        """Use LLM ranking first; use deterministic fallback on failure."""
        if not user_profile:
            return self._fallback_rank(candidates, None, num_items)

        try:
            profile_summary = {
                "segments": [s.value for s in user_profile.segments],
                "preferred_categories": user_profile.preferred_categories,
                "price_range": list(user_profile.price_range),
            }
            candidate_summary = [
                {
                    "id": p.product_id,
                    "name": p.name,
                    "category": p.category,
                    "price": p.price,
                    "tags": p.tags,
                }
                for p in candidates
            ]
            prompt = RERANK_PROMPT.format(
                num_items=num_items,
                user_profile=json.dumps(profile_summary, ensure_ascii=False),
                candidates=json.dumps(candidate_summary, ensure_ascii=False),
            )
            messages = [
                SystemMessage(content="你是电商推荐排序专家。"),
                HumanMessage(content=prompt),
            ]
            response = await self.llm.ainvoke(messages)

            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

            ranked_ids = json.loads(raw)
            if not isinstance(ranked_ids, list):
                raise ValueError("LLM rerank output must be a JSON array")
            return ranked_ids
        except (json.JSONDecodeError, IndexError, TypeError, ValueError):
            return self._fallback_rank(candidates, user_profile, num_items)

    def _fallback_rank(
        self,
        candidates: list[Product],
        user_profile: UserProfile | None,
        num_items: int,
    ) -> list[str]:
        """Deterministic fallback used only when LLM ranking is unavailable."""
        ordered = list(candidates)
        ordered.sort(key=lambda p: p.product_id)

        if user_profile and user_profile.preferred_categories:
            preferred = set(user_profile.preferred_categories)
            price_min, price_max = user_profile.price_range
            ordered.sort(
                key=lambda p: (
                    p.category in preferred,
                    p.stock > 0,
                    price_min <= p.price <= price_max,
                ),
                reverse=True,
            )

        return [p.product_id for p in ordered[:num_items]]
