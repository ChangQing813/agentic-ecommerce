"""
库存决策 Agent
- 可用库存校验
- 安全库存预警
- 基于库存深度的动态限购
"""

from __future__ import annotations

from typing import Any

from models.schemas import InventoryResult, Product

from .base_agent import BaseAgent

SAFETY_STOCK_THRESHOLD = 50
LOW_STOCK_THRESHOLD = 100
HOT_ITEM_PURCHASE_LIMIT = 2


class InventoryAgent(BaseAgent):
    def __init__(self):
        from config import get_settings

        settings = get_settings()
        super().__init__(
            name="inventory",
            timeout=settings.agent_timeout_inventory,
        )

    async def _execute(self, **kwargs: Any) -> InventoryResult:
        products: list[Product] = kwargs.get("products", [])

        available = []
        low_stock_alerts = []
        purchase_limits: dict[str, int] = {}

        for product in products:
            stock = product.stock

            if stock <= 0:
                continue

            available.append(product.product_id)

            if stock <= SAFETY_STOCK_THRESHOLD:
                low_stock_alerts.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "current_stock": stock,
                    "level": "critical",
                    "action": "urgent_restock",
                })
            elif stock <= LOW_STOCK_THRESHOLD:
                low_stock_alerts.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "current_stock": stock,
                    "level": "warning",
                    "action": "plan_restock",
                })

            limit = self._calc_purchase_limit(product, stock)
            if limit is not None:
                purchase_limits[product.product_id] = limit

        return InventoryResult(
            success=True,
            available_products=available,
            low_stock_alerts=low_stock_alerts,
            purchase_limits=purchase_limits,
            data={
                "total_checked": len(products),
                "available_count": len(available),
                "alert_count": len(low_stock_alerts),
            },
            confidence=0.95,
        )

    def _calc_purchase_limit(self, product: Product, stock: int) -> int | None:
        """Dynamic purchase limit based on stock depth and product heat."""
        is_hot = "新品" in product.tags or "旗舰" in product.tags
        if stock <= SAFETY_STOCK_THRESHOLD:
            return 1
        if stock <= LOW_STOCK_THRESHOLD and is_hot:
            return HOT_ITEM_PURCHASE_LIMIT
        if is_hot and stock <= 300:
            return 3
        return None


