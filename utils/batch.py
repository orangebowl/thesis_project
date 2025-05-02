# utils/batch.py  
import jax, jax.numpy as jnp
from typing import Sequence, Union

Colloc = Union[jnp.ndarray, Sequence[jnp.ndarray]]   # ndarray 或 list[ndarray]


class BatchSampler:
    """
    strategy:
      equal  – 每子域同样多   (bs // n_sub)
      global – 全域随机后再拆桶
      fixed  – 按各子域数据量比例一次性决定 quota_i，之后循环使用
    """
    def __init__(
        self,
        collocation: Colloc,
        batch_size: int,
        *,
        shuffle: bool = True,
        strategy: str = "equal",
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        self.full   = collocation
        self.bs     = batch_size
        self.shuffle= shuffle
        self.key    = key
        self.strat  = strategy.lower()
        assert self.strat in ("equal", "global", "fixed")
        self._reset()

    # ─────────────────── public ──────────────────
    def next(self) -> Colloc:
        if not self.idxs:
            self._reset()
        return self._take()

    def resample(self, new_full: Colloc):
        self.full = new_full
        self._reset()

    # ─────────────────── helpers ─────────────────
    def _reset(self):
        """重建索引池；fixed/global 还要生成平铺索引或 quota 列表"""
        if isinstance(self.full, (list, tuple)):               # ---------- FBPINN ----------
            self.idxs = [jnp.arange(len(x)) for x in self.full]
            if self.shuffle:
                self.key, *sub = jax.random.split(self.key, len(self.full)+1)
                self.idxs = [jax.random.permutation(k, idx)
                             for k, idx in zip(sub, self.idxs)]

            # 平铺索引 (global 用)
            start, flats = 0, []
            for idx in self.idxs:
                flats.append(idx + start)
                start += len(idx)
            self.flat_map = jnp.concatenate(flats)

            # -------- fixed: 计算比例 quota_i --------
            if self.strat == "fixed":
                sizes   = jnp.array([len(x) for x in self.full])
                ratios  = sizes / sizes.sum()
                quota   = jnp.maximum(1, jnp.round(self.bs * ratios)).astype(int)

                # 校正配额总和，补偿到样本量最大的子域
                diff = int(quota.sum()) - self.bs
                if diff != 0:
                    big = int(jnp.argmax(sizes))
                    quota = quota.at[big].add(-diff)

                # 锁定每子域 quota_i 个索引
                fixed = []
                for i, q in enumerate(quota):
                    sel      = self.idxs[i][:q]
                    fixed.append(sel)                         # (q,)
                    self.idxs[i] = jnp.concatenate([          # 余量留作循环
                        self.idxs[i][q:], sel
                    ])
                self.fixed_sel = fixed
                self.quota     = quota

        else:                                                # ---------- PINN ----------
            self.idxs = jnp.arange(len(self.full))
            if self.shuffle:
                self.key, sub = jax.random.split(self.key)
                self.idxs = jax.random.permutation(sub, self.idxs)

    # ----------------------------------------------------------
    def _take(self) -> Colloc:
        # ========== FBPINN ==========
        if isinstance(self.full, (list, tuple)):
            n_sub = len(self.full)

            # ----- equal -----
            if self.strat == "equal":
                base  = self.bs // n_sub
                batch = []
                for i, (x_i, idx_i) in enumerate(zip(self.full, self.idxs)):
                    while len(idx_i) < base:
                        self._reset(); idx_i = self.idxs[i]
                    sel, self.idxs[i] = idx_i[:base], idx_i[base:]
                    batch.append(x_i[sel])
                return batch

            # ----- fixed -----
            if self.strat == "fixed":
                batch = []
                for i, (x_i, q) in enumerate(zip(self.full, self.quota)):
                    sel, self.fixed_sel[i] = (
                        self.fixed_sel[i][:q],
                        jnp.roll(self.fixed_sel[i], -q)
                    )
                    batch.append(x_i[sel])
                return batch

            # ----- global -----
            self.key, sub = jax.random.split(self.key)
            flat_sel = jax.random.choice(sub, self.flat_map,
                                         (self.bs,), replace=False)
            batch, start = [[] for _ in range(n_sub)], 0
            for i, x_i in enumerate(self.full):
                end  = start + len(x_i)
                mask = (flat_sel >= start) & (flat_sel < end)
                batch[i] = x_i[flat_sel[mask] - start]
                start = end
            return batch

        # ========== PINN ==========
        sel, self.idxs = self.idxs[:self.bs], self.idxs[self.bs:]
        return self.full[sel]
