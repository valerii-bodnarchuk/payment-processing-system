"""
Phase 2a similarity analysis.

Fetches all synthetic Case rows (source='synthetic') from the DB, computes
pairwise cosine similarities using their stored pgvector embeddings, then
reports:
  1. Cluster breakdown (total rows by source and cluster)
  2. Mean intra-cluster similarity (post Phase 2a)
  3. Mean cross-cluster similarity (post Phase 2a)
  4. Top-20 most similar pairs (with cross-cluster flag)
  5. Bottom-10 pairs (to check SYN-009 isolation)
  6. For each boundary case SYN-021..SYN-040: top-3 most similar cases

Cluster assignments:
    SYN-001..004  velocity_abuse       (1)
    SYN-005..008  geo_mismatch         (2)
    SYN-009..012  account_takeover     (3)
    SYN-013..016  merchant_collusion   (4)
    SYN-017..020  legit_high_risk      (5)
    SYN-021..025  boundary_A (3×5)     (A)
    SYN-026..030  boundary_B (1×4)     (B)
    SYN-031..035  boundary_C (2×5)     (C)
    SYN-036..040  boundary_D (4)       (D)
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from env_bootstrap import load_env

# Must run before the os.getenv below, or .env values lose to the default.
load_env()

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/payment_system",
)


def _cluster(case_id: str) -> str:
    try:
        n = int(case_id.split("-")[1])
    except (IndexError, ValueError):
        return "unknown"
    if 1 <= n <= 4:
        return "1_velocity_abuse"
    if 5 <= n <= 8:
        return "2_geo_mismatch"
    if 9 <= n <= 12:
        return "3_account_takeover"
    if 13 <= n <= 16:
        return "4_merchant_collusion"
    if 17 <= n <= 20:
        return "5_legit_high_risk"
    if 21 <= n <= 25:
        return "A_ato_high_value"
    if 26 <= n <= 30:
        return "B_velocity_collusion"
    if 31 <= n <= 35:
        return "C_geo_vip_travel"
    if 36 <= n <= 40:
        return "D_merchant_friendly"
    return "unknown"


def _base_cluster(cluster: str) -> str:
    """Map boundary cluster back to its primary cluster for intra/cross calc."""
    return {
        "A_ato_high_value": "3_account_takeover",
        "B_velocity_collusion": "1_velocity_abuse",
        "C_geo_vip_travel": "5_legit_high_risk",
        "D_merchant_friendly": "4_merchant_collusion",
    }.get(cluster, cluster)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # vectors are already L2-normalised by bge


async def fetch_cases(conn) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT "caseId", "source", "verdict", "riskLevel",
               embedding::text AS emb_text
        FROM "Case"
        WHERE "source" IN ('synthetic', 'seed')
          AND embedding IS NOT NULL
        ORDER BY "caseId"
        """
    )
    cases = []
    for r in rows:
        emb_raw = r["emb_text"]
        vec = np.array([float(x) for x in emb_raw.strip("[]").split(",")], dtype=np.float32)
        cases.append({
            "case_id": r["caseId"],
            "source": r["source"],
            "verdict": r["verdict"],
            "cluster": _cluster(r["caseId"]),
            "vec": vec,
        })
    return cases


async def main() -> None:
    conn = await asyncpg.connect(DB_URL)
    try:
        await register_vector(conn)

        # ── 1. Row counts ────────────────────────────────────────────────
        total_rows = await conn.fetchval('SELECT COUNT(*) FROM "Case"')
        source_rows = await conn.fetch(
            'SELECT source, COUNT(*) AS n FROM "Case" GROUP BY source ORDER BY source'
        )
        print("\n=== 1. Case table counts ===")
        print(f"Total rows: {total_rows}")
        for r in source_rows:
            print(f"  source={r['source']}: {r['n']}")

        # Synthetic breakdown by cluster
        cases = await fetch_cases(conn)
        syn_cases = [c for c in cases if c["source"] == "synthetic"]
        from collections import Counter, defaultdict
        cluster_counts = Counter(c["cluster"] for c in syn_cases)
        print("\nSynthetic cases by cluster:")
        for cl, n in sorted(cluster_counts.items()):
            print(f"  {cl}: {n}")

        if len(syn_cases) < 2:
            print("Not enough synthetic cases with embeddings to compute similarity.")
            return

        # ── 2. Pairwise similarity matrix (synthetic only) ───────────────
        n = len(syn_cases)
        vecs = np.stack([c["vec"] for c in syn_cases])
        sim_matrix = vecs @ vecs.T  # cosine sim (normalised vectors)

        # Build all (i,j) pairs
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = syn_cases[i], syn_cases[j]
                same_base = _base_cluster(ci["cluster"]) == _base_cluster(cj["cluster"])
                cross = ci["cluster"] != cj["cluster"]
                pairs.append({
                    "i": i, "j": j,
                    "id_i": ci["case_id"], "id_j": cj["case_id"],
                    "cl_i": ci["cluster"], "cl_j": cj["cluster"],
                    "same_base": same_base,
                    "cross": cross,
                    "sim": float(sim_matrix[i, j]),
                })

        intra = [p["sim"] for p in pairs if not p["cross"]]
        cross = [p["sim"] for p in pairs if p["cross"]]

        print("\n=== 2. Similarity metrics ===")
        print(f"Mean intra-cluster similarity: {np.mean(intra):.4f}  (n={len(intra)} pairs)")
        print(f"Mean cross-cluster similarity: {np.mean(cross):.4f}  (n={len(cross)} pairs)")
        print(f"Median intra: {np.median(intra):.4f}  Median cross: {np.median(cross):.4f}")

        # ── 3. Top-20 most similar pairs ────────────────────────────────
        top20 = sorted(pairs, key=lambda p: -p["sim"])[:20]
        n_cross_in_top20 = sum(1 for p in top20 if p["cross"])
        print(f"\n=== 3. Top-20 most similar pairs ({n_cross_in_top20}/20 are cross-cluster) ===")
        print(f"{'Rank':<5} {'Case A':<12} {'Case B':<12} {'Sim':>6} {'Cross?':<8} {'Clusters'}")
        print("-" * 72)
        for rank, p in enumerate(top20, 1):
            cross_flag = "CROSS" if p["cross"] else "intra"
            print(
                f"{rank:<5} {p['id_i']:<12} {p['id_j']:<12} {p['sim']:>6.4f} "
                f"{cross_flag:<8} {p['cl_i']} × {p['cl_j']}"
            )

        # ── 4. Bottom-10 pairs (isolation check for SYN-009) ────────────
        bottom10 = sorted(pairs, key=lambda p: p["sim"])[:10]
        print("\n=== 4. Bottom-10 least similar pairs (SYN-009 isolation check) ===")
        print(f"{'Rank':<5} {'Case A':<12} {'Case B':<12} {'Sim':>6} {'Clusters'}")
        print("-" * 60)
        for rank, p in enumerate(bottom10, 1):
            print(
                f"{rank:<5} {p['id_i']:<12} {p['id_j']:<12} {p['sim']:>6.4f} "
                f"{p['cl_i']} × {p['cl_j']}"
            )
        syn009_in_bottom = sum(
            1 for p in bottom10 if "SYN-009" in (p["id_i"], p["id_j"])
        )
        print(f"\nSYN-009 appearances in bottom-10: {syn009_in_bottom}")

        # ── 5. Top-3 most similar for each boundary case ────────────────
        boundary_ids = {c["case_id"] for c in syn_cases if c["cluster"].startswith(("A_", "B_", "C_", "D_"))}
        all_syn_ids = {c["case_id"] for c in syn_cases}

        print("\n=== 5. Top-3 most similar for each boundary case (SYN-021..040) ===")
        print(f"{'Boundary':>12} | {'#1':>10} ({'>':>6}) | {'#2':>10} ({'>':>6}) | {'#3':>10} ({'>':>6}) | Cross-cluster?")
        print("-" * 100)

        idx_map = {c["case_id"]: i for i, c in enumerate(syn_cases)}
        for c in syn_cases:
            if c["case_id"] not in boundary_ids:
                continue
            i = idx_map[c["case_id"]]
            sims = [(syn_cases[j]["case_id"], float(sim_matrix[i, j])) for j in range(n) if j != i]
            sims.sort(key=lambda x: -x[1])
            top3 = sims[:3]
            cross_flags = []
            for cid, _ in top3:
                other = next(x for x in syn_cases if x["case_id"] == cid)
                cross_flags.append("X" if c["cluster"] != other["cluster"] else "·")
            row = f"{c['case_id']:>12} |"
            for (cid, s), flag in zip(top3, cross_flags):
                row += f" {cid:>10} ({s:.4f}) {flag} |"
            print(row)

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
