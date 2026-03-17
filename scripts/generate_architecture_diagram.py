"""Generate the RetentionIQ architecture diagram as a high-quality PNG."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_box(ax, x, y, w, h, label, sublabel=None, color="#E8F4FD",
             edge_color="#2196F3", fontsize=10, bold=True, sublabel_size=8,
             icon=None, alpha=1.0, text_color="#1a1a1a"):
    """Draw a rounded box with label."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor=edge_color, linewidth=1.8, alpha=alpha
    )
    ax.add_patch(box)

    weight = "bold" if bold else "normal"
    text_y = y + h / 2 + (0.15 if sublabel else 0)
    if icon:
        ax.text(x + w / 2, text_y + 0.15, icon, ha="center", va="center",
                fontsize=fontsize + 6, fontfamily="Segoe UI Emoji")
        text_y -= 0.15

    ax.text(x + w / 2, text_y, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color,
            fontfamily="Segoe UI")

    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.25, sublabel, ha="center",
                va="center", fontsize=sublabel_size, color="#555",
                fontfamily="Segoe UI", style="italic")


def draw_arrow(ax, x1, y1, x2, y2, label=None, color="#666",
               style="-|>", lw=1.5, curved=False):
    """Draw an arrow with optional label."""
    if curved:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, color=color, linewidth=lw,
            connectionstyle="arc3,rad=0.2",
            mutation_scale=15
        )
    else:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, color=color, linewidth=lw,
            mutation_scale=15
        )
    ax.add_patch(arrow)

    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.12
        ax.text(mx, my, label, ha="center", va="center", fontsize=7,
                color="#888", fontfamily="Segoe UI",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.9))


def draw_section(ax, x, y, w, h, title, color="#f0f0f0",
                 edge_color="#ccc"):
    """Draw a section boundary with title."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2",
        facecolor=color, edgecolor=edge_color,
        linewidth=1.2, linestyle="--", alpha=0.5
    )
    ax.add_patch(box)
    ax.text(x + 0.3, y + h - 0.15, title, ha="left", va="top",
            fontsize=9, fontweight="bold", color="#777",
            fontfamily="Segoe UI")


def main():
    fig, ax = plt.subplots(1, 1, figsize=(22, 14))
    ax.set_xlim(-0.5, 21.5)
    ax.set_ylim(-0.5, 13.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ===== Title =====
    ax.text(10.5, 13.1, "RetentionIQ — System Architecture",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color="#1a1a1a", fontfamily="Segoe UI")
    ax.text(10.5, 12.7,
            "Prescriptive Retention System  |  ML + Causal Inference + "
            "Stochastic Optimization + LLM Agents",
            ha="center", va="center", fontsize=10, color="#777",
            fontfamily="Segoe UI")

    # ================================================================
    # SECTION 1: DATA PLATFORM (top row)
    # ================================================================
    draw_section(ax, 0, 9.2, 21, 3.2, "Data Platform — Dagster Orchestration",
                 color="#E3F2FD", edge_color="#90CAF9")

    # Data Sources
    draw_box(ax, 0.5, 10, 2.2, 1.6,
             "Data Sources", "CSV / Parquet / API",
             color="#FFF3E0", edge_color="#FF9800", icon=None)
    # Icon placeholder (text-based, no emoji)
    ax.text(1.6, 11.8, "◉", ha="center", fontsize=14, color="#FF9800")

    # DVC
    draw_box(ax, 0.5, 9.5, 2.2, 0.5,
             "DVC Versioning", color="#FFF8E1", edge_color="#FFC107",
             fontsize=8, bold=False)

    # Bronze
    draw_box(ax, 3.5, 10, 2.5, 1.6,
             "Bronze Layer", "Schema validation\nAppend-only",
             color="#FFEBEE", edge_color="#EF5350")

    # Silver
    draw_box(ax, 6.8, 10, 2.5, 1.6,
             "Silver Layer", "Dedup, cleaning\nBusiness rules",
             color="#FFF3E0", edge_color="#FF9800")

    # Gold
    draw_box(ax, 10.1, 10, 2.5, 1.6,
             "Gold Layer", "Member 360°\nLocation aggregates",
             color="#E8F5E9", edge_color="#66BB6A")

    # Feast Feature Store
    draw_box(ax, 13.4, 10, 2.8, 1.6,
             "Feast Feature Store", "Point-in-time joins\nOnline + Offline",
             color="#E8EAF6", edge_color="#5C6BC0")

    # Great Expectations
    draw_box(ax, 17, 10.4, 3.3, 1.0,
             "Great Expectations", "Quality gates at each layer",
             color="#FCE4EC", edge_color="#EC407A", fontsize=9)

    # Evidently Monitoring
    draw_box(ax, 17, 9.3, 3.3, 1.0,
             "Evidently Monitoring", "Data drift + model perf",
             color="#F3E5F5", edge_color="#AB47BC", fontsize=9)

    # Arrows: Data Pipeline
    draw_arrow(ax, 2.7, 10.8, 3.5, 10.8, color="#FF9800")
    draw_arrow(ax, 6.0, 10.8, 6.8, 10.8, color="#FF9800")
    draw_arrow(ax, 9.3, 10.8, 10.1, 10.8, color="#66BB6A")
    draw_arrow(ax, 12.6, 10.8, 13.4, 10.8, label="features",
               color="#5C6BC0")

    # Quality arrows
    draw_arrow(ax, 16.3, 10.8, 17.0, 10.8, label="validates",
               color="#EC407A", style="-|>", lw=1.0)

    # ================================================================
    # SECTION 2: MODEL LAYER (middle-left)
    # ================================================================
    draw_section(ax, 0, 5.3, 7.5, 3.5, "Model Layer — MLflow Tracked",
                 color="#E8F5E9", edge_color="#A5D6A7")

    # Survival
    draw_box(ax, 0.5, 6.4, 2.0, 1.5,
             "Survival", "Cox PH / KM / AFT\nlifelines",
             color="#E8F5E9", edge_color="#43A047", fontsize=9)

    # Churn
    draw_box(ax, 2.8, 6.4, 2.0, 1.5,
             "Churn Scoring", "XGBoost\nTemporal CV",
             color="#E8F5E9", edge_color="#43A047", fontsize=9)

    # LTV
    draw_box(ax, 5.1, 6.4, 2.0, 1.5,
             "LTV", "Survival × Revenue\nNPV discounted",
             color="#E8F5E9", edge_color="#43A047", fontsize=9)

    # MLflow
    draw_box(ax, 1.5, 5.5, 4.5, 0.7,
             "MLflow — Experiment Tracking + Model Registry",
             color="#FFF8E1", edge_color="#FFA000", fontsize=8, bold=True)

    # Arrow: Features → Models
    draw_arrow(ax, 14.8, 10.0, 14.8, 9.0, color="#5C6BC0", lw=1.2)
    draw_arrow(ax, 14.8, 9.0, 3.8, 8.2, label="features",
               color="#5C6BC0", lw=1.2, curved=True)

    # ================================================================
    # SECTION 3: CAUSAL ENGINE (middle-center)
    # ================================================================
    draw_section(ax, 8, 5.3, 6.5, 3.5,
                 "Decision Engine — Causal Inference",
                 color="#FFF3E0", edge_color="#FFCC80")

    # DoWhy
    draw_box(ax, 8.5, 6.4, 2.5, 1.5,
             "DoWhy", "Causal DAG\nBackdoor identification",
             color="#FFF3E0", edge_color="#E65100", fontsize=9)

    # EconML
    draw_box(ax, 11.3, 6.4, 2.8, 1.5,
             "EconML", "CausalForestDML\nHeterogeneous effects",
             color="#FFF3E0", edge_color="#E65100", fontsize=9)

    # Refutation
    draw_box(ax, 9.5, 5.5, 3.5, 0.7,
             "Refutation Tests — Placebo / Random Cause / Subset",
             color="#FBE9E7", edge_color="#BF360C", fontsize=7.5)

    # Arrow: DoWhy → EconML
    draw_arrow(ax, 11.0, 7.15, 11.3, 7.15, label="estimand",
               color="#E65100")

    # Arrow: Models → Causal
    draw_arrow(ax, 7.1, 7.15, 8.5, 7.15, label="scores",
               color="#43A047")

    # ================================================================
    # SECTION 4: OPTIMIZATION (middle-right)
    # ================================================================
    draw_section(ax, 15, 5.3, 5.5, 3.5,
                 "Optimization — Pyomo",
                 color="#E8EAF6", edge_color="#9FA8DA")

    # Deterministic
    draw_box(ax, 15.5, 7.0, 2.2, 1.2,
             "Deterministic", "MILP Baseline",
             color="#E8EAF6", edge_color="#3F51B5", fontsize=9)

    # Stochastic
    draw_box(ax, 18.0, 7.0, 2.2, 1.2,
             "Stochastic", "Two-stage\nRobust allocation",
             color="#C5CAE9", edge_color="#283593", fontsize=9)

    # Constraints
    draw_box(ax, 15.5, 5.6, 4.7, 1.0,
             "Budget R$50K | 250 locations | Staff hours | "
             "1 action/member",
             color="#F5F5F5", edge_color="#9E9E9E", fontsize=7, bold=False)

    # Arrow: CATE → Optimizer
    draw_arrow(ax, 14.1, 7.15, 15.5, 7.15, label="CATE ± CI",
               color="#E65100")

    # Arrow: Deterministic → Stochastic
    draw_arrow(ax, 17.7, 7.6, 18.0, 7.6, color="#3F51B5", lw=1.0)

    # ================================================================
    # SECTION 5: AGENT LAYER (bottom)
    # ================================================================
    draw_section(ax, 0, 0.5, 14.5, 4.3,
                 "Agent Layer — LangGraph + pgvector",
                 color="#F3E5F5", edge_color="#CE93D8")

    # Router
    draw_box(ax, 0.5, 2.5, 2.2, 1.4,
             "Router", "Query classification\nfactual/diagnostic/\nprescriptive",
             color="#F3E5F5", edge_color="#8E24AA", fontsize=8,
             sublabel_size=7)

    # Analyst
    draw_box(ax, 3.3, 2.5, 2.5, 1.4,
             "Analyst Agent", "SQL queries\nSurvival analysis\nCohort comparison",
             color="#EDE7F6", edge_color="#7B1FA2", fontsize=8,
             sublabel_size=7)

    # Strategist
    draw_box(ax, 6.4, 2.5, 2.5, 1.4,
             "Strategist Agent", "CATE estimation\nBudget optimizer\nIntervention sim",
             color="#EDE7F6", edge_color="#7B1FA2", fontsize=8,
             sublabel_size=7)

    # Writer
    draw_box(ax, 9.5, 2.5, 2.5, 1.4,
             "Writer Agent", "Manager-friendly\nlanguage\nNo jargon",
             color="#EDE7F6", edge_color="#7B1FA2", fontsize=8,
             sublabel_size=7)

    # Memory
    draw_box(ax, 12.5, 2.5, 1.7, 1.4,
             "pgvector", "Conversation\nEpisodic\nSemantic",
             color="#E1F5FE", edge_color="#0288D1", fontsize=8,
             sublabel_size=7)

    # Guardrails
    draw_box(ax, 0.5, 0.8, 6.5, 1.2,
             "Guardrails — PII Masking (CPF/email/phone) | SQL Injection "
             "Block | Confidence Threshold | LGPD Audit",
             color="#FFEBEE", edge_color="#C62828", fontsize=7.5, bold=True)

    # Agent arrows
    draw_arrow(ax, 2.7, 3.2, 3.3, 3.2, color="#8E24AA")
    draw_arrow(ax, 5.8, 3.2, 6.4, 3.2, color="#7B1FA2")
    draw_arrow(ax, 8.9, 3.2, 9.5, 3.2, color="#7B1FA2")

    # ================================================================
    # SECTION 6: SERVING (bottom-right)
    # ================================================================
    draw_section(ax, 15, 0.5, 5.5, 4.3,
                 "Serving & CI/CD",
                 color="#E0F2F1", edge_color="#80CBC4")

    # FastAPI
    draw_box(ax, 15.5, 2.8, 2.2, 1.5,
             "FastAPI", "/predict/churn\n/optimize\n/agent/query",
             color="#E0F2F1", edge_color="#00897B", fontsize=9)

    # Docker
    draw_box(ax, 18.0, 2.8, 2.2, 1.5,
             "Docker", "PostgreSQL 16\npgvector\nMLflow",
             color="#E0F2F1", edge_color="#00897B", fontsize=9)

    # GitHub Actions
    draw_box(ax, 15.5, 0.8, 4.7, 1.2,
             "GitHub Actions CI/CD — Lint (ruff) | Unit Tests (pytest) | "
             "Integration Tests | Data Quality",
             color="#F1F8E9", edge_color="#689F38", fontsize=7.5, bold=True)

    # Arrow: Agent → FastAPI
    draw_arrow(ax, 14.2, 3.2, 15.5, 3.5, color="#00897B", lw=1.5)

    # Arrow: Optimizer → Agent (CATE results)
    draw_arrow(ax, 16.5, 5.3, 7.6, 3.9, label="allocation results",
               color="#3F51B5", lw=1.2, curved=True)

    # ================================================================
    # VERTICAL FLOW ARROWS (connecting layers)
    # ================================================================

    # Features → Models (already drawn)

    # Gold → Models
    draw_arrow(ax, 3.8, 10.0, 3.8, 7.9, label="gold tables",
               color="#66BB6A", lw=1.5)

    # Models → Agents (scores)
    draw_arrow(ax, 3.8, 6.4, 3.8, 3.9, label="churn scores\nsurvival curves",
               color="#43A047", lw=1.2)

    # Causal → Agents (CATE)
    draw_arrow(ax, 11.0, 6.4, 7.6, 3.9, label="CATE estimates",
               color="#E65100", lw=1.2, curved=True)

    # Monitoring arrow
    draw_arrow(ax, 18.6, 9.3, 18.6, 8.8,
               label="monitors", color="#AB47BC", lw=1.0)
    ax.text(18.6, 8.6, "|", ha="center", va="center",
            fontsize=10, color="#AB47BC", fontweight="bold")

    # ================================================================
    # USER PERSONAS (far right)
    # ================================================================
    # User personas (text icons)
    draw_box(ax, 20.5, 3.0, 1.0, 1.0,
             "●", color="#E0F2F1", edge_color="#00897B", fontsize=16)
    ax.text(21.0, 2.8, "Location\nManager", ha="center",
            fontsize=7, color="#555", fontfamily="Segoe UI")

    draw_box(ax, 20.5, 1.3, 1.0, 1.0,
             "●●", color="#E0F2F1", edge_color="#00897B", fontsize=12)
    ax.text(21.0, 1.1, "Regional\nDirector", ha="center",
            fontsize=7, color="#555", fontfamily="Segoe UI")

    draw_arrow(ax, 20.2, 3.55, 20.5, 3.5, color="#00897B", lw=1.5)
    draw_arrow(ax, 20.2, 2.5, 20.5, 1.9, color="#00897B", lw=1.0)

    # ================================================================
    # LEGEND
    # ================================================================
    legend_y = -0.2
    legend_items = [
        ("#FFEBEE", "#EF5350", "Data Ingestion"),
        ("#E8F5E9", "#43A047", "ML Models"),
        ("#FFF3E0", "#E65100", "Causal Inference"),
        ("#E8EAF6", "#3F51B5", "Optimization"),
        ("#F3E5F5", "#8E24AA", "Agent Layer"),
        ("#E0F2F1", "#00897B", "Infrastructure"),
    ]
    for i, (fc, ec, label) in enumerate(legend_items):
        lx = 2 + i * 3.2
        box = FancyBboxPatch(
            (lx, legend_y), 0.4, 0.3,
            boxstyle="round,pad=0.05",
            facecolor=fc, edgecolor=ec, linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(lx + 0.6, legend_y + 0.15, label, ha="left", va="center",
                fontsize=8, color="#555", fontfamily="Segoe UI")

    # Save
    plt.tight_layout()
    fig.savefig(
        "docs/images/architecture.png",
        dpi=200, bbox_inches="tight",
        facecolor="white", edgecolor="none",
        pad_inches=0.3
    )
    plt.close()
    print("Saved: docs/images/architecture.png")


if __name__ == "__main__":
    main()
