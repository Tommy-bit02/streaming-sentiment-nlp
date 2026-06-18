"""
=============================================================================
TV Streaming Sentiment NLP — Visualizations
=============================================================================
⚠️  ACADEMIC RESEARCH / PORTFOLIO USE ONLY
Run AFTER 01_nlp_pipeline.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

C = {
    "bg":      "#080C12", "panel":   "#0E1420", "border":  "#1A2035",
    "text":    "#E2E8F0", "sub":     "#5A6A85",
    "netflix": "#E50914", "disney":  "#113CCF", "hbo":     "#8B2FC9",
    "amazon":  "#FF9900", "apple":   "#A8A9AD", "peacock": "#00A4E0",
    "green":   "#22C55E", "gold":    "#F5C518", "red":     "#EF4444",
}

PLATFORM_COLORS = {
    'Netflix':      C["netflix"],  'Disney+':     C["disney"],
    'HBO Max':      C["hbo"],      'Amazon Prime': C["amazon"],
    'Apple TV+':    C["apple"],    'Peacock':      C["peacock"],
}
DISCLAIMER = ("⚠️  Academic research & portfolio use only  ·  "
              "Synthetic data modeled from real industry events  ·  "
              "Not financial advice  ·  June 2026")

def style(fig, axes):
    fig.patch.set_facecolor(C["bg"])
    for ax in (axes if isinstance(axes, list) else [axes]):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["sub"], labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        ax.xaxis.label.set_color(C["sub"])
        ax.yaxis.label.set_color(C["sub"])

def footer(fig):
    fig.text(0.5, 0.005, DISCLAIMER, ha='center', fontsize=7,
             color=C["sub"], style='italic')


# =============================================================================
# CHART 1 — SENTIMENT DASHBOARD
# =============================================================================

def chart_sentiment_dashboard():
    df   = pd.read_csv("outputs/sentiment_processed.csv", parse_dates=['date'])
    evts = pd.read_csv("outputs/sentiment_shock_events.csv", parse_dates=['date'])

    fig = plt.figure(figsize=(18, 11), facecolor=C["bg"])
    fig.suptitle("TV Streaming Sentiment NLP Dashboard — 2020–2026",
                 fontsize=21, fontweight="bold", color=C["text"], y=0.97)
    fig.text(0.5, 0.935,
             "Net Sentiment Momentum (Sₜ) · Shock Events · Platform Comparison · "
             "Regime Distribution  |  FinBERT-calibrated · 9,918 daily records",
             ha='center', fontsize=10, color=C["sub"])

    gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35,
                  left=0.06, right=0.97, top=0.90, bottom=0.07)

    platforms = ['Netflix','Disney+','HBO Max','Amazon Prime','Apple TV+','Peacock']

    # ── Panel A: Netflix sentiment time series (top full row) ─────────────
    ax1 = fig.add_subplot(gs[0, :])
    style(fig, ax1)

    for platform in platforms:
        sub = df[df['platform'] == platform].sort_values('date')
        roll = sub.set_index('date')['net_sentiment_momentum'].rolling('30D').mean()
        ax1.plot(roll.index, roll.values,
                 color=PLATFORM_COLORS[platform], linewidth=1.8,
                 label=platform, alpha=0.85)

    # Mark shock events
    for _, ev in evts.iterrows():
        col = C["red"] if ev['sentiment_score'] < 0.2 else C["green"]
        ax1.axvline(ev['date'], color=col, linewidth=1, linestyle=':', alpha=0.6)
        ax1.text(ev['date'], ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 0.9,
                 ev['platform'][:3], rotation=90, fontsize=6.5,
                 color=col, va='top', ha='right')

    ax1.set_title("30-Day Rolling Net Sentiment Momentum (Sₜ) — All Platforms with Shock Events",
                  fontsize=13, fontweight='bold', color=C["text"], pad=8)
    ax1.set_ylabel("Sₜ Score", color=C["sub"])
    ax1.legend(fontsize=8.5, framealpha=0.2, labelcolor=C["text"],
               facecolor=C["panel"], edgecolor=C["border"],
               loc='lower left', ncol=3)
    ax1.axhline(0, color=C["sub"], linewidth=0.8, linestyle='--', alpha=0.4)

    # ── Panel B: Avg sentiment by platform (bar) ──────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style(fig, ax2)

    summ = df.groupby('platform')['net_sentiment_momentum'].mean().sort_values()
    colors2 = [PLATFORM_COLORS[p] for p in summ.index]
    hb = ax2.barh(summ.index, summ.values, color=colors2,
                  edgecolor=C["bg"], height=0.6)
    for bar, val in zip(hb, summ.values):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va='center', fontsize=9,
                 color=C["text"], fontweight='bold')
    ax2.set_title("Avg Sₜ Score\nby Platform", fontsize=11,
                  fontweight='bold', color=C["text"], pad=8)
    ax2.set_xlabel("Net Sentiment Momentum", color=C["sub"])
    ax2.tick_params(axis='y', labelsize=9, colors=C["text"])
    ax2.set_xlim(0, summ.max() * 1.22)

    # ── Panel C: Shock events table ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1:])
    style(fig, ax3)
    ax3.set_xticks([]); ax3.set_yticks([])

    evts_s = evts.sort_values('date')
    ax3.set_title("Key Sentiment Shock Events Detected by NLP Engine",
                  fontsize=11, fontweight='bold', color=C["text"], pad=8)
    y_pos = 0.92
    ax3.text(0.02, y_pos, f"{'Date':<13}{'Platform':<18}{'Sₜ':>7}  Event",
             transform=ax3.transAxes, fontsize=8.5,
             color=C["sub"], fontfamily='monospace')
    y_pos -= 0.06
    ax3.axhline(y_pos + 0.01, color=C["border"], linewidth=1)
    y_pos -= 0.02

    for _, row in evts_s.iterrows():
        icon  = "🔴" if row['sentiment_score'] < 0.2 else "🟢"
        col   = PLATFORM_COLORS.get(row['platform'], C["text"])
        txt   = (f"{str(row['date'].date()):<13}"
                 f"{row['platform']:<18}"
                 f"{row['net_sentiment_momentum']:>6.3f}  "
                 f"{icon} {row['event_note'][:45]}")
        ax3.text(0.02, y_pos, txt, transform=ax3.transAxes,
                 fontsize=8, color=col, fontfamily='monospace')
        y_pos -= 0.072

    # ── Panel D: Headline volume by platform (bottom full row) ────────────
    ax4 = fig.add_subplot(gs[2, :])
    style(fig, ax4)

    monthly = df.copy()
    monthly['month'] = monthly['date'].dt.to_period('M').astype(str)
    vol = monthly.groupby(['month','platform'])['n_headlines'].sum().reset_index()

    for platform in platforms:
        sub = vol[vol['platform'] == platform].sort_values('month')
        if len(sub) > 0:
            x = range(len(sub))
            roll_vol = pd.Series(sub['n_headlines'].values).rolling(3, min_periods=1).mean()
            ax4.plot(list(x), roll_vol.values,
                     color=PLATFORM_COLORS[platform], linewidth=1.8,
                     label=platform, alpha=0.8)

    ax4.set_title("Monthly Headline Volume by Platform (3-Month Rolling) — Media Coverage Intensity",
                  fontsize=12, fontweight='bold', color=C["text"], pad=8)
    ax4.set_ylabel("Headlines / Month", color=C["sub"])
    tick_positions = list(range(0, len(vol['month'].unique()), 6))
    tick_labels    = sorted(vol['month'].unique())[::6]
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels, rotation=30, ha='right',
                        fontsize=8, color=C["sub"])
    ax4.legend(fontsize=8.5, framealpha=0.2, labelcolor=C["text"],
               facecolor=C["panel"], edgecolor=C["border"], ncol=3)

    footer(fig)
    path = "outputs/chart1_sentiment_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=C["bg"])
    plt.close(fig)
    print(f"  ✅ Chart 1 saved → {path}")


# =============================================================================
# CHART 2 — NLP TOPIC MODEL & CONTENT SIMILARITY
# =============================================================================

def chart_nlp_analysis():
    df    = pd.read_csv("outputs/sentiment_processed.csv", parse_dates=['date'])
    sim   = pd.read_csv("outputs/cosine_similarity_matrix.csv", index_col=0)
    shows = pd.read_csv("outputs/show_similarity.csv")

    fig = plt.figure(figsize=(18, 10), facecolor=C["bg"])
    fig.suptitle("NLP Analysis — Topic Modeling, Content Similarity & Sentiment Regimes",
                 fontsize=19, fontweight="bold", color=C["text"], y=0.97)
    fig.text(0.5, 0.935,
             "LDA topic themes · TF-IDF cosine similarity · Sentiment regime breakdown by platform",
             ha='center', fontsize=10, color=C["sub"])

    gs = GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.35,
                  left=0.06, right=0.97, top=0.90, bottom=0.08)

    # ── A: Cosine similarity heatmap ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[:, :2])
    style(fig, ax1)

    im = ax1.imshow(sim.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(sim.columns)))
    ax1.set_yticks(range(len(sim.index)))
    ax1.set_xticklabels(sim.columns, rotation=45, ha='right',
                        fontsize=8.5, color=C["text"])
    ax1.set_yticklabels(sim.index, fontsize=8.5, color=C["text"])

    for i in range(len(sim.index)):
        for j in range(len(sim.columns)):
            val = sim.values[i, j]
            if i != j:
                ax1.text(j, i, f"{val:.2f}", ha='center', va='center',
                         fontsize=7.5, color=C["white"] if val > 0.5 else C["sub"])

    plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.02).set_label(
        'Cosine Similarity', color=C["sub"], fontsize=8)
    ax1.set_title("TF-IDF Content Similarity Matrix — Content Recommendation Engine\n"
                  "('If you liked X, you might enjoy Y' — based on plot synopsis semantics)",
                  fontsize=12, fontweight='bold', color=C["text"], pad=8)

    # ── B: Sentiment regime stacked bar ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style(fig, ax2)

    df['regime_simple'] = df['sentiment_regime'].astype(str)
    regime_colors = {
        'Strongly Bearish': C["red"],
        'Bearish':          '#F97316',
        'Neutral':          C["sub"],
        'Bullish':          '#84CC16',
        'Strongly Bullish': C["green"],
    }
    regimes = ['Strongly Bearish','Bearish','Neutral','Bullish','Strongly Bullish']
    platforms_list = list(PLATFORM_COLORS.keys())
    x = np.arange(len(platforms_list))
    bottom = np.zeros(len(platforms_list))

    for regime in regimes:
        vals = []
        for plat in platforms_list:
            sub  = df[df['platform'] == plat]
            tot  = len(sub)
            cnt  = (sub['regime_simple'] == regime).sum()
            vals.append(cnt / tot * 100 if tot > 0 else 0)
        ax2.bar(x, vals, bottom=bottom, color=regime_colors[regime],
                label=regime, edgecolor=C["bg"], width=0.7)
        bottom += np.array(vals)

    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace(' ','\n').replace('+','+ ') for p in platforms_list],
                        fontsize=7.5, color=C["text"])
    ax2.set_ylabel("% of Trading Days", color=C["sub"])
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.set_title("Sentiment Regime\nDistribution", fontsize=11,
                  fontweight='bold', color=C["text"], pad=8)
    ax2.legend(fontsize=7, framealpha=0.2, labelcolor=C["text"],
               facecolor=C["panel"], edgecolor=C["border"],
               loc='upper right')

    # ── C: Topic theme word bubble ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor(C["panel"])
    for sp in ax3.spines.values():
        sp.set_edgecolor(C["border"])
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title("LDA Topic Themes Discovered\n(TF-IDF keyword clusters)",
                  fontsize=11, fontweight='bold', color=C["text"], pad=8)

    topics = [
        ('Growth & Earnings',     'subscribers  revenue\nearnings  quarterly  beat', C["green"]),
        ('Content Performance',   'original  hit  award\nviewership  streaming', C["gold"]),
        ('Market Competition',    'competition  market\nchurn  cancellation', C["red"]),
        ('Pricing & Strategy',    'price  bundle  tier\nad-supported  hike', '#F97316'),
        ('Tech & Infrastructure', 'platform  streaming\nquality  technology', C["peacock"]),
    ]
    y_step = 0.17
    for i, (topic, words, color) in enumerate(topics):
        y = 0.88 - i * y_step
        ax3.text(0.04, y, f"● {topic}", transform=ax3.transAxes,
                 fontsize=9, color=color, fontweight='bold')
        ax3.text(0.08, y - 0.055, words, transform=ax3.transAxes,
                 fontsize=7.5, color=C["sub"])

    footer(fig)
    path = "outputs/chart2_nlp_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=C["bg"])
    plt.close(fig)
    print(f"  ✅ Chart 2 saved → {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  STREAMING SENTIMENT — GENERATING CHARTS")
    print("=" * 60 + "\n")
    chart_sentiment_dashboard()
    chart_nlp_analysis()
    print("\n  All charts saved to /outputs/\n")
