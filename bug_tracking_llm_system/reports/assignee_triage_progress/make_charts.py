from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "reports" / "assignee_triage_progress" / "assets"
BASE_METRICS = ROOT / "assignee_triage_accuracy" / "paper_grade" / "reports" / "bmo_public_10k" / "bmo_public_10k_metrics.json"
PHASE7_REPORT = ROOT / "assignee_triage_accuracy" / "phase7_rolling_open_set" / "reports" / "bmo_public_2021q4_conservative_untouched_holdout" / "rolling_holdout_report.json"
PHASE7_PRED = ROOT / "assignee_triage_accuracy" / "phase7_rolling_open_set" / "reports" / "bmo_public_2021q4_conservative_untouched_holdout" / "rolling_holdout_predictions.jsonl"
FONT_PATH = Path("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc")


def fp(size: int = 10):
    return FontProperties(fname=str(FONT_PATH), size=size) if FONT_PATH.exists() else None


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def baseline_chart(metrics, path):
    baselines = metrics["baselines"]
    keys = ["global_majority", "component_majority", "bm25_text_knn", "current_assignee_triager"]
    labels = ["全域多數", "元件多數", "BM25 文字", "Hybrid"]
    top1 = [baselines[k]["top1_accuracy"] for k in keys]
    hit3 = [baselines[k]["hit_at_3"] for k in keys]
    mrr = [baselines[k]["mrr"] for k in keys]
    x = np.arange(len(keys)); width = 0.23
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=180)
    ax.bar(x-width, top1, width, label="Top-1", color="#2E74B5")
    ax.bar(x, hit3, width, label="Hit@3", color="#2A7F9E")
    ax.bar(x+width, mrr, width, label="MRR", color="#C69428")
    ax.set_ylim(0, 1); ax.set_ylabel("比率", fontproperties=fp())
    ax.set_xticks(x, labels, fontproperties=fp()); ax.grid(axis="y", alpha=0.2)
    ax.legend(prop=fp(9), frameon=False, ncol=3, loc="upper left")
    for pos, values in ((x-width, top1), (x, hit3), (x+width, mrr)):
        for xi, value in zip(pos, values):
            ax.text(xi, value+0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def routing_chart(report, path):
    r = report["routing_metrics"]
    labels = ["自動指派", "Top-3 確認", "人工分流"]
    values = [r["auto_assignment_coverage"], r["top3_confirmation_rate"], r["manual_triage_rate"]]
    accuracy = [r["auto_assignment_accuracy"], r["top3_confirmation_accuracy"], None]
    fig, ax = plt.subplots(figsize=(8.8, 3.9), dpi=180)
    bars = ax.barh(labels, values, color=["#2F7D5A", "#2E74B5", "#98A2B3"], height=0.55)
    ax.set_xlim(0, 0.75); ax.set_xlabel("全部 Ticket 中的比例", fontproperties=fp())
    ax.set_yticks(range(len(labels)), labels, fontproperties=fp()); ax.grid(axis="x", alpha=0.2)
    for bar, value, acc in zip(bars, values, accuracy):
        suffix = f"；正確率 {acc*100:.2f}%" if acc is not None else ""
        ax.text(value+0.012, bar.get_y()+bar.get_height()/2, f"{value*100:.2f}%{suffix}", va="center", fontproperties=fp(9))
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def confusion_chart(rows, path):
    counts = Counter(row["expected_assignee"] for row in rows if str(row["expected_assignee"]).startswith("dev_"))
    labels = [name for name, _ in counts.most_common(7)]
    matrix = np.zeros((len(labels), len(labels)+1), dtype=int)
    for row in rows:
        expected = row["expected_assignee"]
        if expected not in labels:
            continue
        pred = row["predicted_assignee"]
        matrix[labels.index(expected), labels.index(pred) if pred in labels else len(labels)] += 1
    fig, ax = plt.subplots(figsize=(8.7, 5.8), dpi=180)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)+1), labels+["其他"], rotation=35, ha="right", fontproperties=fp(8))
    ax.set_yticks(range(len(labels)), labels, fontproperties=fp(8))
    ax.set_xlabel("模型預測負責人", fontproperties=fp()); ax.set_ylabel("真實負責人", fontproperties=fp())
    threshold = matrix.max()*0.55
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j]>threshold else "#183B56", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def flowchart(path):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    def box(x, y, w, h, text, color="#E8EEF5", edge="#2E74B5", size=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, linewidth=1.4, joinstyle="round"))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontproperties=fp(size), wrap=True, color="#183B56")
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#667085", lw=1.5))
    box(0.45,8.45,2.1,.9,"Ticket 輸入\n標題／描述／元件／產品","#F4F7FA")
    box(3.05,8.45,2,.9,"清理與時間過濾\n排除未來標籤","#F4F7FA")
    box(5.55,8.45,1.9,.9,"歷史責任輪廓\n頻率／近期活動","#EAF5EF","#2F7D5A")
    box(7.95,8.45,1.6,.9,"候選池\nTop-30","#EAF5EF","#2F7D5A")
    arrow(2.55,8.9,3.05,8.9); arrow(5.05,8.9,5.55,8.9); arrow(7.45,8.9,7.95,8.9)
    for x, text in ((.6,"產品＋元件\n歷史比例"),(2.55,"BM25\n文字相似"),(4.25,"SBERT\n語意相似"),(5.95,"30／90 日\n活動與衰減"),(7.75,"來源一致性\n與全域先驗")):
        box(x,6.5,1.7 if x==.6 else 1.55,.85,text)
    for x in (1.45,3.275,4.975,6.725,8.575): arrow(x,6.5,8.0,5.65)
    box(6.2,4.55,3.15,1.05,"候選人學習排序（LTR）\n淺層 HistGradientBoosting\n輸出 Top-k 與排序機率","#FFF5D9","#A97516")
    arrow(7.8,4.55,7.0,3.8)
    box(5.05,2.8,2.0,.95,"Logistic 校準\nTop-1 正確機率","#FCECEC","#A33A3A")
    box(7.55,2.8,2.0,.95,"開放集合偵測\n未知負責人風險","#FCECEC","#A33A3A")
    arrow(7.05,3.28,7.55,3.28)
    box(3.5,1.2,1.9,.9,"高信心＋低風險\n自動指派","#EAF5EF","#2F7D5A")
    box(5.75,1.2,1.9,.9,"中信心\nTop-3 人工確認","#E8EEF5","#2E74B5")
    box(8.0,1.2,1.55,.9,"低信心／漂移\n人工分流","#F2F4F7","#667085")
    for x in (4.45,6.7,8.78): arrow(7.7,2.8,x,2.1)
    ax.text(.45,.35,"註：Phase 7 為離線研究流程；目前整合執行時預設仍採 Hybrid fail-closed 路由。",fontproperties=fp(9),color="#667085")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_json(BASE_METRICS); report = load_json(PHASE7_REPORT); rows = load_jsonl(PHASE7_PRED)
    baseline_chart(metrics, ASSET_DIR/"baseline_metrics.png")
    routing_chart(report, ASSET_DIR/"routing_outcomes.png")
    confusion_chart(rows, ASSET_DIR/"confusion_top7.png")
    flowchart(ASSET_DIR/"assignee_flow.png")


if __name__ == "__main__":
    main()
