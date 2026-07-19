from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "assignee_triage_progress"
ASSET_DIR = OUT_DIR / "assets"
OUTPUT_DOCX = OUT_DIR / "自動分流給負責人功能－技術流程與測試進度報告.docx"

DATA_DIR = ROOT / "assignee_triage_accuracy" / "paper_grade" / "data"
REPORT_DIR = ROOT / "assignee_triage_accuracy" / "paper_grade" / "reports"
PHASE6_DIR = ROOT / "assignee_triage_accuracy" / "phase6_candidate_ltr" / "reports"
PHASE7_DIR = ROOT / "assignee_triage_accuracy" / "phase7_rolling_open_set" / "reports"

SUMMARY_PATH = DATA_DIR / "processed" / "bmo_public_10k_dataset_summary.json"
DESC_SUMMARY_PATH = DATA_DIR / "processed" / "bmo_public_10k_desc_dataset_summary.json"
BASE_METRICS_PATH = REPORT_DIR / "bmo_public_10k" / "bmo_public_10k_metrics.json"
PHASE6_REPORT_PATH = PHASE6_DIR / "bmo_public_q4_holdout_source_quota_sbert" / "candidate_ltr_report.json"
PHASE6_CAL_PATH = (
    PHASE6_DIR
    / "bmo_public_q4_holdout_source_quota_sbert"
    / "calibration"
    / "candidate_ltr_calibration_report.json"
)
PHASE7_REPORT_PATH = (
    PHASE7_DIR
    / "bmo_public_2021q4_conservative_untouched_holdout"
    / "rolling_holdout_report.json"
)
PHASE7_PRED_PATH = (
    PHASE7_DIR
    / "bmo_public_2021q4_conservative_untouched_holdout"
    / "rolling_holdout_predictions.jsonl"
)
PHASE7_DEV_PATH = PHASE7_DIR / "bmo_rolling_through_2021q3" / "rolling_open_set_report.json"

INK = "183B56"
BLUE = "2E74B5"
TEAL = "2A7F9E"
LIGHT_BLUE = "E8EEF5"
LIGHTER_BLUE = "F4F7FA"
GRAY = "667085"
LIGHT_GRAY = "F2F4F7"
DARK = "1F2937"
GREEN = "2F7D5A"
LIGHT_GREEN = "EAF5EF"
GOLD = "A97516"
LIGHT_GOLD = "FFF5D9"
RED = "A33A3A"
LIGHT_RED = "FCECEC"
WHITE = "FFFFFF"
BLACK = "000000"

BODY_FONT = "Arial Unicode MS"
MONO_FONT = "Menlo"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float | int, digits: int = 2) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def fmt(value: float | int, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def safe_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"[\w.+-]+@[\w.-]+", "〔已去識別化帳號〕", text)
    return text


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def set_run_font(run, *, name: str = BODY_FONT, size: float | None = None,
                 color: str | None = None, bold: bool | None = None,
                 italic: bool | None = None) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top: int = 90, start: int = 120,
                     bottom: int = 90, end: int = 120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa: list[int], *, indent_dxa: int = 120) -> None:
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_layout = tbl_pr.find(qn("w:tblLayout"))
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tbl_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            set_cell_width(cell, widths_dxa[index])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_table_borders(table, color: str = "D0D5DD", size: int = 6) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), str(size))
        tag.set(qn("w:space"), "0")
        tag.set(qn("w:color"), color)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_keep_with_next(paragraph, value: bool = True) -> None:
    paragraph.paragraph_format.keep_with_next = value


def set_keep_together(paragraph, value: bool = True) -> None:
    paragraph.paragraph_format.keep_together = value


def add_field(paragraph, instruction: str, display: str = "") -> None:
    run = paragraph.add_run()
    fld_char = OxmlElement("w:fldChar")
    fld_char.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text_node = OxmlElement("w:t")
    text_node.text = display
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char, instr, separate, text_node, end])
    set_run_font(run, size=9, color=GRAY)


def add_caption(doc: Document, kind: str, number: int, text: str) -> None:
    p = doc.add_paragraph(style="Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    set_keep_with_next(p, False)
    label = "圖" if kind == "Figure" else "表"
    r = p.add_run(f"{label} ")
    set_run_font(r, size=9, color=GRAY, bold=True)
    add_field(p, f"SEQ {kind} \\* ARABIC", str(number))
    r = p.add_run(f"　{text}")
    set_run_font(r, size=9, color=GRAY)


def add_toc(doc: Document) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run()
    fld_char = OxmlElement("w:fldChar")
    fld_char.set(qn("w:fldCharType"), "begin")
    fld_char.set(qn("w:dirty"), "true")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "目錄將於開啟文件時自動更新。"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char, instr, separate, placeholder, end])
    set_run_font(run, size=10.5, color=GRAY)


def set_update_fields(doc: Document) -> None:
    settings = doc.settings._element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = BODY_FONT
    normal._element.rPr.rFonts.set(qn("w:ascii"), BODY_FONT)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), BODY_FONT)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(DARK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    heading_tokens = {
        "Heading 1": (16, BLUE, 16, 8),
        "Heading 2": (13, BLUE, 12, 6),
        "Heading 3": (11.5, INK, 8, 4),
    }
    for name, (size, color, before, after) in heading_tokens.items():
        style = styles[name]
        style.font.name = BODY_FONT
        style._element.rPr.rFonts.set(qn("w:ascii"), BODY_FONT)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), BODY_FONT)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True

    for name in ("List Bullet", "List Number"):
        style = styles[name]
        style.font.name = BODY_FONT
        style._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        style.font.size = Pt(10.5)
        style.paragraph_format.left_indent = Inches(0.5)
        style.paragraph_format.first_line_indent = Inches(-0.25)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.line_spacing = 1.167

    cap = styles["Caption"]
    cap.font.name = BODY_FONT
    cap._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    cap.font.size = Pt(9)
    cap.font.color.rgb = RGBColor.from_string(GRAY)
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(8)


def set_page_geometry(doc: Document) -> None:
    for section in doc.sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.header_distance = Inches(0.42)
        section.footer_distance = Inches(0.42)
        section.different_first_page_header_footer = True


def _configure_header(header) -> None:
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(2)
    # Keep a restrained rule-only header. LibreOffice's left-page header style
    # inconsistently drops CJK runs even when the same font renders in the body.
    p_pr = p._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "4")
    bottom.set(qn("w:space"), "3")
    bottom.set(qn("w:color"), "D0D5DD")
    p_bdr.append(bottom)
    p_pr.append(p_bdr)

def _configure_footer(footer) -> None:
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    r = p.add_run("— ")
    set_run_font(r, size=8.5, color=GRAY)
    add_field(p, "PAGE", "1")
    r = p.add_run(" —")
    set_run_font(r, size=8.5, color=GRAY)


def configure_header_footer(section) -> None:
    # Populate both odd and even variants explicitly. LibreOffice otherwise
    # suppresses the CJK runs in the shared header on left-hand pages.
    _configure_header(section.header)
    _configure_header(section.even_page_header)
    _configure_footer(section.footer)
    _configure_footer(section.even_page_footer)


def add_paragraph(doc: Document, text: str = "", *, bold_prefix: str | None = None,
                  align: WD_ALIGN_PARAGRAPH | None = None,
                  keep: bool = False) -> Any:
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if bold_prefix and text.startswith(bold_prefix):
        r = p.add_run(bold_prefix)
        set_run_font(r, bold=True)
        r = p.add_run(text[len(bold_prefix):])
        set_run_font(r)
    else:
        r = p.add_run(text)
        set_run_font(r)
    if keep:
        set_keep_together(p)
    return p


def add_bullet(doc: Document, text: str, level: int = 0) -> Any:
    p = doc.add_paragraph(style="List Bullet")
    if level:
        p.paragraph_format.left_indent = Inches(0.5 + 0.25 * level)
    r = p.add_run(text)
    set_run_font(r)
    return p


def add_number(doc: Document, text: str) -> Any:
    p = doc.add_paragraph(style="List Number")
    r = p.add_run(text)
    set_run_font(r)
    return p


def add_callout(doc: Document, label: str, text: str, *, fill: str = LIGHT_BLUE,
                label_color: str = BLUE) -> None:
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    set_table_borders(table, color=fill, size=4)
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.12
    r = p.add_run(f"{label}　")
    set_run_font(r, size=10, color=label_color, bold=True)
    r = p.add_run(text)
    set_run_font(r, size=10, color=DARK)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(1)


def add_table(doc: Document, headers: list[str], rows: Iterable[Iterable[Any]],
              widths: list[int], *, font_size: float = 8.5,
              header_fill: str = LIGHT_BLUE, first_col_bold: bool = False,
              caption: tuple[int, str] | None = None) -> Any:
    rows = list(rows)
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    set_table_borders(table)
    set_repeat_table_header(table.rows[0])
    for idx, text in enumerate(headers):
        cell = table.rows[0].cells[idx]
        set_cell_shading(cell, header_fill)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.05
        r = p.add_run(str(text))
        set_run_font(r, size=font_size, color=INK, bold=True)
    for row_values in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row_values):
            p = cells[idx].paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            if idx > 0 and len(str(value)) < 18:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(safe_text(value))
            set_run_font(r, size=font_size, color=DARK, bold=bool(first_col_bold and idx == 0))
        for cell in cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if caption:
        add_caption(doc, "Table", caption[0], caption[1])
    return table


def add_code_block(doc: Document, lines: list[str]) -> None:
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    set_table_borders(table, color="D0D5DD", size=4)
    cell = table.cell(0, 0)
    set_cell_shading(cell, "F7F8FA")
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.05
    for index, line in enumerate(lines):
        if index:
            p.add_run().add_break()
        r = p.add_run(line)
        set_run_font(r, name=MONO_FONT, size=8.2, color=DARK)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(1)


def add_picture(doc: Document, path: Path, width: float, caption: tuple[int, str]) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    set_keep_with_next(p)
    p.add_run().add_picture(str(path), width=Inches(width))
    add_caption(doc, "Figure", caption[0], caption[1])


def font_prop(size: int = 10):
    from matplotlib.font_manager import FontProperties

    path = Path("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc")
    return FontProperties(fname=str(path), size=size) if path.exists() else None


def save_baseline_chart(metrics: dict[str, Any], path: Path) -> None:
    baselines = metrics["baselines"]
    keys = ["global_majority", "component_majority", "bm25_text_knn", "current_assignee_triager"]
    labels = ["全域多數", "元件多數", "BM25 文字", "Hybrid"]
    top1 = [baselines[k]["top1_accuracy"] for k in keys]
    hit3 = [baselines[k]["hit_at_3"] for k in keys]
    mrr = [baselines[k]["mrr"] for k in keys]
    x = np.arange(len(keys))
    width = 0.23
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=180)
    ax.bar(x - width, top1, width, label="Top-1", color="#2E74B5")
    ax.bar(x, hit3, width, label="Hit@3", color="#2A7F9E")
    ax.bar(x + width, mrr, width, label="MRR", color="#C69428")
    ax.set_ylim(0, 1)
    ax.set_ylabel("比率", fontproperties=font_prop())
    ax.set_xticks(x, labels, fontproperties=font_prop())
    ax.grid(axis="y", alpha=0.2)
    ax.legend(prop=font_prop(9), frameon=False, ncol=3, loc="upper left")
    for pos, values in ((x - width, top1), (x, hit3), (x + width, mrr)):
        for xi, value in zip(pos, values):
            ax.text(xi, value + 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_routing_chart(report: dict[str, Any], path: Path) -> None:
    routing = report["routing_metrics"]
    labels = ["自動指派", "Top-3 確認", "人工分流"]
    values = [routing["auto_assignment_coverage"], routing["top3_confirmation_rate"], routing["manual_triage_rate"]]
    accuracy = [routing["auto_assignment_accuracy"], routing["top3_confirmation_accuracy"], None]
    colors = ["#2F7D5A", "#2E74B5", "#98A2B3"]
    fig, ax = plt.subplots(figsize=(8.8, 3.9), dpi=180)
    bars = ax.barh(labels, values, color=colors, height=0.55)
    ax.set_xlim(0, 0.65)
    ax.set_xlabel("全部 Ticket 中的比例", fontproperties=font_prop())
    ax.set_yticks(range(len(labels)), labels, fontproperties=font_prop())
    ax.grid(axis="x", alpha=0.2)
    for bar, value, acc in zip(bars, values, accuracy):
        suffix = f"；正確率 {acc*100:.2f}%" if acc is not None else ""
        ax.text(value + 0.012, bar.get_y() + bar.get_height()/2, f"{value*100:.2f}%{suffix}", va="center", fontproperties=font_prop(9))
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_confusion_chart(rows: list[dict[str, Any]], path: Path) -> list[str]:
    expected_counts = Counter(row["expected_assignee"] for row in rows if str(row["expected_assignee"]).startswith("dev_"))
    labels = [name for name, _ in expected_counts.most_common(7)]
    matrix = np.zeros((len(labels), len(labels) + 1), dtype=int)
    for row in rows:
        expected = row["expected_assignee"]
        if expected not in labels:
            continue
        pred = row["predicted_assignee"]
        i = labels.index(expected)
        j = labels.index(pred) if pred in labels else len(labels)
        matrix[i, j] += 1
    fig, ax = plt.subplots(figsize=(8.7, 5.8), dpi=180)
    im = ax.imshow(matrix, cmap="Blues")
    xlabels = labels + ["其他"]
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right", fontproperties=font_prop(8))
    ax.set_yticks(range(len(labels)), labels, fontproperties=font_prop(8))
    ax.set_xlabel("模型預測負責人", fontproperties=font_prop())
    ax.set_ylabel("真實負責人", fontproperties=font_prop())
    threshold = matrix.max() * 0.55 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > threshold else "#183B56", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return labels


def save_flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fp = font_prop(9)

    def box(x, y, w, h, text, color="#E8EEF5", edge="#2E74B5", size=9):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, linewidth=1.4, joinstyle="round")
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontproperties=font_prop(size), wrap=True, color="#183B56")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#667085", lw=1.5))

    box(0.45, 8.45, 2.1, 0.9, "Ticket 輸入\n標題／描述／元件／產品", "#F4F7FA")
    box(3.05, 8.45, 2.0, 0.9, "清理與時間過濾\n排除未來標籤", "#F4F7FA")
    box(5.55, 8.45, 1.9, 0.9, "歷史責任輪廓\n頻率／近期活動", "#EAF5EF", "#2F7D5A")
    box(7.95, 8.45, 1.6, 0.9, "候選池\nTop-30", "#EAF5EF", "#2F7D5A")
    arrow(2.55, 8.9, 3.05, 8.9); arrow(5.05, 8.9, 5.55, 8.9); arrow(7.45, 8.9, 7.95, 8.9)

    box(0.6, 6.5, 1.7, 0.85, "產品＋元件\n歷史比例")
    box(2.55, 6.5, 1.45, 0.85, "BM25\n文字相似")
    box(4.25, 6.5, 1.45, 0.85, "SBERT\n語意相似")
    box(5.95, 6.5, 1.55, 0.85, "30／90 日\n活動與衰減")
    box(7.75, 6.5, 1.65, 0.85, "來源一致性\n與全域先驗")
    for x in (1.45, 3.275, 4.975, 6.725, 8.575):
        arrow(x, 6.5, 8.75, 5.65)
    box(6.6, 4.65, 2.8, 1.0, "候選人學習排序（LTR）\n淺層 HistGradientBoosting\n輸出 Top-k 與排序機率", "#FFF5D9", "#A97516")
    arrow(8.0, 4.65, 7.0, 3.85)
    box(5.5, 2.85, 1.95, 0.95, "Logistic 校準\nTop-1 正確機率", "#FCECEC", "#A33A3A")
    box(7.7, 2.85, 1.95, 0.95, "開放集合偵測\n未知負責人風險", "#FCECEC", "#A33A3A")
    arrow(7.45, 3.3, 7.7, 3.3)
    box(3.65, 1.25, 1.8, 0.9, "高信心＋低風險\n自動指派", "#EAF5EF", "#2F7D5A")
    box(5.85, 1.25, 1.8, 0.9, "中信心\nTop-3 人工確認", "#E8EEF5", "#2E74B5")
    box(8.05, 1.25, 1.5, 0.9, "低信心／漂移\n人工分流", "#F2F4F7", "#667085")
    for x in (4.55, 6.75, 8.8):
        arrow(7.6, 2.85, x, 2.15)
    ax.text(0.45, 0.35, "註：Phase 7 為離線研究流程；目前整合執行時預設仍採 Hybrid fail-closed 路由。", fontproperties=fp, color="#667085")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dataset_stats(paths: list[Path]) -> dict[str, Any]:
    rows = [row for path in paths for row in load_jsonl(path)]
    title_lengths = [len(str(row.get("title", "")).split()) for row in rows]
    desc_lengths = [len(str(row.get("description", "")).split()) for row in rows]
    return {
        "rows": len(rows),
        "description_nonempty": sum(bool(str(row.get("description", "")).strip()) for row in rows),
        "description_nonempty_rate": sum(bool(str(row.get("description", "")).strip()) for row in rows) / len(rows),
        "title_median_words": float(np.median(title_lengths)),
        "title_under_5_rate": sum(value < 5 for value in title_lengths) / len(rows),
        "description_median_words": float(np.median(desc_lengths)),
    }


def per_assignee_summary(rows: list[dict[str, Any]], minimum: int = 12) -> tuple[list[list[Any]], list[list[Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        owner = str(row.get("expected_assignee") or "")
        if owner.startswith("dev_"):
            groups[owner].append(row)
    items = []
    for owner, values in groups.items():
        if len(values) < minimum:
            continue
        top1 = sum(bool(v.get("is_top1_correct")) for v in values) / len(values)
        top3 = sum(bool(v.get("is_top3_correct")) for v in values) / len(values)
        items.append((owner, len(values), top1, top3))
    best = sorted(items, key=lambda x: (-x[2], -x[1], x[0]))[:5]
    worst = sorted(items, key=lambda x: (x[2], -x[1], x[0]))[:5]
    return (
        [[owner, n, pct(t1), pct(t3)] for owner, n, t1, t3 in best],
        [[owner, n, pct(t1), pct(t3)] for owner, n, t1, t3 in worst],
    )


def confusion_pairs(rows: list[dict[str, Any]]) -> list[list[Any]]:
    counter = Counter()
    for row in rows:
        expected = str(row.get("expected_assignee") or "")
        predicted = str(row.get("predicted_assignee") or "")
        if expected != predicted and expected.startswith("dev_") and predicted.startswith("dev_"):
            counter[(expected, predicted)] += 1
    return [[expected, predicted, count] for (expected, predicted), count in counter.most_common(8)]


def select_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = ["bmo_1734282", "bmo_1735157", "bmo_1737884", "bmo_1743093", "bmo_1733740"]
    by_id = {row["ticket_id"]: row for row in rows}
    return [by_id[value] for value in ids]


def case_reason(row: dict[str, Any]) -> str:
    ticket = row["ticket_id"]
    reasons = {
        "bmo_1734282": "標題含 scroll frame API，與歷史元件／文字責任輪廓一致；Top-1 分數高，校準後仍跨過自動指派門檻。",
        "bmo_1735157": "JS JIT 斷言與歷史同類錯誤高度相近；BM25、SBERT 與元件活動信號形成一致證據。",
        "bmo_1737884": "WPT 關鍵字強烈拉向高頻同步帳號 dev_0152，真實負責人僅排第 5；顯示模板化標題與多數類偏誤。",
        "bmo_1743093": "同為 WPT 同步類型，模型偏向全域高頻 dev_0152；真實負責人已在第 3 名，Top-3 名單合理但不宜直接自動指派。",
        "bmo_1733740": "Top-1 與真實負責人接近，真實負責人排第 2；校準信心不足，因此正確採人工分流，展示拒絕機制的安全價值。",
    }
    return reasons[ticket]


def build_document() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_json(SUMMARY_PATH)
    desc_summary = load_json(DESC_SUMMARY_PATH)
    base_metrics = load_json(BASE_METRICS_PATH)
    phase6 = load_json(PHASE6_REPORT_PATH)
    phase6_cal = load_json(PHASE6_CAL_PATH)
    phase7 = load_json(PHASE7_REPORT_PATH)
    phase7_dev = load_json(PHASE7_DEV_PATH)
    predictions = load_jsonl(PHASE7_PRED_PATH)

    base_stats = dataset_stats([
        DATA_DIR / "processed" / "bmo_public_10k_history_train.jsonl",
        DATA_DIR / "processed" / "bmo_public_10k_validation_set.jsonl",
        DATA_DIR / "processed" / "bmo_public_10k_test_set.jsonl",
    ])
    desc_stats = dataset_stats([
        DATA_DIR / "processed" / "bmo_public_10k_desc_history_train.jsonl",
        DATA_DIR / "processed" / "bmo_public_10k_desc_validation_set.jsonl",
        DATA_DIR / "processed" / "bmo_public_10k_desc_test_set.jsonl",
    ])

    baseline_chart = ASSET_DIR / "baseline_metrics.png"
    routing_chart = ASSET_DIR / "routing_outcomes.png"
    confusion_chart = ASSET_DIR / "confusion_top7.png"
    flowchart = ASSET_DIR / "assignee_flow.png"
    missing_assets = [
        path for path in (baseline_chart, routing_chart, confusion_chart, flowchart)
        if not path.exists()
    ]
    if missing_assets:
        raise SystemExit(
            "Missing chart assets. Run make_charts.py first: "
            + ", ".join(str(path) for path in missing_assets)
        )

    doc = Document()
    configure_styles(doc)
    set_page_geometry(doc)
    doc.settings.odd_and_even_pages_header_footer = True
    configure_header_footer(doc.sections[0])
    set_update_fields(doc)
    doc.core_properties.title = "自動分流給負責人功能－技術流程與測試進度報告"
    doc.core_properties.subject = "自動分流技術流程、資料集、模型、測試與評估"
    doc.core_properties.author = "專題團隊"
    doc.core_properties.keywords = "assignee triage, bug routing, BM25, SBERT, LTR, open set"

    # Cover: editorial_cover pattern, standard_business_brief tokens.
    for _ in range(5):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(10)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("專題進度報告")
    set_run_font(r, size=11, color=GOLD, bold=True)
    p.paragraph_format.space_after = Pt(18)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("自動分流給負責人功能")
    set_run_font(r, size=28, color=INK, bold=True)
    p.paragraph_format.space_after = Pt(5)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("技術流程與測試進度報告")
    set_run_font(r, size=18, color=BLUE, bold=True)
    p.paragraph_format.space_after = Pt(24)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Assignee Triage Technical Workflow and Evaluation Progress")
    set_run_font(r, size=10.5, color=GRAY, italic=True)
    p.paragraph_format.space_after = Pt(80)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("報告範圍：技術流程、資料集、模型方法、測試與結果判讀")
    set_run_font(r, size=10.5, color=DARK)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("專案證據盤點日期：2026 年 7 月 15 日")
    set_run_font(r, size=10.5, color=GRAY)
    p.paragraph_format.space_after = Pt(6)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("狀態：離線研究成果已完成；正式執行時仍採保守人工回退")
    set_run_font(r, size=10.5, color=RED, bold=True)
    doc.add_page_break()

    doc.add_heading("目錄", level=1)
    add_table(
        doc,
        ["章節", "頁次"],
        [
            ["報告摘要", "3"],
            ["一、功能目的與問題定義", "3"],
            ["二、資料集內容與用途", "4"],
            ["三、自動分流的完整處理流程", "5"],
            ["四、目前使用的模型與技術", "7"],
            ["五、資料集測試與實驗設計", "8"],
            ["六、評估指標與結果判讀", "10"],
            ["七、實際案例分析", "14"],
            ["八、目前功能完成度", "14"],
            ["九、目前問題、限制與合理結論", "15"],
            ["附錄 A：主要程式檔案與用途", "16"],
            ["附錄 B：實際引用的資料與結果檔案", "17"],
            ["附錄 C：目前缺少或建議補做的測試", "18"],
        ],
        [8160, 1200],
        font_size=9.2,
        header_fill=LIGHT_GRAY,
    )
    doc.add_page_break()

    doc.add_heading("報告摘要", level=1)
    add_callout(
        doc,
        "核心結論",
        "目前功能已具備可執行的 Hybrid 候選排序、候選清單輸出、人工回退、時間洩漏稽核、信心校準、開放集合風險與滾動時間留出評估。最新 Phase 7 冷凍 Q4 測試顯示，高信心自動指派可達 97.75% 正確率，但只覆蓋 24.04% Ticket；54.91% 仍需人工分流。因此可支持『選擇性自動化＋候選人建議』，不能支持全面自動指派。",
        fill=LIGHT_GOLD,
        label_color=GOLD,
    )
    add_table(
        doc,
        ["面向", "已確認的進度", "證據"],
        [
            ["資料", "BMO 公開 10k、描述增補版與後續季度時間窗已整理", "資料摘要、fetch manifest、processed JSONL"],
            ["模型", "Hybrid runtime；Phase 6/7 完成候選 LTR、SBERT、校準與 open-set 研究", "assignee_triager.py、train_assignee_ltr.py、Phase 7 報告"],
            ["測試", "時間切分、重複稽核、leave-assignee-out、未見季度 holdout", "Q4 共 2,779 筆，未用於擬合或選閾值"],
            ["品質", "本次盤點重新執行全測試：100 tests OK", "PYTHONPATH=src python3 -m unittest discover -s tests"],
            ["部署判斷", "研究 bundle 仍為 research_only；production_integration_allowed=false", "rolling_holdout_report.json"],
        ],
        [1150, 4520, 3690],
        font_size=8.5,
        caption=(1, "本次進度報告的證據摘要"),
    )
    add_paragraph(doc, "本文件只討論『自動分流給負責人』，不介紹前端、伺服器、部署拓樸或未來基礎設施規劃。")

    doc.add_heading("一、功能目的與問題定義", level=1)
    doc.add_heading("1.1 問題定義", level=2)
    add_paragraph(doc, "自動分流（Assignee Triage，根據問題內容推薦最適合處理者）要解決的是：新 Ticket／Issue 到達後，系統如何利用標題、描述、產品、元件與過往處理紀錄，產生負責人候選排序，並在證據足夠時自動指派；若信心不足、候選人未出現在歷史中或資料發生漂移，則安全地交回人工分流。")
    add_table(
        doc,
        ["項目", "目前定義"],
        [
            ["模型輸入", "title、description、product、component、priority、severity、created_at；執行時亦可讀取 file_paths、ownership map、active roster 與人工 feedback。"],
            ["模型輸出", "assignee／suggested_assignee、ranked_candidates、candidate_scores/details、confidence、open_set_risk、routing_status、fallback_reason。"],
            ["主要任務", "單一標籤排序：真實標籤是歷史紀錄中的最終 assignee；同時評估 Top-k 候選與是否拒絕自動指派。"],
            ["目前範圍", "公開 Mozilla Bugzilla（BMO）資料上的離線研究，以及整合管線中的保守 Hybrid runtime。"],
            ["不在本報告範圍", "前端顯示、系統部署、伺服器設定、容量規劃。"],
        ],
        [1880, 7480],
        caption=(2, "功能輸入、輸出與範圍"),
    )
    doc.add_heading("1.2 兩條必須區分的實作線", level=2)
    add_bullet(doc, "實際整合執行線：src/modules/assignee_triager.py 的 Hybrid 確定性排序器；預設未核准校準器時，文字／歷史建議不直接自動指派，而是 manual_triage。")
    add_bullet(doc, "離線研究線：Phase 6/7 的候選 LTR、SBERT、Logistic 校準、未知負責人風險與多時間窗策略；結果仍標記 research_only，尚未自動取代 runtime。")
    add_callout(doc, "報告原則", "有程式碼不等於效果已證明；有離線高分也不等於已核准正式自動指派。", fill=LIGHT_RED, label_color=RED)

    doc.add_heading("二、資料集內容與用途", level=1)
    doc.add_heading("2.1 主要資料來源與規模", level=2)
    add_paragraph(doc, "主要基準資料由 fetch_bmo_assignee_dataset.py 透過 Mozilla Bugzilla REST API 擷取，條件為 Core／Firefox 產品、RESOLVED／VERIFIED／CLOSED 且 resolution=FIXED、建立時間自 2020-01-01 起。抓取上限為 10,000，因此 manifest 的 complete=false 表示達到上限，不是下載失敗。")
    add_table(
        doc,
        ["資料版本", "用途", "筆數／切分", "重要特性"],
        [
            ["BMO public 10k", "閉集合基準與 runtime 歷史", f"原始 10,000；可用去重前 {summary['usable_rows_before_frequency_filter']:,}；train/val/test = {summary['train_rows']:,}/{summary['validation_rows']:,}/{summary['test_rows']:,}", "154 位候選；description 全空；時間排序"],
            ["BMO public 10k desc", "描述欄位消融", f"train/val/test = {desc_summary['train_rows']:,}/{desc_summary['validation_rows']:,}/{desc_summary['test_rows']:,}", f"描述非空約 {pct(desc_stats['description_nonempty_rate'])}"],
            ["Phase 6 Q4 holdout", "候選 LTR 與校準安全門檻", "2,705 筆；known 2,342／unseen 363", "新季度；unseen 13.42%"],
            ["Phase 7 2021 Q4", "最終冷凍滾動 open-world 測試", "載入 2,792；去重後 2,779；known 2,638／unseen 141", "未參與擬合或閾值選擇"],
            ["Controlled／Eclipse", "功能煙霧與小型 pilot", "Controlled 26；Eclipse test 25", "不可當作最終研究結論"],
        ],
        [1800, 2050, 2900, 2610],
        font_size=8.1,
        caption=(3, "目前自動分流資料集與用途"),
    )
    doc.add_heading("2.2 每筆資料與欄位意義", level=2)
    add_paragraph(doc, "每一筆 processed JSONL 代表一張已解決且有最終負責人的 Bugzilla Ticket。prepare_paper_grade_dataset.py 將原始欄位正規化，再依建立時間排序、去除 generic assignee、合併顯式 duplicate 與完全相同的 title+description，最後只保留訓練期至少 10 筆的負責人。")
    add_table(
        doc,
        ["欄位", "意義", "在模型中的作用"],
        [
            ["ticket_id", "公開 Bug ID 轉為 bmo_*", "跨 split 重複與稽核鍵；不作語意特徵"],
            ["title", "Bug 摘要／標題", "BM25、SBERT 與關鍵字相似的主要文字"],
            ["description", "首則公開留言或描述", "補充錯誤、重現與模組語意；未清理時可能引入樣板噪音"],
            ["product／component", "產品與功能元件", "歷史 ownership、候選生成與強訊號"],
            ["priority／severity", "既有優先度與嚴重度", "文字／metadata 輔助特徵；不是負責人真值"],
            ["assignee", "最終 assigned_to；處理後匿名為 dev_####", "監督式標籤與評估真值"],
            ["created_at", "Ticket 建立時間", "時間切分、rolling history 與避免未來資料洩漏"],
            ["last_change_time", "最後變更時間（原始資料）", "Phase 7 作為標籤可見時間的保守代理"],
            ["duplicate_of", "重複 Ticket 指向", "跨資料切分前合併 duplicate chain"],
        ],
        [1520, 3070, 4770],
        font_size=8.2,
        caption=(4, "主要欄位與模型用途"),
    )

    doc.add_heading("2.3 去識別化資料範例", level=2)
    add_paragraph(doc, "下列範例取自 description-enriched test split；Ticket ID 改為 T-001～T-003，負責人已由資料準備程式匿名化，描述僅保留摘要，不呈現帳號、Email 或人名。")
    add_table(
        doc,
        ["代碼", "標題／描述摘要", "產品／元件", "P／S", "真實負責人"],
        [
            ["T-001", "延後建立共用 sandbox broker policy；描述指出 Fission 增加 content process，並討論偏好設定快取。", "Core／Security: Process Sandboxing", "P1／S4", "dev_0076"],
            ["T-002", "同步 web-platform-tests 的 transform-box reftests；描述含 PR 同步樣板。", "Core／Web Painting", "P4／--", "dev_0153"],
            ["T-003", "更新 cookie-store.idl；描述顯示自動建立的同步 PR。", "Core／DOM: Core & HTML", "P4／--", "dev_0153"],
        ],
        [750, 4050, 2050, 800, 1710],
        font_size=7.9,
        caption=(5, "實際資料去識別化範例"),
    )
    doc.add_heading("2.4 資料品質與適用性", level=2)
    imbalance = base_metrics["class_imbalance"]
    add_bullet(doc, f"類別不平衡：訓練最大／最小類別為 {imbalance['train_max_count']}／{imbalance['train_min_count']}，比例 {imbalance['train_imbalance_ratio']:.1f} 倍；最大類別占 {pct(imbalance['train_top_assignee_share'])}。")
    add_bullet(doc, f"主要 10k 版本沒有 description（{base_stats['description_nonempty']}/{base_stats['rows']}）；標題中位數 {base_stats['title_median_words']:.0f} 詞，{pct(base_stats['title_under_5_rate'])} 少於 5 詞。")
    add_bullet(doc, f"描述增補版的描述非空率 {pct(desc_stats['description_nonempty_rate'])}，中位數 {desc_stats['description_median_words']:.0f} 詞；但 raw first-comment 含同步樣板、連結與引用，Phase 2 顯示 full description 反而使 Hybrid Top-1 下降 2.08 個百分點。")
    add_bullet(doc, "真實標籤是公開快照中的最終 assigned_to，不代表『最佳』或最初指派者；多人協作、輪值與組織變動會造成標籤噪音。")
    add_bullet(doc, "資料適合測試：具有真實問題文字、元件、時間與最終負責人，可做時間留出及 Top-k；但只代表 Mozilla Core/Firefox，不足以證明跨專案泛化。")

    doc.add_heading("三、自動分流的完整處理流程", level=1)
    add_picture(doc, flowchart, 6.45, (1, "目前自動分流候選排序、信心校準與安全路由流程"))
    doc.add_heading("3.1 依實際程式順序", level=2)
    steps = [
        ("讀取與結構化 Ticket", "src/pipeline/orchestrator.py 呼叫 TicketExtractor，AssigneeTriager.assign() 接收 structured_ticket 與 priority_result。"),
        ("決定可用歷史快照", "AssigneeTriager._history_profile(as_of=Ticket 時間) 載入 JSONL；有 feedback 時僅合併當時已確認的紀錄，排除未來時間。"),
        ("清理欄位", "_normalize_component／_normalize_assignee 標準化；_row_text 串接 title、description、product、component、severity、priority。"),
        ("建立責任輪廓", "_build_profile 建立 global、product、component、product+component 的負責人計數與 token 文件頻率；build_assignee_profiles.py 另輸出可讀的責任摘要。"),
        ("候選生成", "runtime 合併 component／product+component／product／BM25-like 文字信號；Phase 6 研究流程以來源 quota 組成 Top-30 候選池。"),
        ("計算候選特徵", "歷史比例、平滑 ownership、BM25、SBERT、30/90 日活動、時間衰減、owner recency、來源一致性。"),
        ("候選排序", "runtime 使用固定權重 Hybrid；Phase 6/7 使用淺層 HistGradientBoosting 對候選人重新排序。"),
        ("信心與未知風險", "_confidence 或 portable Logistic 估計 Top-1 正確性；open-set detector 估計真實負責人未見風險；PSI 監控分數漂移。"),
        ("路由決策", "高信心低風險 auto_assign；中信心 top3_confirmation；低信心、未知風險、候選無效或漂移則 manual_triage。"),
        ("輸出與稽核", "輸出 ranked_candidates、scores、confidence、open_set_risk、routing_status、fallback_reason；orchestrator 寫入 assignee_prediction_result.json。"),
    ]
    for title, detail in steps:
        p = add_number(doc, "")
        r = p.add_run(title + "：")
        set_run_font(r, bold=True, color=INK)
        r = p.add_run(detail)
        set_run_font(r)
    doc.add_heading("3.2 資料不足與低信心處理", level=2)
    add_table(
        doc,
        ["情境", "處理方式", "程式輸出"],
        [
            ["沒有歷史或候選", "不猜測；交人工", "manual_triage／missing_assignee_history"],
            ["只有文字或廣泛先驗", "預設只建議，不自動指派", "text_only_or_broad_history_only"],
            ["信心低於 threshold", "人工分流", "low_confidence／calibrated_low_confidence"],
            ["未知負責人風險高", "人工分流", "manual_triage_open_set／unseen_owner_risk"],
            ["Active roster 遺失或無效", "fail closed", "active_roster_unavailable"],
            ["校準 artifact 未核准或版本不符", "拒絕套用，保留原始信心", "artifact_not_approved／version_mismatch"],
            ["PSI > 0.25", "Phase 7 禁止 auto_assign，降級為 Top-3 確認", "score_distribution_drift"],
        ],
        [2300, 3300, 3760],
        font_size=8.2,
        caption=(6, "異常、低信心與資料不足時的回退策略"),
    )

    doc.add_heading("四、目前使用的模型與技術", level=1)
    add_paragraph(doc, "技術名詞第一次出現時均附中文解釋。下列項目依『目前 runtime』與『已完成的離線研究』分別描述。")
    tech_rows = [
        ["歷史 ownership／規則", "依 product+component、component、product 的歷史負責比例與已審核 ownership map 推薦", "輸入 metadata＋歷史計數；輸出候選分數", "直觀、可解釋；但偏向高頻人員、專長會過時", "assignee_triager.py::_rank_candidates／_mapped_owner"],
        ["BM25（詞頻逆文件頻率排序）", "找出文字相似的歷史 Ticket，再把相似案件的負責人分數彙總", "輸入文字；輸出 owner similarity", "離線可跑、關鍵字強；同義詞與樣板文字有限制", "evaluate_paper_grade_benchmarks.py::BM25Index；train_assignee_ltr.py::BM25Index"],
        ["SBERT（句子語意向量）", "把 Ticket 轉成向量，以 cosine/dot product 找語意相似歷史案件", "輸入文字；輸出 sbert_owner_score", "能補詞彙不一致；需本機模型與快取，仍受領域偏移", "train_assignee_ltr.py::SemanticBackend"],
        ["候選來源 quota", "為 component、BM25、SBERT、近期活動等保留候選名額，避免單一來源壟斷 Top-30", "輸入各來源排序；輸出候選池", "提升 recall；池外真實負責人仍無法挽回", "train_assignee_ltr.py::source_quotas／CandidateIndex.candidates"],
        ["LTR（學習排序）", "將每個 Ticket×候選人視為二元樣本，學習誰應排前", "輸入候選特徵；輸出排序機率", "比固定權重彈性；需嚴格時間 fold 防洩漏", "train_assignee_ltr.py::fit_rankers／rank_with_model"],
        ["時間衰減與近期活動", "越近期的 component ownership 權重越高，半衰期預設 90 天", "輸入 created_at；輸出 recency shares", "反映專長變動；last_change_time 仍只是代理", "CandidateIndex.recency_counter／owner_recency"],
        ["Logistic 信心校準", "以 Top probability、margin、entropy、ownership 與來源一致性估計 Top-1 是否正確", "輸入排序診斷；輸出 calibrated_probability", "可選擇性自動化；跨時間可能失準", "train_assignee_rolling_open_set.py；assignee_open_set_common.py"],
        ["Open-set（開放集合）偵測", "估計真實負責人是否不在歷史候選中，使用 leave-assignee-out 合成未知案例", "輸入排序／歷史特徵；輸出 unknown risk", "保護新負責人；Q4 AUROC 僅 0.6963，會拒絕許多已知者", "build_leave_assignee_out_folds／fit_portable_logistic"],
        ["PSI（母體穩定度指標）", "比較分數分布是否漂移；>0.25 時停用自動指派", "輸入信心分布；輸出 drift flag", "簡單可稽核；只看分布，不能找出因果", "evaluate_score_drift"],
    ]
    add_table(
        doc,
        ["技術", "目的／原理", "輸入→輸出", "優點與限制", "實作位置"],
        tech_rows,
        [1450, 2280, 1600, 2180, 1850],
        font_size=7.1,
        caption=(7, "模型技術、角色、輸出與程式位置"),
    )
    doc.add_heading("4.1 runtime Hybrid 的固定分數", level=2)
    add_paragraph(doc, "AssigneeTriager._rank_candidates() 的主要權重為：product+component share×4、component share×3、product share×1、文字相似×1.5，已審核 component/file ownership 另給強信號。原始信心由 Top-1 分數與 Top-1/Top-2 margin 組合，再受 min_score、confidence_threshold、active roster、校準核准與 open-set gate 約束。")
    add_callout(doc, "重要限制", "Phase 7 的 HistGradientBoosting bundle 尚未接到 src/modules/assignee_triager.py 作為正式預設；因此最新離線分數不能描述為目前 runtime 已部署表現。", fill=LIGHT_RED, label_color=RED)

    doc.add_heading("五、資料集測試與實驗設計", level=1)
    doc.add_heading("5.1 閉集合基準", level=2)
    add_bullet(doc, "先依 created_at 排序，原始可用資料以 80%／10%／10% 切 train／validation／test，再以 train 中至少 10 筆的負責人建立 roster，並將 val/test 限制在 roster 內。")
    add_bullet(doc, f"凍結 test 為 {summary['test_rows']} 筆；leakage audit 為 ticket ID、exact title、exact title+description 重疊皆 0，且 train 中沒有晚於 test_start 的資料。")
    add_bullet(doc, "限制：同一 Mozilla 專案內切分，沒有 project-level holdout；早期 exact duplicate 稽核不等於完整語意近重複稽核。Phase 1 額外移除 18 筆近重複後，Top-1 從 0.7656 降至 0.7414，顯示近重複會樂觀化結果。")
    doc.add_heading("5.2 Phase 6/7 時間與開放世界設計", level=2)
    add_bullet(doc, "候選 LTR 的訓練樣本來自 train 內 expanding temporal folds；query 不使用未來歷史，長尾 query 權重為 1/sqrt(owner frequency)。")
    add_bullet(doc, "Phase 7 以 validation、2020 Q3/Q4 擬合校準與 open-set；2021 Q1/Q2/Q3 選同一組路由 policy；2021 Q4 只評估一次。")
    add_bullet(doc, f"Q4 query 2,792 筆，跨邊界去除 13 筆後評估 {phase7['protocol']['query_rows']} 筆；有 3,447 筆歷史因 last_change_time 晚於 cutoff 而被扣留。")
    add_bullet(doc, "Leave-assignee-out 將部分負責人從 history 移除，再用其 Ticket 形成 synthetic unknown；稽核要求 hidden owners 與 fold history 不重疊。")
    add_bullet(doc, "已明確作廢的 2021 Q3 初次結果不列入證據，原因是 creation-time history 暴露了尚不可見的 final assignee。")
    add_table(
        doc,
        ["問題", "目前處理", "仍有風險"],
        [
            ["同 Ticket 重複", "跨 split ticket ID 與 exact content 去重", "語意近重複不一定完全捕捉"],
            ["未來標籤洩漏", "時間切分；Phase 7 以 last_change_time 作 label_available_at", "缺少精確 assignee change history"],
            ["新負責人", "open-world known/unseen 分組＋leave-assignee-out", "公共資料沒有經審核的 active/departed roster"],
            ["類別不平衡", "最低 10 筆、Macro-F1、long-tail weighting", "高頻負責人仍主導"],
            ["跨專案泛化", "目前尚未完成", "核心外部效度缺口"],
        ],
        [1900, 3650, 3810],
        font_size=8.3,
        caption=(8, "資料洩漏與評估有效性的控制情況"),
    )
    doc.add_heading("5.3 執行指令與輸出", level=2)
    add_code_block(doc, [
        "# 閉集合基準",
        "python3 assignee_triage_accuracy/paper_grade/scripts/evaluate_paper_grade_benchmarks.py --dataset bmo_public_10k",
        "",
        "# Phase 6 候選 LTR",
        "python3 assignee_triage_accuracy/scripts/train_assignee_ltr.py --temporal-folds 3 --embedding-backend sbert",
        "",
        "# Phase 7 滾動 open-world 與冷凍 Q4",
        "python3 assignee_triage_accuracy/scripts/train_assignee_rolling_open_set.py",
        "python3 assignee_triage_accuracy/scripts/evaluate_assignee_rolling_holdout.py --holdout assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2021q4_raw.jsonl --holdout-name 2021_q4 --output-dir assignee_triage_accuracy/phase7_rolling_open_set/reports/bmo_public_2021q4_conservative_untouched_holdout --confirm-untouched-holdout",
        "",
        "# 本次程式品質檢查",
        "PYTHONPATH=src python3 -m unittest discover -s tests  # Ran 100 tests, OK",
    ])

    doc.add_heading("六、評估指標與結果判讀", level=1)
    doc.add_heading("6.1 指標定義", level=2)
    add_table(
        doc,
        ["指標", "代表意義", "判讀"],
        [
            ["Top-1 Accuracy", "第一名候選等於真實負責人的比例", "越高越好；單看此值易受多數類影響"],
            ["Hit@3／@5／@10", "真實負責人出現在前 k 名的比例", "衡量候選名單是否可供人工確認"],
            ["MRR", "真實負責人排名倒數的平均；第 1 名=1、第 2 名=0.5", "越高表示正確者越靠前"],
            ["Macro-F1", "每位負責人 F1 等權平均", "長尾類別差時會顯著下降；比 Accuracy 更能揭露不平衡"],
            ["Candidate pool recall", "真實負責人是否進入 Top-30 候選池", "排序器的上限；池外即無法選中"],
            ["Auto accuracy／coverage", "被自動指派樣本的正確率／占全部比例", "安全性與自動化程度需同時看"],
            ["Brier／ECE", "機率誤差與校準誤差", "越低越好；ECE 低表示信心較符合實際正確率"],
            ["Open-set AUROC／AUPRC", "區分已知與未知負責人的能力", "越高越好；未知比例低時 AUPRC 特別重要"],
            ["PSI", "開發與測試分數分布差異", "越低越穩定；本專案 >0.25 視為嚴重漂移"],
        ],
        [1800, 4600, 2960],
        font_size=8.1,
        caption=(9, "本專案採用的主要評估指標"),
    )
    add_paragraph(doc, "Precision、Recall 與 Weighted-F1 並未由主要 Phase 7 report 單獨輸出；目前專案尚未提供。單標籤多類別的 micro precision／recall 在數值上會等同 accuracy，但本報告不另行捏造未輸出的欄位。")

    doc.add_heading("6.2 BMO 10k 閉集合結果", level=2)
    base = base_metrics["baselines"]
    add_table(
        doc,
        ["方法", "Top-1", "Hit@3", "Hit@5", "Hit@10", "MRR", "Macro-F1"],
        [
            ["Global majority", fmt(base["global_majority"]["top1_accuracy"]), fmt(base["global_majority"]["hit_at_3"]), fmt(base["global_majority"]["hit_at_5"]), fmt(base["global_majority"]["hit_at_10"]), fmt(base["global_majority"]["mrr"]), fmt(base["global_majority"]["macro_f1"])],
            ["Component majority", fmt(base["component_majority"]["top1_accuracy"]), fmt(base["component_majority"]["hit_at_3"]), fmt(base["component_majority"]["hit_at_5"]), fmt(base["component_majority"]["hit_at_10"]), fmt(base["component_majority"]["mrr"]), fmt(base["component_majority"]["macro_f1"])],
            ["Product+component", fmt(base["product_component_majority"]["top1_accuracy"]), fmt(base["product_component_majority"]["hit_at_3"]), fmt(base["product_component_majority"]["hit_at_5"]), fmt(base["product_component_majority"]["hit_at_10"]), fmt(base["product_component_majority"]["mrr"]), fmt(base["product_component_majority"]["macro_f1"])],
            ["BM25 text kNN", fmt(base["bm25_text_knn"]["top1_accuracy"]), fmt(base["bm25_text_knn"]["hit_at_3"]), fmt(base["bm25_text_knn"]["hit_at_5"]), fmt(base["bm25_text_knn"]["hit_at_10"]), fmt(base["bm25_text_knn"]["mrr"]), fmt(base["bm25_text_knn"]["macro_f1"])],
            ["Hybrid ranker", fmt(base["current_assignee_triager"]["top1_accuracy"]), fmt(base["current_assignee_triager"]["hit_at_3"]), fmt(base["current_assignee_triager"]["hit_at_5"]), fmt(base["current_assignee_triager"]["hit_at_10"]), fmt(base["current_assignee_triager"]["mrr"]), fmt(base["current_assignee_triager"]["macro_f1"])],
        ],
        [2140, 1200, 1200, 1200, 1200, 1200, 1220],
        font_size=8.1,
        caption=(10, "BMO public 10k 凍結測試集的閉集合基準"),
    )
    add_picture(doc, baseline_chart, 6.35, (2, "閉集合基準方法的 Top-1、Hit@3 與 MRR 比較"))
    add_paragraph(doc, f"Hybrid Top-1 為 {pct(base['current_assignee_triager']['top1_accuracy'])}，高於 BM25 的 {pct(base['bm25_text_knn']['top1_accuracy'])}；但 Macro-F1 只有 {fmt(base['current_assignee_triager']['macro_f1'])}，遠低於 Accuracy，表示少數負責人表現仍差。Hybrid 的 Top-1 95% bootstrap CI 為 [{fmt(base['current_assignee_triager']['top1_bootstrap_95ci']['low'])}, {fmt(base['current_assignee_triager']['top1_bootstrap_95ci']['high'])}]。")
    add_callout(doc, "不可誤讀", "該閉集合 evaluation 的推薦 Top-1 為 52.71%，但 runtime 預設因未核准校準器而全部交人工（unable_to_decide_rate=100%）。推薦品質與正式自動指派覆蓋率是不同指標。", fill=LIGHT_GOLD, label_color=GOLD)

    doc.add_heading("6.3 Phase 6 與 Phase 7 結果", level=2)
    p6_known = phase6["test_exploratory_known_owner"]
    p7_known = phase7["recommendation_metrics_known_owner"]
    routing = phase7["routing_metrics"]
    add_table(
        doc,
        ["實驗", "Top-1", "Hit@3", "Hit@5", "MRR", "Macro-F1", "候選池 recall"],
        [
            ["Phase 6 Q4 known-owner shallow LTR", fmt(p6_known["top1_accuracy"]), fmt(p6_known["hit_at_3"]), fmt(p6_known["hit_at_5"]), fmt(p6_known["mrr"]), fmt(p6_known["macro_f1"]), fmt(p6_known["candidate_recall_at_pool"])],
            ["Phase 7 2021 Q4 known-owner", fmt(p7_known["top1_accuracy"]), fmt(p7_known["top3_accuracy"]), fmt(p7_known["top5_accuracy"]), fmt(p7_known["mrr"]), fmt(p7_known["macro_f1"]), fmt(p7_known["candidate_pool_recall"])],
            ["Phase 7 2021 Q4 全部 open-world", fmt(phase7["recommendation_metrics"]["top1_accuracy"]), fmt(phase7["recommendation_metrics"]["top3_accuracy"]), fmt(phase7["recommendation_metrics"]["top5_accuracy"]), fmt(phase7["recommendation_metrics"]["mrr"]), fmt(phase7["recommendation_metrics"]["macro_f1"]), fmt(phase7["recommendation_metrics"]["candidate_pool_recall"])],
        ],
        [2570, 1120, 1120, 1120, 1120, 1180, 1130],
        font_size=8.0,
        caption=(11, "候選 LTR 在後續時間留出資料的推薦結果"),
    )
    add_table(
        doc,
        ["Phase 7 Q4 路由指標", "結果", "判讀"],
        [
            ["自動指派", f"{routing['auto_assignment_rows']} 筆；coverage {pct(routing['auto_assignment_coverage'])}；accuracy {pct(routing['auto_assignment_accuracy'])}", "高精確率，但只處理約四分之一"],
            ["未知負責人誤自動指派", f"{routing['unseen_auto_assignment_rows']} 筆；rate {pct(routing['unseen_auto_assignment_rate'])}", "安全門檻通過"],
            ["Top-3 人工確認", f"{routing['top3_confirmation_rows']} 筆；rate {pct(routing['top3_confirmation_rate'])}；Top-3 accuracy {pct(routing['top3_confirmation_accuracy'])}", "名單可提供合理縮小範圍"],
            ["人工分流", f"rate {pct(routing['manual_triage_rate'])}", "超過一半仍無法自動化"],
            ["校準", f"Brier {fmt(phase7['calibration_metrics']['brier'])}；ECE {fmt(phase7['calibration_metrics']['ece'])}", "信心分布與正確率接近，但不等於排序準確"],
            ["Open-set", f"AUROC {fmt(phase7['open_set_metrics']['auroc'])}；AUPRC {fmt(phase7['open_set_metrics']['auprc'])}", "區分能力普通，未知比例低時 AUPRC 很低"],
            ["漂移", f"max PSI {fmt(phase7['score_drift']['maximum_psi'])} < 0.25", "本次未觸發 fail-closed drift gate"],
        ],
        [2300, 3550, 3510],
        font_size=8.0,
        caption=(12, "Phase 7 冷凍 2021 Q4 的安全路由結果"),
    )
    add_picture(doc, routing_chart, 6.35, (3, "Phase 7 Q4 自動指派、Top-3 確認與人工分流比例"))
    add_paragraph(doc, "這些數值足以證明『在嚴格拒絕機制下，可安全自動處理一部分高信心 Ticket』；不足以證明『模型能全面取代人工分流』。Known-owner Top-1 48.41%、Macro-F1 14.21%，加上 candidate pool recall 89.27%，顯示未來時間窗中的長尾與候選生成仍是主要瓶頸。")

    doc.add_heading("6.4 各負責人與錯誤型態", level=2)
    best, worst = per_assignee_summary(predictions)
    add_table(doc, ["較佳負責人（至少 12 筆）", "筆數", "Top-1", "Top-3"], best,
              [3600, 1200, 2280, 2280], font_size=8.3, caption=(13, "Phase 7 Q4 中樣本充足且表現較佳的匿名負責人"))
    add_table(doc, ["較差負責人（至少 12 筆）", "筆數", "Top-1", "Top-3"], worst,
              [3600, 1200, 2280, 2280], font_size=8.3, caption=(14, "Phase 7 Q4 中樣本充足但表現較差的匿名負責人"))
    add_picture(doc, confusion_chart, 6.2, (4, "Phase 7 Q4 最常見七位真實負責人的部分混淆矩陣（由逐筆預測衍生）"))
    add_table(doc, ["真實負責人", "最常誤預測為", "次數"], confusion_pairs(predictions),
              [3200, 3200, 2960], font_size=8.3, caption=(15, "Phase 7 Q4 最常見匿名錯誤配對"))
    add_paragraph(doc, "混淆矩陣顯示 dev_0152 常被模型選為其他負責人的替代答案，與其在 BMO 10k train 中占 22.31% 的高頻先驗一致。這也是 Accuracy 看似合理而 Macro-F1 偏低的主要原因之一。完整全類別混淆矩陣目前專案尚未輸出為檔案；本圖由 rolling_holdout_predictions.jsonl 依實際逐筆結果計算。")

    doc.add_heading("七、實際案例分析", level=1)
    add_paragraph(doc, "案例取自 Phase 7 2021 Q4 冷凍 holdout。Ticket ID 改為案例編號，僅保留已匿名的 dev_####，不呈現候選名單中的 Email 或原始帳號。")
    cases = select_cases(predictions)
    case_rows = []
    case_types = ["正確", "正確", "錯誤", "錯誤", "低信心／人工"]
    for idx, (case, kind) in enumerate(zip(cases, case_types), start=1):
        candidates = [value for value in case["ranked_candidates"] if str(value).startswith("dev_")][:3]
        true_rank = None
        if case["expected_assignee"] in case["candidate_pool"]:
            true_rank = case["candidate_pool"].index(case["expected_assignee"]) + 1
        case_rows.append([
            f"C-{idx:02d}\n{kind}",
            case["title"],
            f"真實：{case['expected_assignee']}\n預測：{case['predicted_assignee']}\n真實排名：{true_rank or '>30'}",
            f"Top prob {case['top_probability']:.3f}\n校準 {case['calibrated_probability']:.3f}\n未知風險 {case['open_set_probability']:.3f}\n路由：{case['routing_status']}",
            f"Top-3（僅匿名）：{', '.join(candidates)}\n{case_reason(case)}",
        ])
    add_table(
        doc,
        ["案例", "Ticket 標題／摘要", "真實與預測", "分數／路由", "原因與反映的優劣"],
        case_rows,
        [850, 2700, 1550, 1700, 2560],
        font_size=7.2,
        caption=(16, "兩筆正確、兩筆錯誤與一筆低信心案例"),
    )
    add_paragraph(doc, "案例 C-03、C-04 共同反映 WPT／同步樣板的文字相似性會把預測推向高頻負責人。C-04 的真實負責人已在 Top-3，表示候選清單仍有實用性，但自動指派門檻對此類模板化 Ticket 應更保守。C-05 則證明『拒絕』本身是功能的一部分：即使第一名錯誤，人工只需在前幾名中確認。")

    doc.add_heading("八、目前功能完成度", level=1)
    add_table(
        doc,
        ["狀態", "項目", "客觀判斷"],
        [
            ["已實作", "Ticket 欄位整理與歷史載入", "AssigneeTriager 可讀取 title/description/product/component 等欄位與 JSONL 歷史。"],
            ["已實作", "Hybrid 候選生成與 Top-k", "可輸出 ranked_candidates、candidate_scores、signals。"],
            ["已實作", "人工回退與 fail-closed", "缺資料、低信心、inactive、artifact 無效、open-set 風險均可回退。"],
            ["已實作", "責任輪廓與 feedback 更新", "可由歷史建立專長摘要，並將人工確認結果合併為後續歷史。"],
            ["已實作並測試", "資料準備、時間切分、去重與洩漏稽核", "BMO 10k 與 Phase 7 均有逐步 audit；本次相關 23 tests 及全套 100 tests 均 OK。"],
            ["已完成離線實驗", "候選 LTR、SBERT、校準、open-set、PSI", "有程式與 Q4 實驗結果，但 bundle 仍 research_only。"],
            ["部分證明", "選擇性自動指派", "Q4 97.75% accuracy@24.04% coverage；只證明高信心子集合。"],
            ["尚未完成", "Phase 7 正式 runtime 整合與核准", "production_integration_allowed=false；現行 runtime 未預設載入 Phase 7 bundle。"],
            ["尚未完成", "目標專案 active/departed roster 與權威 ownership", "公共 BMO 不提供可審核的現況名單。"],
            ["尚未完成", "跨專案泛化與完整 assignment history", "目前沒有 project holdout，也沒有精確指派變更時間。"],
        ],
        [1300, 2600, 5460],
        font_size=8.0,
        caption=(17, "程式實作與效果證明的完成度區分"),
    )

    doc.add_heading("九、目前問題、限制與合理結論", level=1)
    doc.add_heading("9.1 主要限制", level=2)
    limitations = [
        "負責人分布嚴重不平衡，高頻 dev_0152 在 train 占 22.31%；模型容易以高頻先驗取代少數專家。",
        "主要 10k baseline 沒有 description；增補描述雖覆蓋約 90%，但 raw first-comment 樣板與引用會降低 BM25／Hybrid 表現。",
        "最終 assigned_to 只代表歷史結果，不等於最佳初始分流；同一問題也可能由輪值、休假或跨團隊協作決定。",
        "Phase 7 以 last_change_time 當標籤可見時間，雖比 creation_time 保守，仍不是精確 assignment-change timestamp。",
        "Known-owner Top-1 48.41%、Macro-F1 14.21%，表示對少數負責人與未來資料的泛化仍弱。",
        "Open-set AUROC 0.6963、AUPRC 0.0819，未知偵測能力有限；目前靠保守 threshold 換取安全，導致人工率 54.91%。",
        "候選池 recall 在 Phase 7 known-owner 為 89.27%，約一成真實負責人連 Top-30 都未進入，排序器無法補救。",
        "資料只涵蓋 Mozilla Core/Firefox；尚未在另一專案、另一組團隊或真實線上流程驗證。",
        "公共資料包含可能已離職或改職責帳號，缺少經維護者審核的 active/inactive roster 與 ownership map。",
    ]
    for item in limitations:
        add_bullet(doc, item)

    doc.add_heading("9.2 可向教授報告的合理結論", level=2)
    add_callout(
        doc,
        "結論",
        "目前專題已完成從公開 Ticket 歷史建模、候選負責人產生、Top-k 排序、低信心拒絕，到時間留出與開放世界評估的完整研究流程。現有證據支持：系統可以在約四分之一的高信心案件上，以接近 98% 的正確率進行選擇性自動指派，並對另外約五分之一提供具 71% Hit@3 的人工確認名單。現有證據不支持：對所有 Ticket 自動指派、跨專案泛化、或把公共資料中的最終 assignee 視為唯一最佳答案。下一階段應優先補強候選池召回與長尾負責人、取得目標專案 active roster／精確 assignment history、進行跨專案與另一個未見時間窗測試；完成後才考慮核准 Phase 7 bundle 或比較 LLM reranker。",
        fill=LIGHT_GREEN,
        label_color=GREEN,
    )
    add_table(
        doc,
        ["優先度", "下一階段補強", "驗收方式"],
        [
            ["P0", "目標專案 active/inactive roster 與 authority ownership", "人工審核、版本化、缺檔 fail-closed 測試"],
            ["P0", "精確 assignment-change history／label availability", "重新跑 rolling split，確認每筆歷史在 query 當時可見"],
            ["P1", "候選池召回與長尾", "known-owner pool recall ≥90%，Macro-F1 顯著提升且多數類錯配下降"],
            ["P1", "跨專案或第二個未見季度 holdout", "固定模型與閾值後只評估一次，禁止回頭調參"],
            ["P2", "描述清理與模板去重", "title+metadata 對照 cleaned description 消融；近重複稽核"],
            ["P2", "LLM reranker 比較", "同一候選池、同一時間 protocol、報告延遲／成本／Top-k；未勝出不升級"],
        ],
        [900, 4150, 4310],
        font_size=8.1,
        caption=(18, "建議補做工作的優先順序與驗收準則"),
    )

    doc.add_heading("附錄 A：主要程式檔案與用途", level=1)
    file_rows = [
        ["src/modules/assignee_triager.py", "目前整合 runtime：Hybrid 排序、校準、open-set、roster、ownership 與 fallback。"],
        ["src/modules/assignee_deployment.py", "部署 bundle、時間群組切分、roster 與候選 ownership 建立。"],
        ["src/modules/assignee_feedback.py", "記錄與合併人工確認的最終負責人。"],
        ["src/pipeline/orchestrator.py", "整合管線第 4 步呼叫 assign() 並寫 checkpoint。"],
        ["src/config.py", "assignee dataset、Top-k、threshold、roster、artifact 路徑設定。"],
        ["assignee_triage_accuracy/paper_grade/scripts/fetch_bmo_assignee_dataset.py", "從 BMO REST 擷取公開資料與 manifest。"],
        ["assignee_triage_accuracy/paper_grade/scripts/prepare_paper_grade_dataset.py", "正規化、去重、時間切分、最低頻率過濾與匿名化。"],
        ["assignee_triage_accuracy/paper_grade/scripts/evaluate_paper_grade_benchmarks.py", "多基準 Top-k、MRR、Macro-F1、bootstrap、imbalance、leakage audit。"],
        ["assignee_triage_accuracy/scripts/build_assignee_profiles.py", "由 train-only 歷史推論每位負責人的責任領域。"],
        ["assignee_triage_accuracy/scripts/train_assignee_ltr.py", "Top-30 來源 quota、BM25/SBERT、候選特徵與 LTR。"],
        ["assignee_triage_accuracy/scripts/assignee_open_set_common.py", "portable Logistic、leave-assignee-out、routing policy、PSI。"],
        ["assignee_triage_accuracy/scripts/train_assignee_rolling_open_set.py", "多時間窗滾動歷史、校準與 open-set 訓練。"],
        ["assignee_triage_accuracy/scripts/evaluate_assignee_rolling_holdout.py", "冷凍 holdout 驗證、drift gate 與報告輸出。"],
        ["tests/test_assignee_deployment.py", "bundle、切分、roster 與部署保護測試。"],
        ["tests/test_assignee_ltr.py", "來源 quota、時間歷史、leave-owner-out、校準與 rolling 測試。"],
        ["tests/test_assignee_paper_dataset.py", "資料切分、重複與 metric 邏輯測試。"],
        ["tests/test_pipeline.py", "AssigneeTriager 的 fallback、artifact、roster、feedback 與整合行為。"],
    ]
    add_table(doc, ["檔案", "用途"], file_rows, [3800, 5560], font_size=7.7, caption=(19, "本功能主要程式與測試檔案"))

    doc.add_heading("附錄 B：實際引用的資料與結果檔案", level=1)
    data_rows = [
        [rel(DATA_DIR / "raw" / "bmo_public_10k_fetch_manifest.json"), "來源、條件、10,000 筆與 incomplete/cap 說明"],
        [rel(SUMMARY_PATH), "切分筆數、roster、元件與類別分布"],
        [rel(DESC_SUMMARY_PATH), "描述增補版切分與覆蓋"],
        [rel(DATA_DIR / "processed" / "bmo_public_10k_history_train.jsonl"), "閉集合歷史／訓練"],
        [rel(DATA_DIR / "processed" / "bmo_public_10k_validation_set.jsonl"), "校準／模型選擇"],
        [rel(DATA_DIR / "processed" / "bmo_public_10k_test_set.jsonl"), "凍結閉集合測試"],
        [rel(DATA_DIR / "processed" / "bmo_public_10k_desc_test_set.jsonl"), "去識別化資料範例與描述品質"],
        [rel(BASE_METRICS_PATH), "閉集合多基準、Macro-F1、bootstrap、imbalance、leakage"],
        [rel(PHASE6_REPORT_PATH), "Q4 候選 LTR known/unseen 與候選池結果"],
        [rel(PHASE6_CAL_PATH), "Phase 6 校準與 failed safety gate"],
        [rel(PHASE7_DEV_PATH), "Phase 7 fit/selection windows 與 frozen policy"],
        [rel(PHASE7_REPORT_PATH), "2021 Q4 冷凍 holdout、路由、校準、open-set、PSI"],
        [rel(PHASE7_PRED_PATH), "案例、各負責人正確率、混淆矩陣與錯誤配對"],
        ["assignee_triage_accuracy/phase7_rolling_open_set/reports/bmo_public_2021q3_untouched_holdout/INVALIDATED.md", "排除無效 Q3 結果的依據"],
        ["reports/health_check/health_check_latest.json", "既有整合健康檢查；本次另重跑 100 tests"],
    ]
    add_table(doc, ["資料／結果檔", "本報告引用用途"], data_rows, [5200, 4160], font_size=7.6, caption=(20, "實際引用的資料與結果檔案"))

    doc.add_heading("附錄 C：目前缺少或建議補做的測試", level=1)
    missing_rows = [
        ["精確 assignment-time 標籤", "缺少；last_change_time 只是保守代理", "高"],
        ["第二個真正 untouched 時間 holdout", "Q4 已看過，不可再用於調參", "高"],
        ["跨專案／跨組織 holdout", "尚未完成", "高"],
        ["經審核 active/departed roster", "公共 BMO 尚未提供", "高"],
        ["完整近重複與模板群組切分", "有 exact/部分近重複 audit，但非所有 Phase 7 window 完整群組化", "中高"],
        ["Weighted-F1、per-class precision/recall 全表", "主要 Phase 7 report 尚未輸出", "中"],
        ["完整全類別 confusion matrix artifact", "本報告由逐筆輸出衍生部分矩陣；專案未固定輸出", "中"],
        ["低信心閾值的成本敏感分析", "有 accuracy/coverage gate；尚缺人工作業成本與錯派成本", "中"],
        ["線上 A/B 或 shadow-mode 測試", "尚未完成；需目標專案與人工審核", "中"],
        ["LLM reranker 全量比較", "只有 3 筆 runtime smoke，無可報告 accuracy gain", "低於上述資料工作"],
    ]
    add_table(doc, ["測試項目", "目前狀態", "優先度"], missing_rows, [3500, 4660, 1200], font_size=8.0, caption=(21, "缺少、無法確認或建議補做的測試"))

    add_paragraph(doc, "文件結束。所有數字均取自上述專案檔案或本次實際測試輸出；找不到的指標已標示為目前專案尚未提供或尚未完成。", align=WD_ALIGN_PARAGRAPH.CENTER)

    # Prevent orphan title/table combinations where possible.
    for p in doc.paragraphs:
        if p.style and p.style.name.startswith("Heading"):
            set_keep_with_next(p)
    set_page_geometry(doc)
    doc.save(OUTPUT_DOCX)
    print(OUTPUT_DOCX)


if __name__ == "__main__":
    build_document()
