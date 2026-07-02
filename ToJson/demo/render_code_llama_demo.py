import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "demo" / "code_llama_to_json_input.jsonl"
OUTPUT_PATH = ROOT / "demo" / "code_llama_to_json_output.jsonl"
PNG_PATH = ROOT / "demo" / "code_llama_to_json_demo.png"

FONT_CJK = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_CJK_LIGHT = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def load_first_jsonl(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSONL records found: {path}")


def wrap_by_pixels(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue

        current = ""
        for word in paragraph.split():
            candidate = word if not current else f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font_obj)
            if bbox[2] - bbox[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def rounded_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], radius: int, fill: str, outline: str) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font_obj: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_height: int,
) -> int:
    x, y = xy
    for line in wrap_by_pixels(draw, text, font_obj, max_width):
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += line_height
    return y


def draw_code_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    code: str,
    font_obj: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_height: int,
) -> None:
    x, y = xy
    for raw_line in code.splitlines():
        wrapped = wrap_by_pixels(draw, raw_line, font_obj, max_width)
        if not wrapped:
            y += line_height
            continue
        for index, line in enumerate(wrapped):
            prefix = "" if index == 0 else "  "
            draw.text((x, y), f"{prefix}{line}", font=font_obj, fill=fill)
            y += line_height


def main() -> None:
    record = load_first_jsonl(INPUT_PATH)
    prediction = load_first_jsonl(OUTPUT_PATH)

    title = (record.get("meta") or {}).get("title", "")
    report = record.get("bug_report", "")
    pred_json = prediction.get("predicted_json", {})
    pretty_json = json.dumps(pred_json, ensure_ascii=False, indent=2)

    width, height = 1600, 900
    image = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(image)

    title_font = font(FONT_CJK, 42)
    subtitle_font = font(FONT_CJK_LIGHT, 23)
    heading_font = font(FONT_CJK, 27)
    body_font = font(FONT_CJK_LIGHT, 23)
    label_font = font(FONT_CJK, 18)
    code_font = font(FONT_MONO, 20)

    draw.text((58, 38), "Code Llama：使用者錯誤回報轉 JSON 欄位", font=title_font, fill="#1d2530")
    draw.text((58, 92), "Demo flow: bug_report → codellama:7b-instruct → predicted_json", font=subtitle_font, fill="#657181")
    draw.line((58, 132, 1542, 132), fill="#d8dee8", width=2)

    left = (58, 168, 745, 790)
    right = (855, 168, 1542, 790)
    rounded_rect(draw, left, 14, "#ffffff", "#d8dee8")
    rounded_rect(draw, right, 14, "#ffffff", "#d8dee8")

    draw.rounded_rectangle((left[0], left[1], left[2], left[1] + 76), radius=14, fill="#fff8e7", outline="#fff8e7")
    draw.rectangle((left[0], left[1] + 56, left[2], left[1] + 76), fill="#fff8e7")
    draw.rounded_rectangle((right[0], right[1], right[2], right[1] + 76), radius=14, fill="#eef7f1", outline="#eef7f1")
    draw.rectangle((right[0], right[1] + 56, right[2], right[1] + 76), fill="#eef7f1")

    draw.text((84, 190), "使用者錯誤回報文字", font=heading_font, fill="#1d2530")
    draw.text((881, 190), "Code Llama 輸出 JSON 欄位", font=heading_font, fill="#1d2530")
    draw.text((774, 418), "→", font=font(FONT_CJK, 62), fill="#2d6cdf")

    y = 272
    draw.text((84, y), "Title", font=label_font, fill="#657181")
    y += 32
    y = draw_wrapped(draw, (84, y), title, body_font, "#1d2530", 610, 34)
    y += 22
    draw.text((84, y), "Bug Report", font=label_font, fill="#657181")
    y += 34
    draw_wrapped(draw, (84, y), report, body_font, "#1d2530", 610, 34)

    code_box = (881, 272, 1516, 724)
    draw.rounded_rectangle(code_box, radius=10, fill="#142033", outline="#142033")
    draw_code_block(draw, (907, 298), pretty_json, code_font, "#e8edf7", 560, 33)

    draw.text((58, 824), "輸入檔：demo/code_llama_to_json_input.jsonl", font=subtitle_font, fill="#657181")
    draw.text((885, 824), "輸出檔：demo/code_llama_to_json_output.jsonl", font=subtitle_font, fill="#657181")

    image.save(PNG_PATH)
    print(f"Wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
