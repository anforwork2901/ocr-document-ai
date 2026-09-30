from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    output_dir = Path("samples/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "receipt_sample.png"

    width, height = 900, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 42)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    lines = [
        ("MINI MART AN PHU", title_font),
        ("Ngay ban: 12/08/2025", body_font),
        ("Sua tuoi Vinamilk 2 x 28000", body_font),
        ("Banh mi 1 x 15000", body_font),
        ("Nuoc suoi 3 x 7000", body_font),
        ("Tong cong: 92000 VND", body_font),
    ]

    y = 60
    for text, font in lines:
        draw.text((70, y), text, fill="black", font=font)
        y += 82

    draw.rectangle((38, 34, width - 38, height - 34), outline=(40, 40, 40), width=3)
    image.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()

