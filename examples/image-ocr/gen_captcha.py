import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def random_char(length):
    # 随机选择字符
    characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # 生成指定长度的验证码
    code = ''.join(random.choice(characters) for i in range(length))
    return code


def generate_captcha():
    width, height = 160, 60
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 36)

    # 验证码图片
    code = random_char(4)
    for idx,t in enumerate(code):
        # "rtl", "ltr",
        draw.text((40 * idx + 10, 10), t, font=font, fill=random_color(),direction='ltr')

    # 生成干扰点
    for x in range(0,width,10):
        for y in range(0,height,10):
            draw.point((x, y), fill=(0,0,0))



    # 添加干扰线
    for i in range(3):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=2)
    # image = image.filter(ImageFilter.GaussianBlur(radius=2))
    image.save("captcha.jpg", "jpeg")

generate_captcha()
