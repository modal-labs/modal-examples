"""
Inpainting removes unwanted parts of an image. The module has
inpainting functionality to remove the Pokémon name that appears on the 'base' card,
eg. Articuno, so that it can be replaced with a new, made up name for the model generated
Pokémon character.

This code is partly based on code from github.com/Sanster/lama-cleaner/.
"""
import io

import modal

cv_image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "opencv-python~=4.6.0.66",
            "Pillow~=9.3.0",
            "numpy~=1.23.5",
        ]
    )
    .run_commands(
        [
            "apt-get update",
            # Required to install libs such as libGL.so.1
            "apt-get install ffmpeg libsm6 libxext6 --yes",
        ]
    )
)

# Configs for opencv inpainting
# opencv document https://docs.opencv.org/4.6.0/d7/d8b/group__photo__inpaint.html#gga8002a65f5a3328fbf15df81b842d3c3ca05e763003a805e6c11c673a9f4ba7d07
cv2_flag: str = "INPAINT_NS"
cv2_radius: int = 4


# From lama-cleaner
def load_img(img_bytes, gray: bool = False):
    import cv2
    import numpy as np

    alpha_channel = None
    nparr = np.frombuffer(img_bytes, np.uint8)
    if gray:
        np_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    else:
        np_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if len(np_img.shape) == 3 and np_img.shape[2] == 4:
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGRA2RGB)
        else:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    return np_img, alpha_channel


def numpy_to_bytes(image_numpy, ext: str) -> bytes:
    import cv2

    data = cv2.imencode(
        f".{ext}",
        image_numpy,
        [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_PNG_COMPRESSION), 0],
    )[1]
    image_bytes = data.tobytes()
    return image_bytes


def new_pokemon_name(card_image: bytes, pokemon_name: str = "Randomon") -> bytes:
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    # 1. Paint out the existing name.

    flag_map = {"INPAINT_NS": cv2.INPAINT_NS, "INPAINT_TELEA": cv2.INPAINT_TELEA}
    img, alpha_channel = load_img(card_image)

    pokecard_name_top_left_crnr = (139, 43)
    pokecard_name_size = (225, 45)  # (width, height)

    mask_im = Image.new("L", img.shape[:2][::-1], 0)
    draw = ImageDraw.Draw(mask_im)
    (left, upper, right, lower) = (
        pokecard_name_top_left_crnr[0],
        pokecard_name_top_left_crnr[1],
        pokecard_name_top_left_crnr[0] + pokecard_name_size[0],
        pokecard_name_top_left_crnr[1] + pokecard_name_size[1],
    )
    draw.rectangle((left, upper, right, lower), fill=255)
    mask_im = mask_im.convert("L")
    mask_bytesio = io.BytesIO()
    mask_im.save(mask_bytesio, format="PNG")
    mask_img_bytes = mask_bytesio.getvalue()
    mask, _ = load_img(mask_img_bytes)

    assert img.shape[:2] == mask.shape[:2], "shapes of base image and mask must match"

    # "No GPU is required, and for simple backgrounds, the results may even be better than AI models."
    cur_res = cv2.inpaint(
        img[:, :, ::-1],
        mask[:, :, 0],  # Slicing ensures we get 1 channel not 3.
        inpaintRadius=cv2_radius,
        flags=flag_map[cv2_flag],
    )

    # 2. Insert the new name!

    # make a blank image for the text, initialized to transparent text color
    txt = Image.new("RGBA", cur_res.shape[:2][::-1], (255, 255, 255, 0))
    # Dejavu is only font installed on Debian-slim images.
    # TODO: Get the real Pokémon card font. (This Dejavu is pretty close though)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fnt = ImageFont.truetype(font_path, size=40)
    fnt.fontmode = "L"  # antialiasing
    # get a drawing context
    d = ImageDraw.Draw(txt)

    # draw text, full opacity
    # -3 is done to put text at right line height position
    text_position = (
        pokecard_name_top_left_crnr[0],
        pokecard_name_top_left_crnr[1] - 5,
    )
    # Note that the text is a *little* transparent. This looks closer to the original
    # text. Full opacity is too flat.
    d.text(text_position, pokemon_name, font=fnt, fill=(20, 20, 20, 230))

    # https://stackoverflow.com/a/45646235/4885590
    cur_res_correct_color = cv2.cvtColor(cur_res, cv2.COLOR_BGR2RGB)
    cur_res_image = Image.fromarray(cur_res_correct_color).convert("RGBA")
    out = Image.alpha_composite(cur_res_image, txt)

    img_bytesio = io.BytesIO()
    out.save(img_bytesio, format="PNG")
    return img_bytesio.getvalue()
