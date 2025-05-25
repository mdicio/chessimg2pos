#!/usr/bin/env python3

import os
from glob import glob

def save_output_html(chessboards_dir, tile_dirs, output_file = "images.html"):
    html = '<html lang="en">'
    html += '<link rel="stylesheet" href="./web/style.css" />'
    for tile_dir in tile_dirs:
        img_dir = tile_dir.split("/")[-2]
        img_filename_prefix = tile_dir.split("/")[-1]
        chessboard_img_path = os.path.join(
            chessboards_dir, img_dir, "{}.png".format(img_filename_prefix)
        )
        html += "<h3>{}</h3>".format(chessboard_img_path)
        html += '<img src="{}" class="chessboard"/>'.format(chessboard_img_path)
        html += "<h3>{}</h3>".format(tile_dir)
        square_map = {}
        for tile_img_path in glob(os.path.join(tile_dir, "*.png")):
            square_id = tile_img_path[-8:-6]
            fen_char = tile_img_path[-5]
            square_map[square_id] = {
                "img_src": tile_img_path,
                "fen_char": fen_char,
            }
        for rank in [8, 7, 6, 5, 4, 3, 2, 1]:
            for file in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                square_id = "{}{}".format(file, rank)
                square = square_map[square_id]
                fen_char = square["fen_char"]
                html += '<img src="{}"/>'.format(square["img_src"])
                html += '<span class="fen-char {}">{}</span>'.format(
                    "empty" if fen_char is "1" else "", fen_char
                )
            html += "<br />"
        html += "<br />"
    html += "</html>"
    with open(output_file, "w") as f:
        f.write(html)

