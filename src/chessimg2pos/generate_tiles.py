#!/usr/bin/env python3

# Generate tile images for all chessboard images in input folder
# Used for building training datasets

import math
import os
from glob import glob

import numpy as np
from .chessboard_image import get_chessboard_tiles


def _img_filename_prefix(chessboard_img_path):
    """
    Extracts the piece layout prefix from a filename like:
    'rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR.png'
    """
    filename = os.path.basename(chessboard_img_path)  # gets just the file name
    fen_like_part = filename[:-4]  # strip ".png"
    return fen_like_part  # will be split later into 8 parts using "-"


def _img_sub_dir(chessboard_img_path, tiles_dir):
    """The sub-directory where the chessboard tile images will be saved"""
    filename = os.path.basename(chessboard_img_path)
    sub_dir = filename[:-4]  # strip .png
    # print("subdir 1", sub_dir)
    # print("subdir 2", os.path.join(tiles_dir, sub_dir))
    return os.path.join(tiles_dir, sub_dir)


def save_tiles(tiles, chessboard_img_path, tiles_dir):
    """Saves all 64 tiles as 32x32 PNG files with this naming convention:

    a1_R.png (white rook on a1)
    d8_q.png (black queen on d8)
    c4_1.png (nothing on c4)
    """
    sub_dir = _img_sub_dir(chessboard_img_path, tiles_dir)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    img_save_dir = _img_sub_dir(chessboard_img_path, tiles_dir)
    # print("\tSaving tiles to {}\n".format(img_save_dir))
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
        
    piece_positions = _img_filename_prefix(chessboard_img_path).split("-")
    #`print("piece_positions", piece_positions)
    files = "abcdefgh"
    for i in range(64):
        piece = piece_positions[math.floor(i / 8)][i % 8]
        sqr_id = "{}{}".format(files[i % 8], 8 - math.floor(i / 8))
        tile_img_filename = "{}/{}_{}.png".format(img_save_dir, sqr_id, piece)
        tiles[i].save(tile_img_filename, format="PNG")


def generate_tiles_from_all_chessboards(chessboards_dir, tiles_dir, use_grayscale = True, overwrite = True):
    """Generates 32x32 PNGs for each square of all chessboards
    in chessboards_dir
    """
    if not os.path.exists(tiles_dir):
        os.makedirs(tiles_dir)

    chessboard_img_filenames = glob("{}/*.png".format(chessboards_dir))

    # chessboard_img_filenames = chessboard_img_filenames[:1]
    num_chessboards = len(chessboard_img_filenames)
    num_success = 0
    num_skipped = 0
    num_failed = 0
    for i, chessboard_img_path in enumerate(chessboard_img_filenames):
        # print("%3d/%d %s" % (i + 1, num_chessboards, chessboard_img_path))
        img_save_dir = _img_sub_dir(chessboard_img_path, tiles_dir)
        if os.path.exists(img_save_dir) and not overwrite:
            print("\tIgnoring existing {}\n".format(img_save_dir))
            num_skipped += 1
            continue
        tiles = get_chessboard_tiles(chessboard_img_path, use_grayscale=use_grayscale)
        if len(tiles) != 64:
            print("\t!! Expected 64 tiles. Got {}\n".format(len(tiles)))
            num_failed += 1
            continue
        save_tiles(tiles, chessboard_img_path, tiles_dir)
        num_success += 1
    print(
        "Processed {} chessboard images ({} generated, {} skipped, {} failed)".format(
            num_chessboards, num_success, num_skipped, num_failed
        )
    )
