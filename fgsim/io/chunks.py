from pathlib import Path
from typing import List, Tuple

from fgsim.config import conf

ChunkType = List[Tuple[Path, int, int]]
chunk_size = conf.loader.chunk_size
batch_size = conf.loader.batch_size


def compute_chucks(files, len_dict) -> List[ChunkType]:
    chunk_coords: List[ChunkType] = [[]]
    ifile = 0
    ielement = 0
    current_chunck_elements = 0
    while ifile < len(files):
        elem_left_in_cur_file = len_dict[files[ifile]] - ielement
        elem_to_add = chunk_size - current_chunck_elements
        if elem_left_in_cur_file > elem_to_add:
            chunk_coords[-1].append(
                (files[ifile], ielement, ielement + elem_to_add)
            )
            ielement += elem_to_add
            current_chunck_elements += elem_to_add
        else:
            chunk_coords[-1].append(
                (files[ifile], ielement, ielement + elem_left_in_cur_file)
            )
            ielement = 0
            current_chunck_elements += elem_left_in_cur_file
            ifile += 1
        if current_chunck_elements == chunk_size:
            current_chunck_elements = 0
            chunk_coords.append([])

    # remove the last, uneven chunk
    chunk_coords = list(
        filter(
            lambda chunk: sum([part[2] - part[1] for part in chunk]) == chunk_size,
            chunk_coords,
        )
    )
    return chunk_coords
