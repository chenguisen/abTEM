"""
materials - 晶体材料数据接口。

提供从 Materials Project 数据库获取晶体结构的功能。
"""

from abtem.materials.mp_client import (
    fetch_structure,
    get_mp_api_key,
    get_structure_by_id,
    is_available,
    mp_structure_to_atoms,
    mp_structure_to_cif,
    search_materials,
)

__all__ = [
    "fetch_structure",
    "get_mp_api_key",
    "get_structure_by_id",
    "is_available",
    "mp_structure_to_atoms",
    "mp_structure_to_cif",
    "search_materials",
]
