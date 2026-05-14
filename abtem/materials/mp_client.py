"""
Materials Project 数据客户端 - 从 Materials Project 数据库获取晶体结构。

通过 Materials Project 的 REST API 获取经过 DFT 计算验证的晶体结构，
确保原子位置、晶格参数和空间群信息的准确性。

依赖：
    - mp-api: Materials Project 官方 Python API
    - pymatgen: 结构转换工具
    - ase: 原子模拟环境

API Key 配置（优先级）:
    1. MP_API_KEY 环境变量
    2. Materials_Project_API_KEY 环境变量
    可在 https://materialsproject.org/api 申请免费 API Key。
"""

from typing import Any, Optional
import os

try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False

try:
    from pymatgen.io.ase import AseAtomsAdaptor
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


def get_mp_api_key() -> Optional[str]:
    """获取 Materials Project API Key。"""
    key = os.getenv("MP_API_KEY", "") or os.getenv("Materials_Project_API_KEY", "")
    return key if key else None


def is_available() -> bool:
    """检查 Materials Project 集成是否可用（依赖已安装且 API Key 已配置）。"""
    return MP_API_AVAILABLE and PYMATGEN_AVAILABLE and ASE_AVAILABLE and get_mp_api_key() is not None


def search_materials(
    formula: str,
    max_results: int = 10,
    crystal_system: Optional[str] = None,
    space_group_number: Optional[int] = None,
    fields: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    在 Materials Project 数据库中搜索材料。

    Args:
        formula: 化学式，如 'Si', 'GaAs', 'Al2CuLi'
        max_results: 最大返回结果数
        crystal_system: 晶系 (e.g., 'Cubic', 'Hexagonal', 'Triclinic')
        space_group_number: 空间群编号 (1-230)
        fields: 需要返回的额外字段列表

    Returns:
        {"success": True, "results": [...], "count": N} 或
        {"success": False, "error": "..."}
    """
    if not MP_API_AVAILABLE:
        return {"success": False, "error": "mp-api 未安装，请运行: pip install mp-api"}

    api_key = get_mp_api_key()
    if not api_key:
        return {
            "success": False,
            "error": "未配置 Materials Project API Key。"
                     "请在环境变量中设置 MP_API_KEY=your_key_here。"
                     "可在 https://materialsproject.org/api 获取免费 API Key。",
        }

    search_kwargs: dict[str, Any] = {"formula": formula}
    if crystal_system is not None:
        search_kwargs["crystal_system"] = crystal_system.title()
    if space_group_number is not None:
        search_kwargs["space_group_number"] = space_group_number

    default_fields = [
        "material_id",
        "formula_pretty",
        "structure",
        "symmetry",
        "density",
        "volume",
        "nsites",
        "energy_above_hull",
        "is_stable",
        "band_gap",
        "formation_energy_per_atom",
    ]

    try:
        with MPRester(api_key) as mpr:
            docs = mpr.summary.search(
                fields=default_fields + (fields or []),
                num_chunks=1,
                **search_kwargs,
            )

            if not docs:
                msg = f"在 Materials Project 中未找到 '{formula}' 相关材料。"
                if crystal_system:
                    msg += f" (晶系: {crystal_system})"
                return {"success": True, "message": msg, "results": [], "count": 0}

            # 按稳定性排序：稳定优先，然后按能量高低排序
            docs.sort(
                key=lambda x: (
                    not x.is_stable,
                    x.energy_above_hull if x.energy_above_hull is not None else float("inf"),
                )
            )
            docs = docs[:max_results]

            results = []
            for doc in docs:
                symmetry = doc.symmetry
                entry: dict[str, Any] = {
                    "material_id": str(doc.material_id),
                    "formula": doc.formula_pretty,
                    "nsites": doc.nsites,
                    "density": round(doc.density, 3) if doc.density else None,
                    "volume": round(doc.volume, 2) if doc.volume else None,
                    "energy_above_hull": round(doc.energy_above_hull, 4)
                    if doc.energy_above_hull is not None
                    else None,
                    "is_stable": doc.is_stable,
                    "band_gap": round(doc.band_gap, 3) if doc.band_gap is not None else None,
                    "formation_energy": round(doc.formation_energy_per_atom, 4)
                    if doc.formation_energy_per_atom is not None
                    else None,
                }

                if symmetry:
                    entry["crystal_system"] = str(symmetry.crystal_system) if hasattr(symmetry, "crystal_system") else None
                    entry["space_group"] = getattr(symmetry, "symbol", None)
                    entry["space_group_number"] = getattr(symmetry, "number", None)

                if doc.structure:
                    lattice = doc.structure.lattice
                    entry["lattice"] = {
                        "a": round(lattice.a, 4),
                        "b": round(lattice.b, 4),
                        "c": round(lattice.c, 4),
                        "alpha": round(lattice.alpha, 2),
                        "beta": round(lattice.beta, 2),
                        "gamma": round(lattice.gamma, 2),
                    }

                results.append(entry)

            return {
                "success": True,
                "message": f"找到 {len(results)} 个与 '{formula}' 相关的材料。",
                "results": results,
                "count": len(results),
            }

    except Exception as e:
        return {"success": False, "error": f"Materials Project 搜索失败: {str(e)}"}


def get_structure_by_id(
    material_id: str,
    conventional: bool = True,
) -> dict[str, Any]:
    """
    通过 Materials Project ID 获取晶体结构。

    Args:
        material_id: Materials Project ID，如 'mp-149' (Si)
        conventional: 是否返回常规晶胞（True）或原胞（False）

    Returns:
        {"success": True, "structure": pymatgen Structure, "info": {...}} 或
        {"success": False, "error": "..."}
    """
    if not MP_API_AVAILABLE:
        return {"success": False, "error": "mp-api 未安装"}

    api_key = get_mp_api_key()
    if not api_key:
        return {"success": False, "error": "未配置 MP_API_KEY"}

    try:
        with MPRester(api_key) as mpr:
            doc = mpr.summary.get_data_by_id(material_id)

            if doc is None:
                return {"success": False, "error": f"未找到材料 ID: {material_id}"}

            structure = doc.structure
            if structure is None:
                return {
                    "success": False,
                    "error": f"材料 {material_id} 没有可用的结构数据",
                }

            # 如需常规晶胞
            if conventional:
                try:
                    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

                    sga = SpacegroupAnalyzer(structure)
                    structure = sga.get_conventional_standard_structure()
                except Exception:
                    pass

            lattice = structure.lattice
            info: dict[str, Any] = {
                "material_id": str(doc.material_id),
                "formula": doc.formula_pretty,
                "lattice": {
                    "a": round(lattice.a, 4),
                    "b": round(lattice.b, 4),
                    "c": round(lattice.c, 4),
                    "alpha": round(lattice.alpha, 2),
                    "beta": round(lattice.beta, 2),
                    "gamma": round(lattice.gamma, 2),
                },
                "nsites": structure.num_sites,
                "elements": [str(e) for e in structure.composition.elements],
            }

            symmetry = doc.symmetry
            if symmetry:
                info["space_group"] = getattr(symmetry, "symbol", None)
                info["space_group_number"] = getattr(symmetry, "number", None)
                if hasattr(symmetry, "crystal_system"):
                    info["crystal_system"] = str(symmetry.crystal_system)

            return {"success": True, "structure": structure, "info": info}

    except Exception as e:
        return {"success": False, "error": f"获取结构失败: {str(e)}"}


def mp_structure_to_atoms(
    structure,
    supercell: tuple[int, int, int] = (1, 1, 1),
):
    """
    将 pymatgen Structure 转换为 ASE Atoms 对象。

    Args:
        structure: pymatgen Structure 对象
        supercell: 超胞大小 (nx, ny, nz)

    Returns:
        ASE Atoms 对象
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen 未安装，无法执行结构转换。")
    if not ASE_AVAILABLE:
        raise ImportError("ase 未安装，无法执行结构转换。")

    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)

    if supercell != (1, 1, 1):
        atoms = atoms * supercell

    return atoms


def mp_structure_to_cif(structure) -> Optional[str]:
    """
    将 pymatgen Structure 转换为 CIF 格式字符串。

    Args:
        structure: pymatgen Structure 对象

    Returns:
        CIF 格式字符串，失败时返回 None
    """
    try:
        from pymatgen.io.cif import CifWriter

        writer = CifWriter(structure)
        return str(writer)
    except Exception:
        # Fallback: 通过 ASE 生成 CIF
        try:
            import io
            from ase.io import write as ase_write

            atoms = mp_structure_to_atoms(structure)
            buf = io.BytesIO()
            ase_write(buf, atoms, format="cif")
            return buf.getvalue().decode("utf-8")
        except Exception:
            return None


def fetch_structure(
    formula: str,
    supercell: tuple[int, int, int] = (1, 1, 1),
    conventional: bool = True,
):
    """
    高便捷性接口：搜索 → 选最稳定结构 → 返回 ASE Atoms 对象。

    这是最常用的入口函数，适合快速从 Materials Project 获取结构用于模拟。

    Args:
        formula: 化学式，如 'Si', 'GaAs', 'Al2CuLi'
        supercell: 超胞大小 (nx, ny, nz)，默认 (1, 1, 1) 不扩展
        conventional: 是否使用常规晶胞

    Returns:
        ASE Atoms 对象

    Raises:
        RuntimeError: API 不可用、搜索失败或未找到结构
        ValueError: 搜索结果为空
    """
    if not is_available():
        raise RuntimeError(
            "Materials Project 不可用。请确保已安装 mp-api 和 pymatgen，"
            "并设置 MP_API_KEY 环境变量。"
        )

    result = search_materials(formula, max_results=1)
    if not result.get("success"):
        raise RuntimeError(f"搜索 '{formula}' 失败: {result.get('error')}")
    if not result.get("results"):
        raise ValueError(f"在 Materials Project 中未找到 '{formula}'。")

    best = result["results"][0]
    material_id = best["material_id"]

    struct_result = get_structure_by_id(material_id, conventional=conventional)
    if not struct_result.get("success"):
        raise RuntimeError(f"获取 {material_id} 结构失败: {struct_result.get('error')}")

    atoms = mp_structure_to_atoms(struct_result["structure"], supercell=supercell)

    print(f"✓ 从 Materials Project ({material_id}) 获取 {formula} 结构成功")
    print(f"  晶格: a={best['lattice']['a']:.4f}, b={best['lattice']['b']:.4f}, c={best['lattice']['c']:.4f}")
    print(f"  空间群: {best.get('space_group', 'N/A')} (#{best.get('space_group_number', 'N/A')})")

    return atoms
