#!/usr/bin/env python3
"""MCP 诊断脚本 - 测试服务器功能"""

import asyncio
import sys
sys.path.insert(0, '/media/chenguisen/WD_BLACK/AISI/imagesimulation/abTEM/abtem-mcp-server/src')

from abtem_mcp_server.tools import AbTEMTools

async def diagnose():
    print("=" * 60)
    print("abTEM MCP 服务器诊断")
    print("=" * 60)
    
    # 测试 1: 获取材料列表
    print("\n1. 测试 get_supported_materials...")
    try:
        result = await AbTEMTools.get_supported_materials()
        if result.get("success"):
            materials = result.get("materials", {})
            print(f"   ✓ 成功 - 找到 {len(materials)} 种材料")
            print(f"   材料: {', '.join(materials.keys())}")
        else:
            print(f"   ✗ 失败 - {result.get('message')}")
    except Exception as e:
        print(f"   ✗ 异常 - {e}")
    
    # 测试 2: 创建晶体结构
    print("\n2. 测试 create_crystal_structure...")
    try:
        result = await AbTEMTools.create_crystal_structure(
            formula="Si",
            a=5.43,
            cache_key="test_si"
        )
        if result.get("success"):
            print(f"   ✓ 成功")
            print(f"   原子数: {result.get('num_atoms')}")
            print(f"   缓存键: {result.get('cache_key')}")
            print(f"   包含图像: {'image_base64' in result}")
        else:
            print(f"   ✗ 失败 - {result.get('message')}")
    except Exception as e:
        print(f"   ✗ 异常 - {e}")
    
    # 测试 3: 计算投影势
    print("\n3. 测试 calculate_projected_potential...")
    try:
        result = await AbTEMTools.calculate_projected_potential(
            formula="Si",
            a=5.43,
            cache_key="test_si",
            use_cache=True
        )
        if result.get("success"):
            print(f"   ✓ 成功")
            print(f"   厚度: {result.get('thickness'):.2f} Å")
            print(f"   使用缓存: {result.get('cache_used')}")
            print(f"   包含图像: {'image_base64' in result}")
        else:
            print(f"   ✗ 失败 - {result.get('message')}")
    except Exception as e:
        print(f"   ✗ 异常 - {e}")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(diagnose())
