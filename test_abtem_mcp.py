#!/usr/bin/env python3
"""
测试abtem MCP服务器功能
"""

import subprocess
import time
import sys
import os

def test_mcp_server():
    """测试MCP服务器启动和基本功能"""
    
    print("=" * 60)
    print("测试 abtem MCP 服务器")
    print("=" * 60)
    
    # 检查Python环境
    print("\n1. 检查Python环境...")
    python_path = "/home/chenguisen/miniconda3/envs/py4dstem/bin/python"
    if not os.path.exists(python_path):
        print(f"错误: Python路径不存在: {python_path}")
        return False
    print(f"✓ Python路径: {python_path}")
    
    # 检查abtem-mcp-server模块
    print("\n2. 检查abtem-mcp-server模块...")
    try:
        result = subprocess.run(
            [python_path, "-c", "import sys; sys.path.insert(0, '/media/chenguisen/WD_BLACK/AISI/imagesimulation/abTEM/abtem-mcp-server/src'); from abtem_mcp_server.server import main; print('✓ 模块加载成功')"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"错误: {result.stderr}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False
    
    # 启动MCP服务器
    print("\n3. 启动MCP服务器...")
    server_process = subprocess.Popen(
        [
            python_path,
            "-m",
            "abtem_mcp_server.server"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": "/media/chenguisen/WD_BLACK/AISI/imagesimulation/abTEM/abtem-mcp-server/src"
        }
    )
    
    # 等待服务器启动
    time.sleep(3)
    
    # 检查服务器是否在运行
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        print(f"错误: 服务器已退出")
        print(f"标准输出: {stdout}")
        print(f"标准错误: {stderr}")
        return False
    
    print("✓ MCP服务器正在运行")
    
    # 获取进程ID
    pid = server_process.pid
    print(f"✓ 服务器PID: {pid}")
    
    # 测试服务器功能
    print("\n4. 测试服务器功能...")
    print("   请通过Continue插件测试以下命令:")
    print("   - '列出abTEM支持的材料'")
    print("   - '创建硅晶体结构'")
    print("   - '模拟硅的TEM图像'")
    
    # 保持服务器运行
    print("\n5. 服务器状态:")
    print(f"   PID: {pid}")
    print("   状态: 运行中")
    print("\n   按Ctrl+C停止服务器")
    
    try:
        # 等待用户中断
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n停止服务器...")
        server_process.terminate()
        server_process.wait()
        print("服务器已停止")
    
    return True

def check_continue_config():
    """检查Continue配置"""
    print("\n" + "=" * 60)
    print("检查 Continue 配置")
    print("=" * 60)
    
    config_path = os.path.expanduser("~/.continue/config.yaml")
    if os.path.exists(config_path):
        print(f"✓ Continue配置文件存在: {config_path}")
        
        # 读取配置
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # 检查关键配置
        checks = [
            ("models", "模型配置"),
            ("mcpServers", "MCP服务器配置"),
            ("capabilities:.*tool_use", "工具调用能力"),
            ("DeepSeek Chat", "DeepSeek模型"),
        ]
        
        for pattern, description in checks:
            if pattern in config_content:
                print(f"  ✓ {description}: 已配置")
            else:
                print(f"  ✗ {description}: 未找到")
        
        return True
    else:
        print(f"✗ Continue配置文件不存在: {config_path}")
        return False

def main():
    """主函数"""
    print("abtem MCP 环境测试")
    print("=" * 60)
    
    # 检查Continue配置
    if not check_continue_config():
        print("\n请先配置Continue插件")
        return
    
    # 测试MCP服务器
    print("\n" + "=" * 60)
    print("开始测试MCP服务器...")
    print("=" * 60)
    
    success = test_mcp_server()
    
    if success:
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        print("\n下一步:")
        print("1. 重启VS Code/Cursor")
        print("2. 打开Continue插件")
        print("3. 选择'Agent'模式")
        print("4. 输入: '列出abTEM支持的材料'")
        print("5. 等待DeepSeek调用MCP工具")
    else:
        print("\n" + "=" * 60)
        print("测试失败！")
        print("=" * 60)
        print("\n请检查:")
        print("1. Python环境是否正确")
        print("2. abtem-mcp-server是否已安装")
        print("3. 依赖包是否完整")

if __name__ == "__main__":
    main()