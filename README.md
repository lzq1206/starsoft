# 星点柔焦

Windows 桌面工具：读取相机 RAW，识别点状星体，按每颗星的亮度调整柔焦半径，输出全分辨率 16 位 RGB TIFF。

## 使用

1. 打开 `星点柔焦.exe`，程序会自动打开本机浏览器界面。页面右下方“退出程序”会结束后台处理服务。
2. 选择 RAW 文件并检查读取到的相机、镜头、焦距和光圈信息。
3. 调整星点灵敏度、柔焦强度和最小 / 最大柔焦半径。
4. 点击“开始星点柔焦”，完成后点“下载 16 位 TIFF”。浏览器会按其下载设置保存文件。

当前适配 LibRaw 支持的常见格式，包括 CR3、CR2、NEF、ARW、DNG、ORF、RW2、RAF、PEF、3FR、IIQ、KDC、MRW 和 X3F。具体机型支持情况取决于所带 LibRaw 版本。

## 星点识别与柔焦

- 识别在降采样后的线性 RAW 传感器数据上进行，按局部背景与噪声估算阈值，并检查局部峰值、亮度支撑范围和形状圆度。
- RAW 元数据中的焦距用于调整检测图的尺寸与近邻抑制距离；镜头名、焦距和光圈会记录在 TIFF 描述信息中。rawpy 不提供所有 RAW 格式的机身型号字段，因此界面会把机身型号标为“未读取”。缺少焦距时会使用保守的默认识别比例。
- 柔焦在 RAW 解码后的线性 RGB 图上执行。星点亮度用局部通量衡量；通量越高，半径越接近设置的最大半径。最后转换为常见 sRGB 传递曲线并保存为 16 位 TIFF。
- 识别是星点形态启发式方法，严重拖线、云层、密集星团与热像素可能需要调整灵敏度；柔焦参数也会影响最终观感。

## 从源码构建

需要 Windows 64 位 Python 3.12 和网络连接以安装依赖。双击 `build.ps1` 或在 PowerShell 执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build.ps1
```

独立程序生成于 `dist\星点柔焦.exe`。

## 开源组件

- [rawpy](https://github.com/letmaik/rawpy) / [LibRaw](https://github.com/LibRaw/LibRaw)：RAW 解码与线性 RGB 输出。
- [Photutils DAOStarFinder](https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html)：局部峰值、尖锐度和圆度判别的算法参考。本项目以轻量 NumPy 实现相近的筛选步骤，没有复制 Photutils 代码。
- [tifffile](https://github.com/cgohlke/tifffile)：写入 16 位 TIFF。
- [PyInstaller](https://github.com/pyinstaller/pyinstaller)：打包 Windows 可执行文件。
