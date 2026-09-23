# 星点柔焦

Windows 本地星点柔焦工具。支持相机 RAW、TIFF 和 JPG 输入，使用 SEP 检测点源并按圆孔径通量排序，为选中的星点生成具有原星色的扩展光晕，输出 16 位 TIFF。

## 使用

1. 打开 `星点柔焦.exe`，程序会自动打开本机浏览器界面。
2. 选择相机 RAW、TIFF 或 JPG 文件。RAW 可读取镜头信息并据焦距选择检测分辨率；TIFF/JPG 可读取其可用的相机和镜头元数据。
3. 设置“全局柔焦星点数量”：按 SEP 孔径通量从亮到暗选前 N 个，范围 0–500，默认 200。
4. 设置强度和最小 / 最大柔焦半径。默认强度 10，范围 0–30；强度控制高斯 PSF 扩散，不是图层不透明度。
5. 点击“开始星点柔焦”，完成后下载 16 位 TIFF。

RAW 支持情况取决于随程序打包的 LibRaw 版本。常见扩展名包括 CR3、CR2、NEF、ARW、DNG、ORF、RW2、RAF、PEF 等。栅格图像目前支持 RGB 或灰度 TIFF/JPG；CMYK、调色板和其他 TIFF/JPG 色彩模式需要先转换成 RGB 或灰度。

## 柔焦方法

相机把点光源成像为点扩散函数（PSF）。Gaussian PSF 是常用近似；两个 Gaussian 卷积后的协方差相加，因此对 SEP 测出的星像椭圆协方差 `Σ_star` 使用 Gaussian 扩散核后，有：

```text
Σ_out = Σ_star + σ_diffusion² I
```

程序使用 SEP 二阶矩 `a`、`b`、`theta` 构成星像椭圆，按该协方差生成扩展后的 Gaussian 轮廓。每个颜色通道用本地背景和星像实测峰值设定 Gaussian 翼的起始强度，再对原图采用 Lighten / max 合成，保留原始星心亮度，只在原星像不够亮的外侧显出扩散翼。这样保留核心亮度，因此不是守恒总光通量的模糊卷积；它对应摄影 bloom 中常见的非能量守恒光学散射效果。Gaussian 翼连续递减，约 3.75σ 后低于峰值的千分之一，并逐渐融入局部背景。[Gaussian 核的定义与归一化](https://docs.astropy.org/en/latest/api/astropy.convolution.Gaussian2DKernel.html)、[卷积 bloom 与光学散射](https://dev.epicgames.com/documentation/unreal-engine/bloom-in-unreal-engine)。

每颗星的扩散范围随其相对孔径通量非线性变化：

```text
q_i = 星点 SEP 孔径通量 / 最亮候选的 SEP 孔径通量
R_i = r_min + (r_max - r_min) × sqrt(q_i)
σ_diffusion = (R_i / 3) × sqrt(strength / 10)
```

平方根曲线是用于图像观感的亮度响应曲线，不是光学定律，也不等同于天文星等。SEP 通量用于按亮度排名和控制范围，不声称是经星表定标的真实星等。每通道独立生成光晕，因此保留红、绿、蓝星色。星像椭圆的方向和宽度也参与扩散核的计算。

默认最大半径 42 px、强度 10，是以用户提供的 `LZQ_9331.tif` 未柔焦图和 `LZQ_9332.tif`、`LZQ_9333.tif` 柔焦参考图进行视觉和星像宽度校准的起点。三张是分别拍摄的图像，星点位置、曝光和局部背景并非完全相同；该校准用于接近参考的光晕观感，不代表逐像素复原某个编辑参数。

SEP 使用局部背景 / 噪声估计、PSF 匹配滤波、点源形态筛选和圆孔径测光；没有固定 50 颗上限，实际处理数量由用户选择的 0–500 控制。为适配星密场景和 JPEG 压缩纹理，程序在需要时按检测图尺寸调高 SEP 的活动像素缓冲区。[SEP 点源提取和孔径测光](https://sep.readthedocs.io/en/stable/tutorial.html)、[SEP 匹配滤波](https://sep.readthedocs.io/en/stable/filter.html)、[SEP 缓冲区 API](https://sep.readthedocs.io/en/stable/api/sep.set_extract_pixstack.html)。PSF 在严重拖线、饱和、彗差、云雾或紧密星团下可能不符合 Gaussian 近似；Moffat 函数通常用于描述大气视宁度下的宽翼。[Trujillo 等人的 Moffat PSF 研究](https://arxiv.org/abs/astro-ph/0109067)

## 色彩和输出

- RAW 通过 rawpy / LibRaw 显影为线性 sRGB，输出嵌入 sRGB ICC。
- TIFF/JPG 输入输出均为 16 位 TIFF，并原样保留嵌入的 ICC 配置文件；无 ICC 的输入继续保持无 ICC。像素方向应用输入 EXIF / TIFF 方向后写成正常朝向。
- sRGB 或未标记的图像在柔焦时使用线性光数值；其他嵌入式 ICC 图像保留原通道编码与 ICC，不做色彩空间转换。
- TIFF/JPG 的星点柔焦只修改 RGB 或灰度通道。带 Alpha 的 RGB TIFF 保留 Alpha 通道。
- JPEG 输入原为 8 位，转存成 16 位 TIFF 不会恢复 JPEG 中已丢失的图像细节。

输出 TIFF 的元数据记录图像尺寸、检测到的候选数量、所选星点数量、SEP 通量比例、每颗星的半径和扩散参数，以及输入 ICC 名称。

## 从源码构建

需要 Windows 64 位 Python 3.12 和网络连接以安装依赖。双击 `build.ps1` 或在 PowerShell 执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build.ps1
```

独立程序生成于 `dist\星点柔焦.exe`。

## 算法与开源组件参考

- [SEP 文档：背景、源提取和孔径测光](https://sep.readthedocs.io/en/stable/tutorial.html) 与 [API](https://sep.readthedocs.io/en/stable/reference.html)
- [SEP 匹配滤波](https://sep.readthedocs.io/en/stable/filter.html)
- [Astropy Gaussian2DKernel](https://docs.astropy.org/en/latest/api/astropy.convolution.Gaussian2DKernel.html)
- [Unreal Engine Bloom 与卷积散射核](https://dev.epicgames.com/documentation/unreal-engine/bloom-in-unreal-engine)
- [Trujillo et al., The effects of seeing on Sersic profiles II: The Moffat PSF](https://arxiv.org/abs/astro-ph/0109067)
- [rawpy API](https://letmaik.github.io/rawpy/api/rawpy.Params.html) 与 [LibRaw](https://github.com/LibRaw/LibRaw)
- [tifffile](https://github.com/cgohlke/tifffile)
- [PyInstaller](https://github.com/pyinstaller/pyinstaller)
