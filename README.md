# 星点柔焦

星点柔焦的 Windows/macOS 本机引擎和 GitHub Pages 网页前端。支持相机 RAW、TIFF 和 JPG，使用 SEP 检测点源与测光，按逐星 RGB 径向 PSF 生成正圆光晕，输出 16 位 TIFF。当前版本见 `version.py`；Windows 构建脚本会在版本 ZIP 已存在时自动递增补丁号。

## 使用

1. Windows 解压对应版本 ZIP 后双击 `星点柔焦.exe`。macOS 解压对应架构 ZIP 后，在 Finder 中按住 Control 并点按“星点柔焦.app”，选择“打开”。
2. 程序先打开本机处理界面；点按标题栏的“网页版”可打开 GitHub Pages 前端。网页通过带随机令牌的本机回环连接运行图像处理，图片不会上传至 GitHub。
3. 选择 RAW、TIFF 或 JPG 文件。RAW 会读取可用镜头数据，并依焦距决定星点检测分辨率。
4. “柔焦星点亮度范围控制”设置相对星等差上限，默认 3.0 等，范围 0–10 等。数值越大，纳入的较暗星点越多；符合范围的点源全部处理，不限制数量。
5. 柔焦强度默认 10、范围 0–30。光晕不透明度独立控制，默认 30%，数值越高光晕越明显。
6. 完成后下载 16 位 TIFF。

## 星点和非星点筛选

检测使用 SEP 的局部背景与 RMS、PSF 匹配滤波和圆孔径测光。场景预览按逐行亮度跃变估计地平线，再用低通亮度抑制地景和树木剪影；检测背景和星点通量只统计天空遮罩内像素。点源形态允许圆度低至 0.35、长轴达到 `max(6 px, 5 × PSF σ)`（最高 12 px），避免把亮星、轻微拖线或像差星像误判成扩展目标；更大的弥散结构会被剔除。当地背景 RMS 高于天空全图 4 倍时，只有附近也检测到至少 4 个紧致点源的候选才保留；因此球状星团等拥挤星场中可分辨的成员星仍能柔化，弥散星云结构仍会被排除。此项是基于 SEP 形态与局部紧致源密度的简化拥挤场筛选，并非 DAOPHOT 的 PSF 拟合流程。DAOPHOT 文献讨论了密集星场中的重叠星像及逐星 PSF 测光问题。[Stetson 1987, DAOPHOT](https://articles.adsabs.harvard.edu/pdf/1987PASP...99..191S)

单张图像里，紧凑星系或树枝上的孤立灯点有时会与真实星点具有相近 PSF；仅凭像素无法保证区分所有此类目标。这里的球状星团支持指其图像中可分辨的紧致成员星，不把未分辨的整个团状光斑作为单颗星处理。控制值是 SEP 孔径测光得到的相对星等差上限 `Δm=-2.5 log10(F/Fmax)`；筛选条件为 `F/Fmax ≥ 10^(-0.4×Δm上限)`，不是 Gaia 目录的绝对星等。Gaia DR3 有 G、BP、RP 测光，`BP-RP` 是目录颜色指数；将其用于图像目标需要 WCS 天球坐标及星表匹配。Nova/Astrometry 服务未部署时，程序不冒称使用了星表星等；星色取自源图 RGB 孔径测光。[Gaia DR3 星表字段](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html)

## 光晕算法

每个候选先由 SEP 提供位置和孔径通量，再从原图 RGB 像素测量该星周围的圆对称径向 PSF：以同心圆环内像素的中位数作为各半径的星像剖面。生成的光晕只依赖欧氏半径，二维高斯核的横向和纵向 σ 相同；SEP 的长短轴和角度不会把光晕拉成椭圆。原图中的星核保留自身形状，新增的光晕始终是正圆。

柔焦以光学 PSF 卷积近似生成圆形散射翼：用横纵 σ 相同的正圆二维高斯核 `Gσ` 卷积测得的 RGB 径向 PSF。卷积剖面按历史版本的相对通量与星色重新匹配峰值，以便光晕亮度能达到原 1.4.4 的可见程度；叠加时只增加高于原图的部分，保留星点核心。最终结果为 `Iout = Iin + α × M(r) × max(B + A_i P_i - Iin, 0)`，其中 `P_i` 是逐星实测径向 PSF 经高斯卷积后的形状，`A_i` 是按星点相对通量、颜色和天光适配计算的振幅，`B` 是局部背景，`α` 是独立的光晕不透明度，`M(r)` 是只依赖欧氏半径的圆形羽化遮罩。该叠加会增加光晕亮度，不声称守恒积分光通量。

光晕扩散半径沿用 `18ffa10` 中基于相对星等的映射：

```text
q_i = ln(F_i / F_min) / ln(F_max / F_min)
R_i = r_min + (r_max - r_min) × q_i
A_i ∝ F_i / F_max
```

其中 `F_i` 是该星 SEP 圆孔径通量，`F_min` / `F_max` 是本次选中星点的最暗 / 最亮通量。最暗星落在最小半径，亮星随亮度逐渐扩大；光晕半径上下限可调。扩散核宽度按 `σ = (R_i/3) × sqrt(strength/40)` 计算，默认强度 10、最大 30；光晕不透明度直接控制正向增亮图层的混合量。亮度还按星点相对通量自动缩放，因此暗星获得更窄、更暗的光晕。因为输入 PSF 剖面按逐星 RGB 实测，输出保留星点颜色。每个光晕范围以正圆遮罩限定，边缘平滑融合到背景。

以卷积前后 PSF 形状确定目标光晕，再按目标峰值构成带原背景的 RGB 图层，只叠加目标高于原图的正差值。光晕不透明度直接控制这部分增加量；调到 100% 时，局部会达到完整的峰值匹配光晕效果。遮罩在圆形外缘使用 smoothstep 逐渐衰减，遮罩外像素不变；横纵扩散 σ 相等，遮罩只由欧氏距离决定，因此新增光晕保持正圆。各颜色通道分别使用原图径向剖面和实测星色。

全图天光使用带天空遮罩的 SEP `globalback` 估计。旧版把含暗地景的全图背景值与遮罩后的天空背景值比较，导致 `LZQ_9331.tif` 的倍率被错误顶到 1.5；现在参考值 `B_ref=0.17277` 用同一天空遮罩在 9331 样片上测得，倍率按 `clamp(B/B_ref, 0.70, 1.50)` 调整。这个适配以 Weber 对比度 `ΔL/L_background` 为依据，调节散射剖面的可见度；它不是光学散射定律。[SEP 全局背景估计](https://sep.readthedocs.io/en/v1.0.x/api/sep.Background.html)、[Weber 对比度定义](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019583/)

## 色彩和输出

- 主图像素以 `float32` 高精度缓冲完成检测所需图像读取、PSF 测量和柔焦，最后一步才量化为 16 位 TIFF；不会把主图像处理降成 8 位。JPG 或 8 位 TIFF 源本身只有 8 位信息，导出 16 位不会恢复已丢失的精度。
- TIFF/JPG 输出保留输入 RGB/灰度通道布局与 ICC 字节；无 ICC 输入仍保持无 ICC。sRGB 或未标记图像仅解码/重编码 sRGB 传递曲线以在线性光数值中计算，色度坐标与 ICC 不变；其他 ICC 图像保留原通道数值编码，不转换到 sRGB。支持 RGB 或灰度模式；CMYK、调色板和其他色彩模式需先转换。
- RAW 传感器数据没有已渲染 RGB 文件所带的输出 ICC；使用 rawpy / LibRaw 解马赛克并显影为 16 位线性 sRGB，之后在 float32 中处理并嵌入 sRGB ICC。若 RAW 同目录存在由 Adobe Camera Raw 写出的同名 sRGB TIFF，先测量两者中位亮度并以 TIFF 为显影曝光基准（TIFF 已反映 XMP 参数）；否则用内嵌 sRGB 预览和 XMP `Exposure2012` 曝光补偿校准。遇到其他预览/TIFF ICC 时不做 8 位色彩转换，改用可用的 XMP 曝光信息。没有可用同名 ACR 参考时，LibRaw 与 Adobe 的相机配置文件及色调曲线差异可能造成剩余差别。
- TIFF 的 Alpha 通道原样保留。
- 输出 TIFF 元数据记录星点亮度、颜色、检测信噪比、半径、天光估值和所用适配倍率。

## 从源码构建

Windows 构建需要 64 位 Python 3.12 和网络连接以安装依赖。双击 `build.ps1` 或在 PowerShell 执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build.ps1
```

生成物保存在 `releases\星点柔焦-v<版本>-win-x64.zip`，包含单文件 EXE、使用说明、许可证和版本号。若目标 ZIP 已存在，脚本自动递增补丁号后构建新 ZIP。

Windows 与 macOS 版本由 `.github/workflows/macos.yml` 在 GitHub Actions 上分别构建 ZIP，再统一发布到对应 GitHub Release。macOS 为 Apple 芯片与 Intel 分别构建 `.app`。当前 macOS 构建没有 Apple Developer ID 签名和公证；首次运行按 ZIP 内说明在 Finder 中选择“打开”。可在 macOS 上手动运行 `bash build_macos.sh`。

GitHub Pages 前端由 `.github/workflows/pages.yml` 自动构建和发布。网页需要已启动的本机引擎，以保留 RAW/TIFF/JPG 解码、SEP 检测和 16 位 TIFF 导出；直接打开网页而没有运行桌面程序时，会提示先启动本机引擎。

## 算法与开源组件

- [SEP 源提取、背景与孔径测光](https://sep.readthedocs.io/en/stable/tutorial.html)、[PSF 匹配滤波](https://sep.readthedocs.io/en/stable/filter.html)
- [SciPy 高斯滤波与二维卷积实现](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)、[二维高斯卷积核定义](https://docs.astropy.org/en/latest/api/astropy.convolution.Gaussian2DKernel.html)
- [Gaia DR3 星表数据模型](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html)
- [rawpy / LibRaw](https://github.com/LibRaw/LibRaw)、[tifffile](https://github.com/cgohlke/tifffile)、[PyInstaller](https://github.com/pyinstaller/pyinstaller)
