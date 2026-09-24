# 星点柔焦

星点柔焦的 Windows/macOS 本机引擎和 GitHub Pages 网页前端。支持相机 RAW、TIFF 和 JPG，可切换“根据真实星表亮度”或“图像解析星点亮度”两种筛选方式。星表模式使用随包 ASTAP 本机 WCS 解算器与 Gaia 派生亮星索引；图像模式沿用 1.4.7 的 SEP 相对孔径测光。程序按逐星 RGB 径向 PSF 生成正圆光晕，输出 16 位 TIFF。Windows/macOS ZIP 均含板解算器与亮星索引，无需另装解算器或联网匹配。Windows 构建脚本会在版本 ZIP 已存在时自动递增补丁号。

## 使用

1. Windows 解压对应版本 ZIP 后双击 `星点柔焦.exe`。macOS 解压对应架构 ZIP 后，在 Finder 中按住 Control 并点按“星点柔焦.app”，选择“打开”。macOS 启动失败时会显示错误并写入 `~/Library/Logs/StarSoftFocus/startup.log`。
2. 程序先打开本机处理界面；点按标题栏的“网页版”可打开 GitHub Pages 前端。图像解码、星点检测与 ASTAP 板解算都在本机完成。
3. 选择 RAW、TIFF 或 JPG 文件。RAW 会读取可用镜头数据，并依焦距决定星点检测分辨率。
4. 选择亮度来源：“根据真实星表亮度”用本机 WCS 将星点与 W08 星表匹配，按 Gaia 派生 G 星等筛选；部分广角、严重畸变或星点较少的照片可能解算失败。“图像解析星点亮度”使用 1.4.7 的 SEP 相对圆孔径测光，不依赖板解算。亮度范围控制 Δm 默认 5.0 等，范围 0–10 等；数值越大，纳入的较暗点源越多，不限处理数量。星表模式使用的 W08 索引约完整至 G=8 等、精度为 0.1 等，不含 Gaia source_id 或 BP-RP。
5. 柔焦强度默认 10、范围 0–30。光晕不透明度独立控制，默认 30%，数值越高光晕越明显。
6. 完成后下载 16 位 TIFF。

## 星点和非星点筛选

检测使用 SEP 的局部背景与 RMS、PSF 匹配滤波和圆孔径测光。场景预览按逐行亮度跃变估计地平线，再用低通亮度抑制地景和树木剪影；检测背景和候选星点只统计天空遮罩内像素。点源形态允许圆度低至 0.35、长轴达到 max(6 px, 5 × PSF σ)（最高 12 px），避免把亮星、轻微拖线或像差星像误判为扩展目标；更大的弥散结构会被剔除。当地背景 RMS 高于天空全图 4 倍时，只有附近也检测到至少 4 个紧致点源的候选才保留；因此球状星团等拥挤星场中可分辨的成员星仍能柔化，弥散星云结构仍会被排除。此项是基于 SEP 形态与局部紧致源密度的简化拥挤场筛选，并非 DAOPHOT 的 PSF 拟合流程。[Stetson 1987, DAOPHOT](https://articles.adsabs.harvard.edu/pdf/1987PASP...99..191S)

真实星表模式使用随包 ASTAP 命令行程序与 D05/G05/W08 索引。RAW/TIFF/JPG 的全分辨率 16 位灰度副本只用于解算；广角图像使用更多解算星点和多个重叠视场分别拟合畸变，优先采用离图块边缘较远的 WCS。星点经 SEP 检出后与 W08 交叉匹配；程序还会把局部 WCS 映射回 W08 星表位置，并在预期位置附近以较低门限重新检查紧致点源，因此可以补回部分 SEP 主检测漏掉的亮星。补配前仍需通过本地图像信噪比和点源形态检查。匹配全程离线，不会发送图像或天球坐标。W08 约完整至 G=8 等、精度 0.1 等，没有 Gaia source_id 或 BP-RP；光晕颜色按图像内逐星 RGB 取样。星表模式在相机焦距/画幅信息不足或板解算失败时会提示失败；可切换到图像亮度模式继续处理。相机 EXIF 不提供 35 mm 等效焦距时，只能依据程序内已识别的相机型号估算画幅。

星表模式的控制值为本地 W08 G 星等相对于图内最亮匹配或位置补配星的差值 `ΔG=G−G最亮`。图像模式的控制值为 SEP 圆孔径通量得到的 `Δm=-2.5 log10(F/Fmax)`，完全沿用 1.4.7 的图像测光筛选和半径响应，不调用板解算或星表。

## 光晕算法

每个候选先由 SEP 提供位置和孔径通量，再从原图 RGB 像素测量该星周围的圆对称径向 PSF：以同心圆环内像素的中位数作为各半径的星像剖面。生成的光晕只依赖欧氏半径，二维高斯核的横向和纵向 σ 相同；SEP 的长短轴和角度不会把光晕拉成椭圆。原图中的星核保留自身形状，新增的光晕始终是正圆。

柔焦以光学 PSF 卷积近似生成圆形散射翼：用横纵 σ 相同的正圆二维高斯核 `Gσ` 卷积测得的 RGB 径向 PSF。卷积剖面按历史版本的相对通量与星色重新匹配峰值，以便光晕亮度能达到原 1.4.4 的可见程度；叠加时只增加高于原图的部分，保留星点核心。最终结果为 `Iout = Iin + α × M(r) × max(B + A_i P_i - Iin, 0)`，其中 `P_i` 是逐星实测径向 PSF 经高斯卷积后的形状，`A_i` 是按星点相对通量、颜色和天光适配计算的振幅，`B` 是局部背景，`α` 是独立的光晕不透明度，`M(r)` 是只依赖欧氏半径的圆形羽化遮罩。该叠加会增加光晕亮度，不声称守恒积分光通量。

光晕扩散半径沿用 `18ffa10` 中基于相对星等的映射：

```text
F_i = 10^(-0.4 × (G_i - G_brightest))
q_i = ln(F_i / F_faintest) / ln(F_brightest / F_faintest)
R_i = r_min + (r_max - r_min) × q_i
A_i ∝ F_i / F_max
```

其中 `G_i` 是随包 W08 索引提供的 Gaia 派生 G 星等（0.1 等精度），`F_i` 为相对目录通量；`F_faintest` / `F_brightest` 是本次选中星点中的最暗 / 最亮相对目录通量。最暗星落在最小半径，亮星随目录星等非线性地扩大；光晕半径上下限可调。扩散核宽度按 `σ = (R_i/3) × sqrt(strength/40)` 计算，默认强度 10、最大 30；光晕不透明度直接控制正向增亮图层的混合量。亮度还按目录相对通量自动缩放，因此暗星获得更窄、更暗的光晕。因为输入 PSF 剖面按逐星 RGB 实测，输出保留星点颜色。每个光晕范围以正圆遮罩限定，边缘平滑融合到背景。

以卷积前后 PSF 形状确定目标光晕，再按目标峰值构成带原背景的 RGB 图层，只叠加目标高于原图的正差值。光晕不透明度直接控制这部分增加量；调到 100% 时，局部会达到完整的峰值匹配光晕效果。遮罩在圆形外缘使用 smoothstep 逐渐衰减，遮罩外像素不变；横纵扩散 σ 相等，遮罩只由欧氏距离决定，因此新增光晕保持正圆。各颜色通道分别使用原图径向剖面和实测星色。

全图天光使用带天空遮罩的 SEP `globalback` 估计。旧版把含暗地景的全图背景值与遮罩后的天空背景值比较，导致 `LZQ_9331.tif` 的倍率被错误顶到 1.5；现在参考值 `B_ref=0.17277` 用同一天空遮罩在 9331 样片上测得，倍率按 `clamp(B/B_ref, 0.70, 1.50)` 调整。这个适配以 Weber 对比度 `ΔL/L_background` 为依据，调节散射剖面的可见度；它不是光学散射定律。[SEP 全局背景估计](https://sep.readthedocs.io/en/v1.0.x/api/sep.Background.html)、[Weber 对比度定义](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019583/)

## 色彩和输出

- 主图像素以 `float32` 高精度缓冲完成检测所需图像读取、PSF 测量和柔焦，最后一步才量化为 16 位 TIFF；不会把主图像处理降成 8 位。JPG 或 8 位 TIFF 源本身只有 8 位信息，导出 16 位不会恢复已丢失的精度。
- TIFF/JPG 输出保留输入 RGB/灰度通道布局与 ICC 字节；无 ICC 输入仍保持无 ICC。sRGB 或未标记图像仅解码/重编码 sRGB 传递曲线以在线性光数值中计算，色度坐标与 ICC 不变；其他 ICC 图像保留原通道数值编码，不转换到 sRGB。支持 RGB 或灰度模式；CMYK、调色板和其他色彩模式需先转换。
- RAW 传感器数据没有已渲染 RGB 文件所带的输出 ICC；使用 rawpy / LibRaw 解马赛克并显影为 16 位线性 sRGB，之后在 float32 中处理并嵌入 sRGB ICC。若 RAW 同目录存在由 Adobe Camera Raw 写出的同名 sRGB TIFF，先测量两者中位亮度并以 TIFF 为显影曝光基准（TIFF 已反映 XMP 参数）；否则用内嵌 sRGB 预览和 XMP `Exposure2012` 曝光补偿校准。遇到其他预览/TIFF ICC 时不做 8 位色彩转换，改用可用的 XMP 曝光信息。没有可用同名 ACR 参考时，LibRaw 与 Adobe 的相机配置文件及色调曲线差异可能造成剩余差别。
- TIFF 的 Alpha 通道原样保留。
- 输出 TIFF 元数据记录星点亮度、颜色、检测信噪比、半径、天光估值和所用适配倍率。

## 从源码构建

Windows 构建需要 64 位 Python 3.12 和网络连接，用于安装依赖并下载 ASTAP 程序及 D05/G05/W08 星表索引。构建 ZIP 已包含这些组件，用户无需单独安装解算器。双击 `build.ps1` 或在 PowerShell 执行：

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
- [ASTAP 命令行板解算器与 W08 星表说明](https://www.hnsky.org/astap.htm)；程序包内附完整 ASTAP MPL-2.0 许可证、上游源码链接、Gaia/ESA/DPAC 归属与数据库 acknowledgement。
- [ASTAP 四星四边形识别说明](https://www.hnsky.org/astap_astrometric_solving.htm)、[Astrometry.net 解算尺度约束与降采样文档](https://astrometrynet.readthedocs.io/en/latest/readme.html)
- [rawpy / LibRaw](https://github.com/LibRaw/LibRaw)、[tifffile](https://github.com/cgohlke/tifffile)、[PyInstaller](https://github.com/pyinstaller/pyinstaller)

### 广角星空板解算

ASTAP 通过四星几何组合匹配图像与本地星表。广角图按天空掩码、SEP 星点数量和离画面中心的距离选择分块，并额外加入一个约占原图 42% 宽高、位于光轴中心的低畸变图块。多个图块最多并行运行 4 个 ASTAP 进程；每个进程使用独立目录和输出名。程序先用 ASTAP 的 `auto` 模式和常用的 500 个候选星搜索，只有快速搜索没有可信结果时，才对最优图块补一次 `slow` 重叠搜索。邻接图块中心落入已解算图块时，可把该位置的 WCS 坐标作为局部搜索种子。

ASTAP 的 `.wcs` 输出是 FITS WCS 头；FITS 像素 y 轴从底部计数，TIFF/SEP 像素行从顶部计数。程序在图块 WCS 正反变换、跨块比较和 Gaia 星表投影时统一转换 `y_wcs = 图块高度 - 1 - y_top`。省略此转换会让解算器成功、但把星表位置投到图像的上下镜像位置，导致跨块解互相否定或匹配不上检测到的星点。

宽场 WCS 通过四边形匹配数、重叠区坐标一致性和后续 Gaia 源点回投做校验：单块至少要有 6 个匹配四边形且匹配比例达到一半；较弱结果须与另一块独立解算的重叠区域一致。ASTAP 手册将 `slow` 描述为更大的搜索重叠，并把 500 列为常用星点数；四边形构造与图块一致性门槛的依据见上游算法说明，门槛数值是本程序的结果筛选策略。

`LZQ_6627.NEF` 用修正后的完整 RAW 管线复核：SEP 检出 3,062 个点源候选，WCS 与 W08 取得 296 个位置匹配，ΔG=5 时选中 180 颗星，生成 16 位 TIFF。旧实现拒绝了同图块之间的正确解，也没能把星表位置匹配回星像；检查发现 ASTAP 的 FITS 行坐标与 TIFF/SEP 行坐标方向相反。中央图块单独验证时，按 FITS 像素方向投影能对应 12 颗 G≤5 亮星。

#### 广角算法资料及适用范围

- [PixInsight StarAlignment 广角畸变校正说明](https://www.pixinsight.com/tutorials/sa-distortion/index.html)：以四边形/多边形几何描述减少错误匹配，再用预测校正、薄板样条和 RANSAC 逐步拟合局部畸变。文档讨论的是多图相对配准，可供重叠图块的局部一致性设计参考。
- [Sequator 手册](https://sites.google.com/view/sequator/manual)：`Complex` 畸变模式面向超广角、混合或不规则畸变，并指出需要足够可见星点；这是多帧星点对齐流程。
- [LoveDaisy/star_alignment](https://github.com/LoveDaisy/star_alignment)：源码把检测星点换算到球面坐标，按邻星角度、距离和亮度构造星群描述，再以 RANSAC 求图像变换。它用于多张星空照片对齐叠加，可借鉴局部星群特征与异常匹配剔除。
- [Astroalign 文档](https://astroalign.quatrope.org/en/v2.2/)：使用三点星群不变量和 RANSAC 求仿射变换，目标也是无 WCS 的多图相对配准。
- [astrometry.net `solve-field` 文档](https://github.com/dstndstn/astrometry.net/blob/main/man/solve-field.1)：公开了视场尺度边界、四边形尺寸范围、搜索深度与降采样选项；准确视场先验可减少要搜索的索引范围。
- [ASTAP 命令行与星表手册](https://www.hnsky.org/astap.htm)：W08 用于约 20°–90° 的广角，`slow` 模式加大星表搜索重叠区域。当前程序以本机 W08 做绝对星表解算，再用局部图块和 Gaia 位置匹配确认结果。
