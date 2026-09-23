# 星点柔焦

Windows RAW 星点处理工具：使用 SEP 检测点源并测量亮度，按相对通量门限筛选后，为所有符合门限的星点分别生成 Gaussian 光晕，输出全分辨率 16 位 RGB TIFF。原始星像保留为清晰底图，光晕使用每颗星的 RGB 颜色和 Lighten 合成；每颗星的相对通量、颜色与半径写入 TIFF 元数据。

## 使用

1. 打开 `星点柔焦.exe`，程序会自动打开本机浏览器界面。页面右下方“退出程序”会结束处理服务。
2. 选择 RAW 文件并检查读取到的镜头、焦距和光圈信息。
3. 调整检测灵敏度、光晕强度与最小 / 最大半径。
4. 点击“开始星点柔焦”，完成后点“下载 16 位 TIFF”。

当前适配 LibRaw 支持的常见格式，包括 CR3、CR2、NEF、ARW、DNG、ORF、RW2、RAF、PEF、3FR、IIQ、KDC、MRW 和 X3F。具体机型支持情况取决于所带 LibRaw 版本。

## 算法

### 星点检测与测光

- 在 RAW 可见传感器区生成 Bayer 安全的 2×2 或更高倍率降采样单通道图。焦距参与大尺寸传感器的检测分辨率选择。
- SEP `Background` 估计空间变化的背景与噪声；先用初步星像测量估算 PSF 宽度，再以该宽度构造 Gaussian 检测核，交由 `sep.extract` 的 matched-filter 检测和去混叠步骤提取对象。
- 依据 SEP 二阶矩的椭圆大小、圆度和边缘标志筛出点源候选；`sep.sum_circle` 在局部背景扣除图上测量降采样 RAW 的相对圆孔径通量。所有达到相对亮度门限的候选都会处理，并按通量从高到低依次处理。该通量用于排序与效果控制，不是经标准星标定的天文星等。
- 相对亮度门限默认设为图中最亮候选通量的 6.3%，低于门限的候选不生成光晕。按标准对数星等定义，这一通量比约对应与最亮候选相差 3 个相对星等；没有星表匹配和零点定标，因此程序只显示相对通量百分比，不将其称为实际视星等。[AAVSO 测光指南](https://www.aavso.org/sites/default/files/CCDPhotometryGuide_3.pdf)
- SEP 按像素形态检测，不保证能在所有场景完美区分星点、热像素、地面灯光、星云和紧密星团。长拖线、饱和或严重像差的星像可能需要调节灵敏度。

### Gaussian 柔焦

成像系统的点扩散函数（PSF）描述理想点光源在图像上的响应；Gaussian 是常用近似，Moffat 函数通常更适合描述大气视宁度产生的宽翼。当前效果采用可控的 Gaussian bloom：

```text
F_i = 第 i 颗星的圆孔径通量；q_i = F_i / max(F)
保留条件：q_i ≥ 0.063（默认，画面内相对通量）
Δm_i = -2.5 log10(q_i)；Δm_max = -2.5 log10(0.063)
b_i = clamp(1 - Δm_i / Δm_max, 0, 1)
R_i = r_min + (r_max - r_min) × b_i
σ_i = R_i / 3
A_i[c] = min(星点局部 RGB 峰值[c], 星心像素[c])
G_i = A_i exp(-r_i² / (2σ_i²))
I_(k+1) = I_k + (strength / 100) × max(G_i - I_k, 0)  （逐颜色通道）
```

`A_i` 是每颗星在显影线性 RGB 图像中测得的局部三通道颜色峰值，并限制在原星心像素亮度以内；按红、绿、蓝通道分别生成光晕以保留星色。柔焦强度同时作为 Gaussian Lighten 层的不透明度，默认 10%，最大 30%。Gaussian 在中心最强，向外平滑递减，在设置半径（3σ）处约为峰值的 1.1%；每颗被选星的中心采样值在合成后恢复为 RAW 显影原值。该效果用于摄影柔焦，不是守恒光通量的科学 PSF 重建。

每颗被处理星的底图坐标、孔径通量、相对最亮候选的通量比例与相对星等差、由二阶矩换算的 Gaussian FWHM 近似值、线性 RGB 光晕峰值和光晕半径作为 `star_photometry_and_halo_parameters` 写入 TIFF 描述元数据，供后续核对。这里的相对星等差仅由相对通量比计算，不是目录星等。RAW 方向标记按 rawpy / LibRaw 的 flip 规则应用到检测图，使检测坐标和显影图坐标对齐。

## astrometry.net 与星表

astrometry.net 的 `solve-field` 用星点图样匹配索引星表，求出图像的天球坐标变换（WCS）；它可使用自带或 Source Extractor 得到的源位置，但 WCS 解算本身不负责完整逐星测光，也不保证识别出图像中的每颗星。该项目以 SEP 完成像素空间检测与测光，因此不要求把 RAW 上传给 Nova 服务，也不要求本机安装 astrometry.net。

若需要显示星名、用 Gaia 等目录交叉匹配，仍需先取得可靠 WCS，再将像素位置转换到天球坐标。当前版本不做该目录匹配。Nova 网站的本地部署还涉及 Django 前端、数据库、异步提交处理和 solve server；本程序没有发现可调用的本地部署，因此没有把未经验证的服务假设写进处理流程。

## 从源码构建

需要 Windows 64 位 Python 3.12 和网络连接以安装依赖。双击 `build.ps1` 或在 PowerShell 执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build.ps1
```

独立程序生成于 `dist\星点柔焦.exe`。

## 算法与开源组件参考

- [SEP 文档：背景、源提取和孔径测光](https://sep.readthedocs.io/en/stable/tutorial.html) 与 [API](https://sep.readthedocs.io/en/stable/reference.html)：`Background`、`extract`、`sum_circle`。
- [SEP matched filter 文档](https://sep.readthedocs.io/en/stable/filter.html)：检测点源时可用与 PSF 形状接近的核；噪声空间变化时 matched-filter 会逐像素处理误差。
- [AAVSO CCD 测光指南](https://www.aavso.org/sites/default/files/CCDPhotometryGuide_3.pdf)：星等差与通量比的对数关系，以及未定标仪器星等的零点限制。
- [Adobe Photoshop Lighten 混合模式](https://helpx.adobe.com/photoshop/desktop/repair-retouch/adjust-light-tone/blending-mode-descriptions.html)：逐颜色通道选择底图与混合层中较亮的值，用于保留亮星核心并显现更宽的 Gaussian 光晕层。
- [Trujillo et al., The effects of seeing on Sersic profiles II: The Moffat PSF](https://arxiv.org/abs/astro-ph/0109067)：讨论 Moffat 与 Gaussian PSF 近似的关系及大气视宁度下的 PSF。
- [Unreal Engine Bloom Convolution 文档](https://dev.epicgames.com/documentation/unreal-engine/bloom-in-unreal-engine)：用光学散射 / 衍射核与图像卷积生成 bloom，并说明添加式标准 bloom 与能量守恒卷积的差异。
- [Astrometry.net 程序说明](https://astrometry.net/doc/readme.html) 与 [Nova 服务部署说明](https://astrometry.net/doc/nova.html)：源位置表、`solve-field` 图样匹配及本地 Nova 服务组成。
- [rawpy API：方向 flip 参数](https://letmaik.github.io/rawpy/api/rawpy.Params.html) 与 [LibRaw](https://github.com/LibRaw/LibRaw)：RAW 解码和方向处理。
- [tifffile](https://github.com/cgohlke/tifffile)：写入 16 位 TIFF。
- [PyInstaller](https://github.com/pyinstaller/pyinstaller)：打包 Windows 可执行文件。
