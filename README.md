# 星点柔焦

Windows 本地星点柔焦工具。支持相机 RAW、TIFF 和 JPG，使用 SEP 检测点源并测量亮度，按星点颜色生成正圆光晕，输出 16 位 TIFF。当前版本见 `version.py`；构建脚本会在版本 ZIP 已存在时自动递增补丁号。

## 使用

1. 解压对应版本的 ZIP，双击 `星点柔焦.exe`，程序会打开本机浏览器界面。
2. 选择 RAW、TIFF 或 JPG 文件。RAW 会读取可用镜头数据，并依焦距决定星点检测分辨率。
3. “柔焦星点亮度范围控制”设置相对星等差上限，默认 3.0 等，范围 0–10 等。数值越大，纳入的较暗星点越多；符合范围的点源全部处理，不限制数量。
4. 柔焦强度默认 10、范围 0–30；已压缩强度响应。光晕不透明度独立控制，默认 30%，数值越高光晕越明显。
5. 完成后下载 16 位 TIFF。

## 星点和非星点筛选

检测使用 SEP 的局部背景与 RMS、PSF 匹配滤波和圆孔径测光。场景预览按逐行亮度跃变估计地平线，再用低通亮度抑制地景和树木剪影；检测背景和星点通量只统计天空遮罩内像素。点源形态允许圆度低至 0.35、长轴达到 `max(6 px, 5 × PSF σ)`（最高 12 px），避免把亮星、轻微拖线或像差星像误判成扩展目标；更大的弥散结构会被剔除。当地背景 RMS 高于天空全图 4 倍时，只有附近也检测到至少 4 个紧致点源的候选才保留；因此球状星团等拥挤星场中可分辨的成员星仍能柔化，弥散星云结构仍会被排除。此项是基于 SEP 形态与局部紧致源密度的简化拥挤场筛选，并非 DAOPHOT 的 PSF 拟合流程。DAOPHOT 文献讨论了密集星场中的重叠星像及逐星 PSF 测光问题。[Stetson 1987, DAOPHOT](https://articles.adsabs.harvard.edu/pdf/1987PASP...99..191S)

单张图像里，紧凑星系或树枝上的孤立灯点有时会与真实星点具有相近 PSF；仅凭像素无法保证区分所有此类目标。这里的球状星团支持指其图像中可分辨的紧致成员星，不把未分辨的整个团状光斑作为单颗星处理。控制值是 SEP 孔径测光得到的相对星等差上限 `Δm=-2.5 log10(F/Fmax)`；筛选条件为 `F/Fmax ≥ 10^(-0.4×Δm上限)`，不是 Gaia 目录的绝对星等。Gaia DR3 有 G、BP、RP 测光，`BP-RP` 是目录颜色指数；将其用于图像目标需要 WCS 天球坐标及星表匹配。Nova/Astrometry 服务未部署时，程序不冒称使用了星表星等；星色取自源图 RGB 孔径测光。[Gaia DR3 星表字段](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html)

## 光晕算法

每个星点使用正圆高斯光晕 `I(r)=I0 × exp(-r²/(2σ²))`。光晕范围沿用 `18ffa10` 中基于相对星等的映射，并按测得通量进一步缩放强度：

```text
q_i = ln(F_i / F_min) / ln(F_max / F_min)
R_i = r_min + (r_max - r_min) × q_i
A_i ∝ F_i / F_max
```

其中 `F_i` 是该星 SEP 圆孔径通量，`F_min` / `F_max` 是本次选中星点的最暗 / 最亮通量。半径按相对星等的对数通量关系映射：最暗星落在最小半径，亮星依亮度逐渐扩大。光晕峰值还按 `F_i/F_max` 缩放，因此较暗星的光晕强度和范围都会减小；这是沿用用户认可的 `18ffa10` 映射并在此基础上增加测光强度缩放。半径上下限可调。生成的光晕是独立的正圆散射高斯，不跟随星像椭圆或镜头边缘畸变；原始星点核心保留不变。扩散宽度通过 `σ_diffusion = (R_i/3) × sqrt(strength/40)` 控制：默认 10、最大 30，整体尺度比旧版压缩；透明度只混合新增光晕。

在 `3σ` 到 `4σ` 的圆形边缘带使用 smoothstep 遮罩，从完整显示平滑衰减到透明。只将高斯目标亮度高于原像素的正向差值叠加后再乘圆形遮罩；遮罩外的矩形处理块像素保持不变，避免出现方形边框。每通道的光晕由源图 RGB 孔径测光颜色决定，颜色分布与实际星像核心颜色混合；源图 ICC 配置文件原样保留。

全图天光使用 SEP `globalback` 估计。为稳定不同背景亮度下光晕相对天光的对比度，新增光晕幅度按 `clamp(B/B_ref, 0.70, 1.50)` 调整，其中 `B_ref=0.05134` 是用户提供的 `LZQ_9331.tif` 在线性检测图上的基准值。该适配以 Weber 对比度 `ΔL/L_background` 为依据，倍率限制在 0.70–1.50；它不是光学散射定律。[SEP 全局背景估计](https://sep.readthedocs.io/en/v1.0.x/api/sep.Background.html)、[Weber 对比度定义](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019583/)

## 色彩和输出

- RAW 使用 rawpy / LibRaw 显影到线性 sRGB，并嵌入 sRGB ICC。若 RAW 同目录存在由 Adobe Camera Raw 写出的同名 TIFF，先测量两者中位亮度并以 TIFF 为显影曝光基准（TIFF 已反映 XMP 参数）；否则用相机内嵌预览和 XMP `Exposure2012` 曝光补偿校准。没有同名 ACR TIFF 时，LibRaw 与 Adobe 的相机配置文件及色调曲线差异可能造成剩余差别。
- TIFF/JPG 输入输出为 16 位 TIFF，保留输入 ICC；无 ICC 输入仍保持无 ICC。支持 RGB 或灰度模式；CMYK、调色板和其他色彩模式需先转换。
- sRGB 或未标记图像在柔焦时转换到线性光数值；其他 ICC 图像保留原通道编码，不做色彩空间转换。
- TIFF 的 Alpha 通道原样保留。JPEG 转 16 位不会恢复源文件丢失的细节。
- 输出 TIFF 元数据记录星点亮度、颜色、检测信噪比、半径、天光估值和所用适配倍率。

## 从源码构建

需要 Windows 64 位 Python 3.12 和网络连接以安装依赖。双击 `build.ps1` 或在 PowerShell 执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build.ps1
```

生成物保存在 `releases\星点柔焦-v<版本>-win-x64.zip`，包含单文件 EXE、使用说明、许可证和版本号。若目标 ZIP 已存在，脚本自动递增补丁号后构建新 ZIP。

## 算法与开源组件

- [SEP 源提取、背景与孔径测光](https://sep.readthedocs.io/en/stable/tutorial.html)、[PSF 匹配滤波](https://sep.readthedocs.io/en/stable/filter.html)
- [高斯二维卷积核](https://docs.astropy.org/en/latest/api/astropy.convolution.Gaussian2DKernel.html)
- [Gaia DR3 星表数据模型](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html)
- [rawpy / LibRaw](https://github.com/LibRaw/LibRaw)、[tifffile](https://github.com/cgohlke/tifffile)、[PyInstaller](https://github.com/pyinstaller/pyinstaller)
