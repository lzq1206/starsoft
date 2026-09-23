# 星点柔焦

Windows 本地星点柔焦工具。支持相机 RAW、TIFF 和 JPG，使用 SEP 检测点源并测量亮度，按星点颜色生成正圆光晕，输出 16 位 TIFF。当前版本见 `version.py`；构建脚本会在版本 ZIP 已存在时自动递增补丁号。

## 使用

1. 解压对应版本的 ZIP，双击 `星点柔焦.exe`，程序会打开本机浏览器界面。
2. 选择 RAW、TIFF 或 JPG 文件。RAW 会读取可用镜头数据，并依焦距决定星点检测分辨率。
3. “全局柔焦星点数量”按 SEP 圆孔径通量排序，默认 200，范围 0–500。
4. 柔焦强度默认 10、范围 0–30；已压缩强度响应。光晕透明度独立控制，默认 30%。
5. 完成后下载 16 位 TIFF。

## 星点和非星点筛选

检测使用 SEP 的局部背景与 RMS、PSF 匹配滤波和圆孔径测光。候选必须呈紧致、近圆的点扩散形态；超过估计 PSF 尺寸，或所在网格背景 RMS 高于全图 4 倍的目标会被剔除，以减少星云、星系和地景灯进入处理。星云和星系的扩展结构也不会作为单个星点柔化。

单张图像里，紧凑星系或地面远处灯点有时会与真实星点具有相近 PSF；不做天球坐标解算和星表匹配时，仅凭像素无法保证区分所有此类目标。Gaia DR3 有 G、BP、RP 测光，`BP-RP` 是目录颜色指数；要拿它对应影像中的星点，需要图像的 WCS 坐标和天球匹配。Nova/Astrometry 服务未部署时，程序以源图 RGB 孔径测光估计每颗星的颜色，记录各通道光通量占比，不把它称作 Gaia 颜色。[Gaia DR3 星表字段](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html)

## 光晕算法

高斯点扩散函数写作 `I(r)=I0 × exp(-r²/(2σ²))`。固定背景噪声等亮度线的半径满足 `r=σ × sqrt(2 ln(I0/Ithreshold))`；程序将该关系用于星点亮度到柔焦范围的分配：

```text
q_i = sqrt(ln(F_i / F_min) / ln(F_max / F_min))
R_i = r_min + (r_max - r_min) × q_i
```

其中 `F_i` 是该星 SEP 圆孔径通量，`F_min` / `F_max` 是本次选中星点的最暗 / 最亮通量。该映射来自高斯固定等亮度轮廓的 `r ∝ sqrt(ln(I0/Ithreshold))` 关系；通量半径严格单调，最暗星落在最小半径，亮星依亮度逐渐扩大。半径上下限可调。生成的光晕是独立的正圆散射高斯，不跟随星像椭圆或镜头边缘畸变；原始星点核心保留不变。扩散宽度通过 `σ_diffusion = (R_i/3) × sqrt(strength/40)` 控制：默认 10、最大 30，整体尺度比旧版压缩；透明度只混合新增光晕。

在 `3σ` 到 `4σ` 的圆形边缘带使用 smoothstep 遮罩，从完整显示平滑衰减到透明。每通道的光晕由源图 RGB 孔径测光颜色决定，颜色分布与实际星像核心颜色混合；源图 ICC 配置文件原样保留。

全图天光使用 SEP `globalback` 估计。为稳定不同背景亮度下光晕相对天光的对比度，新增光晕幅度按 `clamp(B/B_ref, 0.70, 1.50)` 调整，其中 `B_ref=0.05134` 是用户提供的 `LZQ_9331.tif` 在线性检测图上的基准值。该适配以 Weber 对比度 `ΔL/L_background` 为依据，倍率限制在 0.70–1.50；它不是光学散射定律。[SEP 全局背景估计](https://sep.readthedocs.io/en/v1.0.x/api/sep.Background.html)、[Weber 对比度定义](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019583/)

## 色彩和输出

- RAW 使用 rawpy / LibRaw 显影到线性 sRGB，并嵌入 sRGB ICC。
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
