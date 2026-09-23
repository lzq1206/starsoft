# 星点柔焦

Windows RAW 星点处理工具：使用 SEP 检测点源并测量亮度，为最亮的 50 颗星分别生成 Gaussian 光晕，输出全分辨率 16 位 RGB TIFF。原始星像保留为清晰底图，光晕叠加在周围；每颗星的测光值与半径写入 TIFF 元数据。

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
- 依据 SEP 二阶矩的椭圆大小、圆度和边缘标志筛出点源候选；`sep.sum_circle` 在局部背景扣除图上测量降采样 RAW 的相对圆孔径通量，并按通量从高到低选取最多 50 颗。该通量用于排序与效果控制，不是经标准星标定的天文星等。
- SEP 按像素形态检测，不保证能在所有场景完美区分星点、热像素、地面灯光、星云和紧密星团。长拖线、饱和或严重像差的星像可能需要调节灵敏度。

### Gaussian 柔焦

成像系统的点扩散函数（PSF）描述理想点光源在图像上的响应；Gaussian 是常用近似，Moffat 函数通常更适合描述大气视宁度产生的宽翼。当前效果采用可控的 Gaussian bloom：

```text
F_i = 第 i 颗星的圆孔径通量
R_i = r_min + (r_max - r_min) × clamp((log10(F_i) - P10) / (P99 - P10), 0, 1)^0.85
σ_i = R_i / 3
I_out = clamp(I_RAW + strength × Σ A_i exp(-r_i² / (2σ_i²)), 0, 1)
```

`A_i` 按该星局部线性 RGB 峰值和相对通量确定。Gaussian 在中心最强，向外平滑递减，在设置半径（3σ）处约为峰值的 1.1%。清晰底图保留，所以星核不会像直接模糊那样被抹开；叠加光晕会增加输出亮度，因此这是摄影观感效果，不是守恒光通量的科学 PSF 重建。

每颗被处理星的底图坐标、孔径通量、由二阶矩换算的 Gaussian FWHM 近似值和光晕半径作为 `star_photometry_and_halo_parameters` 写入 TIFF 描述元数据，供后续核对。RAW 方向标记按 rawpy / LibRaw 的 flip 规则应用到检测图，使检测坐标和显影图坐标对齐。

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
- [Trujillo et al., The effects of seeing on Sersic profiles II: The Moffat PSF](https://arxiv.org/abs/astro-ph/0109067)：讨论 Moffat 与 Gaussian PSF 近似的关系及大气视宁度下的 PSF。
- [Unreal Engine Bloom Convolution 文档](https://dev.epicgames.com/documentation/unreal-engine/bloom-in-unreal-engine)：用光学散射 / 衍射核与图像卷积生成 bloom，并说明添加式标准 bloom 与能量守恒卷积的差异。
- [Astrometry.net 程序说明](https://astrometry.net/doc/readme.html) 与 [Nova 服务部署说明](https://astrometry.net/doc/nova.html)：源位置表、`solve-field` 图样匹配及本地 Nova 服务组成。
- [rawpy API：方向 flip 参数](https://letmaik.github.io/rawpy/api/rawpy.Params.html) 与 [LibRaw](https://github.com/LibRaw/LibRaw)：RAW 解码和方向处理。
- [tifffile](https://github.com/cgohlke/tifffile)：写入 16 位 TIFF。
- [PyInstaller](https://github.com/pyinstaller/pyinstaller)：打包 Windows 可执行文件。
