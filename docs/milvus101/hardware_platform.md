# 硬件平台

- [实验一：百万向量检索（数据来自 SIFT1B ）](#lab1)
- [实验二：亿级向量检索（数据来自 SIFT1B ）](#lab2)
- [扩展实验：十亿向量检索（ SIFT1B ）](#labx)
- [边缘部署：ARM 平台](#arm)

<a name="lab1"></a>

## 实验一：百万向量检索（数据来自 SIFT1B ）
经实测，以下硬件配置可顺利完成实验。

| Component           | Minimum Config                |
| ------------------ | -------------------------- |
| OS            | Ubuntu LTS 18.04 |
| CPU           | Intel Core i5-8250U           |
| GPU           | Nvidia GeForce MX150, 2GB GDDR5  |
| GPU Driver    | CUDA 10.1, Driver 418.74 |
| Memory        | 4 GB DDR4 ( 2400 Mhz ) x 2          |
| Storage       | NVMe SSD 256 GB             |

<a name="lab2"></a>

## 实验二：亿级向量检索（数据来自 SIFT1B ）
经实测，以下硬件配置可顺利完成实验。实验中约需要消耗 15 GB 内存。

| Component           | Minimum Config                |
| ------------------ | -------------------------- |
| OS            | Ubuntu LTS 18.04 |
| CPU           | Intel Core i7-8700        |
| GPU           | Nvidia GeForce GTX 1060, 6GB GDDR5 |
| GPU Driver    | CUDA 10.1, Driver 418.74 |
| Memory        | 16 GB DDR4 ( 2400 Mhz ) x 2                |
| Storage       | SATA 3.0 SSD 256 GB                  |

<a name="labx"></a>
## 扩展实验：十亿向量检索（ SIFT1B ）
经实测，以下硬件配置可顺利完成实验。实验中约需消耗 140 GB 内存。

| Component           | Minimum Config                |
| ------------------ | -------------------------- |
| OS           | CentOS 7.6               |
| CPU          | Intel Xeon E5-2678 v3 @ 2.50GHz x 2   |
| GPU          | Nvidia GeForce GTX 1080, 8GB GDDR5 x 2|
| GPU Driver   | CUDA 10.1, Driver 418.74 |
| Memory       | 256 GB    |
| Storage      | NVMe SSD 2 TB                       |

<a name="arm"></a>
## 边缘部署：ARM 平台
经实测，以上硬件配置可顺利完成百万人脸向量库（ 512维向量，FP32 ）检索。

| Component           | Minimum Config                   |
| ------------------ | ------------------------------- |
| OS           | Ubuntu LTS 18.04.2               |
| CPU           | Quad-core ARM Cortex-A57 @ 1.43GHz          |
| Memory           | 4GB 64-bit LPDDR4 @ 1600MHz ( 25.6 GB/s )   |
| Storage           | 16 GB eMMC 5.1 Flash |


**注意**： Arm 平台上的实验不包含在本在线训练营之中。如果想了解详情，请联系我们。
