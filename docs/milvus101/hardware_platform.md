# 硬件平台

- [实验一：百万向量检索（数据来自 SIFT1B ）](#----------------sift1b--)
- [实验二：亿级向量检索（数据来自 SIFT1B ）](#----------------sift1b--)
- [扩展实验：十亿向量检索（ SIFT1B ）](#----------------sift1b--)
- [边缘部署：ARM 平台](#-----arm-平台)

## 实验一：百万向量检索（数据来自 SIFT1B ）
经实测，以下硬件配置可顺利完成实验。

| Component           | Min Config                |
| ------------------ | -------------------------- |
| OS            | Ubuntu LTS 18.04 |
| CPU           | Intel Core i5-8250U           |
| GPU           | Nvidia GeForce MX150, 2GB GDDR5  |
| GPU Driver    | CUDA 10.1，Driver 418.74 |
| Memory        | 4 GB DDR4 ( 2400 Mhz ) x 2          |
| Storage       | NVMe SSD 256 GB             |

## 实验二：亿级向量检索（数据来自 SIFT1B ）
经实测，以下硬件配置可顺利完成实验。

| Component           | Min Config                |
| ------------------ | -------------------------- |
| OS            | Ubuntu LTS 18.04 |
| CPU           | Intel Core i7-7700K        |
| GPU           | Nvidia GeForce GTX 1050, 4GB GDDR5 |
| GPU Driver    | CUDA 10.1，Driver 418.74 |
| Memory        | 16 GB DDR4 ( 2400 Mhz ) x 2                |
| Storage       | SATA 3.0 SSD 256 GB                  |

## 扩展实验：十亿向量检索（ SIFT1B ）
经实测，以下硬件配置可顺利完成实验。

| Component           | Min Config                |
| ------------------ | -------------------------- |
| 操作系统           | CentOS 7.6               |
| CPU 配置           | Intel Xeon E5-2678 v3 @ 2.50GHz x 2   |
| GPU 配置           | Nvidia GeForce GTX 1080, 8GB GDDR5 x 2|
| GPU 驱动           | CUDA 10.1，Driver 418.74 |
| 内存型号           | 256 GB (实验中需消耗约 140 GB )    |
| 存储设备           | NVMe SSD 2 TB                       |

## 边缘部署：ARM 平台

| 系统组件           | 最低配置                   |
| ------------------ | ------------------------------- |
| OS           | Ubuntu LTS 18.04.2               |
| CPU           | Quad-core ARM Cortex-A57 @ 1.43GHz          |
| Memory           | 4GB 64-bit LPDDR4 @ 1600MHz ( 25.6 GB/s )   |
| Storage           | 16 GB eMMC 5.1 Flash |

经实测，以上硬件配置可顺利完成 微软 MS-Celeb-1M 百万人脸库检索（ 512维向量，FP32 ）检索。

**注意**： Arm 平台上的实验不包含在本在线训练营之中。如果想了解详情，请联系我们。
