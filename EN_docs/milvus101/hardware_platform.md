# Hardware Requirements

- [Lab Test 1: One Million Vector Search (Data source: SIFT1B)](#lab1)
- [Lab Test 2: 100-Million-Scale Vector Search (Data source: SIFT1B)](#lab2)
- [Extended Test: One Billion Vector Search (Data source: SIFT1B)](#labx)
- [Edge Deployment: ARM Platform](#arm)

<a name="lab1"></a>

## Test 1: One Million Vector Search (Data source: SIFT1B)

| Component           | Minimum Config                |
| ------------------ | -------------------------- |
| OS            | Ubuntu LTS 18.04 |
| CPU           | Intel Core i5-8250U           |
| GPU           | Nvidia GeForce MX150, 2GB GDDR5  |
| GPU Driver    | CUDA 10.1, Driver 418.74 |
| Memory        | 4 GB DDR4 ( 2400 Mhz ) x 2          |
| Storage       | NVMe SSD 256 GB             |

<a name="lab2"></a>

## Test 2: 100 Million Vector Search (Data source: SIFT1B)

About 15 GB memory is needed in this test.

| Component  | Minimum Config                     |
| ---------- | ---------------------------------- |
| OS         | Ubuntu LTS 18.04                   |
| CPU        | Intel Core i7-8700                |
| GPU        | Nvidia GeForce GTX 1060, 6GB GDDR5 |
| GPU Driver | CUDA 10.1, Driver 418.74           |
| Memory     | 16 GB DDR4 ( 2400 Mhz ) x 2        |
| Storage    | SATA 3.0 SSD 256 GB                |

<a name="labx"></a>

## Extended Test: One Billion Vector Search (Data source: SIFT1B)

About 140 GB memory is needed in this test.

| Component  | Minimum Config                         |
| ---------- | -------------------------------------- |
| OS         | CentOS 7.6                             |
| CPU        | Intel Xeon E5-2678 v3 @ 2.50GHz x 2    |
| GPU        | Nvidia GeForce GTX 1080, 8GB GDDR5 x 2 |
| GPU Driver | CUDA 10.1, Driver 418.74               |
| Memory     | 256 GB                                 |
| Storage    | NVMe SSD 2 TB                          |

<a name="arm"></a>

## Edge Deployment: ARM Platform

Edge deployment was tested on face search in a million-face library (512-dimensional vectors, FP32) 

| Component | Minimum Config                            |
| --------- | ----------------------------------------- |
| OS        | Ubuntu LTS 18.04.2                        |
| CPU       | Quad-core ARM Cortex-A57 @ 1.43GHz        |
| Memory    | 4GB 64-bit LPDDR4 @ 1600MHz ( 25.6 GB/s ) |
| Storage   | 16 GB eMMC 5.1 Flash                      |

> **Note**: The experiments on ARM platform is not included in this Boot Camp. To learn more details about it, please contact us. 
