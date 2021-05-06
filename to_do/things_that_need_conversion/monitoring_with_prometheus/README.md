# Prometheus and Grafana were used to monitor and alarm Milvus

Milvus uses Prometheus to monitor and store performance metrics, and uses open source timing data analysis and visualizable platform Grafana to display performance metrics.

## Solution overview

Milvus collects monitoring data and pushes it to Pushgateway.Milvus collects monitoring data and pushes it to Pushgateway.Meanwhile, Prometheus Server will pull data from Pushgateway and save it to its timing database (TSDB) on a regular basis. Prometheus Server will push the alarm information to Alertmanager when an alarm is generated. Grafana can be used to visualize the collected data.

![1](./001.png)

## Preparation

1、Prometheus

2、Alertmanager

3、Grafana

## How to build

Firstly, Prometheus is used to collect Milvus monitoring indicators, and how to connect Alertmanager to Prometheus to realize the visualization of data display and alarm mechanism.

##### Install the Prometheus

Download the  [Prometheus binary zip file](https://prometheus.io/download/).

```
tar xvfz prometheus-*.tar.gz
cd prometheus-*
```

##### Install Pushgateay

Download the  [Pushgateway binary zip file](https://prometheus.io/download/).

```
tar xvfz pushgateway-*.tar.gz
cd pushgateway-*
```

##### Start the Pushgateway

```
./pushgateway
```

<!--**The Pushgateway process must be started before starting Milvus Server.**-->

Turn on Prometheus monitor in **server_config.yaml** and set the address and port number of Pushgateway.

```
metric:
  enable: true       # Set the value to true to turn on Prometheus monitoring
  address: 127.0.0.1 # Set the IP address of Pushgateway
  port: 9091         # Set the port number of Pushgateway.
```

Download the Milvus Prometheus profile:

```
$ wget https://raw.githubusercontent.com/milvus-io/docs/master/v1.0.0/assets/monitoring/prometheus.yml \ -O prometheus.yml

```

Download Milvus alarm rules file to Prometheus root directory:

```
$ wget -P rules https://raw.githubusercontent.com/milvus-io/docs/master/v1.0.0/assets/monitoring/alert_rules.yml
```

Edit Prometheus configuration file according to actual requirements:

- Global: Configure parameters such as **Scrape_Interval** and **evaluation_interval**.

  ```
  global:
   scrape_interval:     2s # Set the fetch time interval to 2S 
   evaluation_interval: 2s # Set the evaluation interval to 2S
  ```

- Alerting: Set the address and port of the Alertmanager.

  ```
  alerting:
  alertmanagers:
  - static_configs:
    - targets: ['localhost:9093']
  ```

- Rule_files: Sets the alarm rule file.

  ```
  rule_files:
    - "alert_rules.yml"
  ```

- Scrape_configs: Sets information such as **Job_name** and **Targets** for fetching data.

  ```
  scrape_configs:
  - job_name: 'prometheus'
    static_configs:
    - targets: ['localhost:9090']
  
  - job_name: 'pushgateway'
    honor_labels: true
    static_configs:
    - targets: ['localhost:9091']
  ```

Start the Prometheus:

```
./prometheus --config.file=prometheus.yml
```

Login through the browser http://<The host providing the prometheus service>:9090，Go to the prometheus user interaction page.

![](./002.png)

##### Configuration Alertmanager

Alertmanager is primarily used to receive alarm messages sent by Prometheus. Here's the events that need to create alarm rules.

- Server down

  Alarm rule: Send an alarm when Milvus server goes down.

  How to tell: When Milvus servers go down, indicators on the monitoring dashboard show No Data.

- The CPU/GPU is too hot

  Alarm rule: Send alarm message when CPU/GPU temperature exceeds 80 ° C.

  How to judge: Check CPU Temperature and GPU Temperature on the monitoring dashboard.

Download the  [Alertmanager binary zip file](https://prometheus.io/download/)

```
tar xvfz Alertmanager-*.tar.gz
cd Alertmanager-*
```

Create the configuration file **alertManager.yml** based on the [configuration Alertmanager](https://prometheus.io/docs/alerting/latest/configuration/), specify the mailbox to which to receive alarm notifications, and add the configuration file to the root of the Alertmanager

Activate the Alertmanager service and specify the configuration file:

```
./alertmanager --config.file=alertmanager.yml
```



## Display of Milvus monitoring indicators using Grafana

- Running Grafana:

```
docker run -i -p 3000:3000 grafana/grafana
```

Open it in a browser http://<Host IP that provides Grafana services>:3000Url, and login to the Grafana User Interaction page.

![Grafana](./004.png)

<!--The default user name and password for Grafana is ADMIN.You can also create new Grafana accounts here.-->

- [Add Prometheus as the data source](https://grafana.com/docs/grafana/latest/features/datasources/add-a-data-source/)

From the Grafana User Interaction page, click **Configuration>Data Sources>Prometheus**, and set the following Data source properties:

![数据源配置](./005.png)

|  Field  | Definition                                                   |
| :-----: | ------------------------------------------------------------ |
|  Name   | Prometheus                                                   |
| Default | True                                                         |
|   URL   | *http://<Host IP providing the services of Prometheus>:9090* |
| Access  | Browser                                                      |

- Download the [Grafana configuration file](https://github.com/milvus-io/docs/blob/master/v1.0.0/assets/monitoring/dashboard.json)

- Import the configuration file into Grafana

  ![img](./008.png)

- Configure the monitoring metrics provided by Milvus through the **Grafana profile** provided by Milvus,The [Milvus monitoring metrics](https://milvus.io/cn/docs/v1.0.0/milvus_metrics.md) are shown below.

  ![](./006.png)
  
  ![](./007.png)
  
  **Milvus performance indicators**

| Indicators            | Instructions                                                 |
| --------------------- | ------------------------------------------------------------ |
| Insert per Second     | The number of vectors inserted per second                    |
| Queries per Minute    | The number of queries run per minute                         |
| Query Time per Vector | Single vector query time = query time/number of vectors      |
| Query Service Level   | Query service level = number of queries within a certain time threshold/total number of queries |
| Uptime                | How long the Milvus server is up (minutes)                   |

​      **System performance index**

| Indicators        | Instructions                                                 |
| ----------------- | ------------------------------------------------------------ |
| GPU Utilization   | GPU utilization rate (%)                                     |
| GPU Memory Usage  | Amount of display (GB) currently used by Milvus              |
| CPU Utilization   | CPU utilization (%) = server task execution time/server total elapsed time |
| Memory Usage      | Current amount of memory used by Milvus (GB)                 |
| Cache Utilization | Cache utilization (%)                                        |
| Network IO        | Read/write speed of network port (GB/s)                      |
| Disk Read Speed   | Disk read speed (GB/s)                                       |
| Disk Write Speed  | Disk write speed (GB/s).                                     |

​     **Hardware storage metrics**

| **Indicators** | **Instructions**                                |
| -------------- | ----------------------------------------------- |
| Data Size      | Total amount of data stored by Milvus (GB)      |
| Total File     | The total number of data files stored in Milvus |

