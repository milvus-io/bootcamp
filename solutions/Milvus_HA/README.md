# README

## 环境准备

本次HA方案需要用到两台机器和一个共享存储设备。

主机：192.168.1.85

备机：192.168.1.38



## 安装milvus

在主备机上安装milvus server。主备机的milvus db目录均指向共享存储的位置。

安装方法：参考https://milvus.io/docs/zh-CN/userguide/install_milvus/

安装完成后，启动主机的milvus server,停止备机的server。

## 安装并配置keepalived

**查看主备机ip**

![1574995669134](pic\1574995669134.png)

![1574996203682](pic\1574996203682.png)

**修改系统网络配置**

在主备机环境中输入如下命令：

```bash
# vim /etc/sysctl.conf
```

将net.ipv4.ip_forward=1前的”#”号删除，将net.ipv4.ip_nonlocal_bind=1插入到其后，保存并退出。

输入命令使其生效：

```bash
# sysctl -p
```

**安装keepalived及其依赖**

在主备机中安装keepalived及其依赖包：

```bash
# apt-get install libssl-dev openssl libpopt-dev
# apt-get install keepalived
```

**配置keepalived**

给主备机配置keepalived。虚拟路由地址设置为

```bash
# vim /etc/keepalived/keepalived.conf

主机配置文件如下：
! Configuration File for keepalived
global_defs {
  router_id sol01 #主备机路由ID
}

vrrp_script chk_milvus {
       script "/etc/keepalived/chk_milvus.sh"   # 检查主机的milvus是否正常运行脚本
       interval 2
       weight -20
}

vrrp_instance VI_SERVER {
  state MASTER               # 主机服务器模式，备机设为BACKUP
  interface enp7s0             # 主机监控网卡实例
  virtual_router_id 51       # VRRP组名，主备机设置必须完全一致
  priority 110               # 优先级(1-254)，主机设置必须比备机高，备机可设为90
  authentication {           # 认证信息，主备机必须完全一致
    auth_type PASS
    auth_pass 1111
  }
  virtual_ipaddress {        # 虚拟IP地址，主备机必须完全一致
    192.168.1.104/24         # 注意配置子网掩码
  }
  track_script {
  chk_milvus
  }
}


备机配置文件如下：
! Configuration File for keepalived
global_defs {
  router_id sol02 #主备机路由ID
}

vrrp_instance VI_SERVER {
  state BACKUP               # 主机服务器模式，备机设为BACKUP
  interface enp3s0             # 主机监控网卡实例
  virtual_router_id 51       # VRRP组名，主备机设置必须完全一致
  priority 91               # 优先级(1-254)，主机设置必须比备机高，备机可设为90
  authentication {           # 认证信息，主备机必须完全一致
    auth_type PASS
    auth_pass 1111
  }
  virtual_ipaddress {        # 虚拟IP地址，主备机必须完全一致
    192.168.1.104/24         # 注意配置子网掩码
  }

  notify_master "/etc/keepalived/start_docker.sh master"
  notify_backup "/etc/keepalived/stop_docker.sh backup"
}

```

在主机/etc/keepalived目录下创建上述提到的chk_milvus.sh脚本,该脚本用于检测milvus server是否正常。

```bash
#!/bin/bash
RUNNING_STATUS=$(docker inspect --format '{{.State.Running}}' <docker id>)
if [[ "${RUNNING_STATUS}" != "true" ]];then
    exit 1
fi
# <docker id>表示主机的milvus server docker id
```

在备机/etc/keepalived目录下创建上述提到的start_docker.sh和stop_docker.sh脚本。start_docker.sh会在主机server异常断掉，虚拟地址指向备机后启动备机的milvus server；stop_docker.sh会在主机回复正常后，停止备机的milvus server。

start_docker.sh脚本：

```bash
#!/bin/bash
RUNNING_STATUS=$(docker inspect --format '{{.State.Running}}' <docker id>)
if [[ "${RUNNING_STATUS}" != "true" ]];then
docker start <docker id>
fi
```

该 <docker id>表示备机的milvus server docker id

stop_docker.sh脚本：

```bash
#!/bin/bash
RUNNING_STATUS=$(docker inspect --format '{{.State.Running}}' <docker id>)
if [[ "${RUNNING_STATUS}" = "true" ]];then
docker start <docker id>
fi
```

 该<docker id>表示备机的milvus server docker id

注：上述三个脚本创建后，需要增加其可执行权限。

```bash
chmod +x chk_milvus.sh
chmod +x start_docker.sh
chmod +x stop_docker.sh
```

**启动主备机的keepalived**

```bash
# service keepalived start
查看keepalived状态
# service keepalived status
```

**查看keepalived日志**

```bash
# cat /var/log/syslog | grep Keepalived | tail
```

## 验证

上述步骤安装完成后，在主机上输入命令ip a，可以看见虚拟ip地址192.168.1.104出现在了主机中。

![1575010275666](pic\1575010275666.png)

此时客户端，可通过地址192.168.1.104连接milvus server, 进行

关闭主机电源或停止主机的milvus server，再次用ip a查看，可以看见虚拟ip已从主机自动移除了。

![1575015544553](pic\1575015544553.png)

此时，在备机输入ip a，可以看见虚拟地址转移到了备机上。

![1575016581429](pic\1575016581429.png)

输入命令docker ps,可以看见备机中的docker 已经自动启动完成了。客户端通过192.168.1.104连接的milvus server实际已从主机转移到备机。由于主备机的milvus server共享一个存储设备，所以两边的milvus数据都是同步的。在主机出现异常的情况下，上述方案可保证客户在实际操作时，可以在秒级的时间server连接恢复正常。

重新恢复主机端milvus server后，虚拟地址会自动转移到主机上，此时客户端连接的server又变回主机server了。备机上的milvus server也将自动停止。