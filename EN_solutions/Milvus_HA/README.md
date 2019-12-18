# Milvus High Availability (HA) Solution

## Environment settings

Two servers and one shared storage device.

Master server: 192.168.1.85

Backup server: 192.168.1.38



## Install Milvus

Install Milvus on the master and backup server. The `db` folder must direct to the shared storage device.

Refer to [https://milvus.io/docs/en/userguide/install_milvus/](https://milvus.io/docs/en/userguide/install_milvus/) to learn how to install Milvus.

After installation, launch the Milvus server on the master server and stop the Milvus server on the backup server.

## Install and configure keepalived

**Check master/backup IP**

![1574995669134](pic/1574995669134.png)

![1574996203682](pic/1574996203682.png)

**Configure system network settings**

Enter the following command in the master and backup servers.

```bash
# vim /etc/sysctl.conf
```

Remove `#` before `net.ipv4.ip_forward=1` and add `net.ipv4.ip_nonlocal_bind=1` after `net.ipv4.ip_forward=1`. Save and quit Vim.

Use the following command to implement the config file.

```bash
# sysctl -p
```

**Install keepalived and related dependencies**

Install keepalived and dependencies on the master and backup servers.

```bash
# apt-get install libssl-dev openssl libpopt-dev
# apt-get install keepalived
```

**Configure keepalived**

Configure keepalived for the master and backup servers. Use the following command to create a config file to configure the virtual router address:

```bash
# vim /etc/keepalived/keepalived.conf
```

In the master server, configure the `keepalived.conf` as follows:

```yaml
! Configuration File for keepalived
global_defs {
  router_id sol01 # Router ID for the master server
}

vrrp_script chk_milvus {
       script "/etc/keepalived/chk_milvus.sh"   # Check whether the Milvus on the master is running correctly
       interval 2
       weight -20
}

vrrp_instance VI_SERVER {
  state MASTER               # Server mode
  interface enp7s0             # Instance of network card monitoring for the master server
  virtual_router_id 51       # The master and backup must have the same VRRP group name.
  priority 110               # Priority (1-254). The master server must have higher priority than the backup server. You can specify 90 for the backup server.
  authentication {           # The master and backup must have the same authentication information.
    auth_type PASS
    auth_pass 1111
  }
  virtual_ipaddress {        # Virtual IP address. The master and backup must have the same virtual IP address.
    192.168.1.104/24         # Subnet mask
  }
  track_script {
  chk_milvus
  }
}
```

In the backup server, configure the `keepalived.conf` as follows:

```yaml
! Configuration File for keepalived
global_defs {
  router_id sol02 # Router ID for the backup server
}

vrrp_instance VI_SERVER {
  state BACKUP               # Server mode
  interface enp3s0             # Instance of network card monitoring for the backup server.
  virtual_router_id 51       # The master and backup must have the same VRRP group name.
  priority 91               # Priority (1-254). The master server must have higher priority than the backup server. You can specify 90 for the backup server.
  authentication {           # The master and backup must have the same authentication information.
    auth_type PASS
    auth_pass 1111
  }
  virtual_ipaddress {        # Virtual IP address. The master and backup must have the same virtual IP address.
    192.168.1.104/24         # Subnet mask
  }

  notify_master "/etc/keepalived/start_docker.sh master"
  notify_backup "/etc/keepalived/stop_docker.sh backup"
}

```

Create `chk_milvus.sh` in `/etc/keepalived` of the master server to check whether the Milvus server runs correctly. Use the following code for `chk_milvus.sh`.

```bash
#!/bin/bash
RUNNING_STATUS=$(docker inspect --format '{{.State.Running}}' <docker id>)
if [[ "${RUNNING_STATUS}" != "true" ]];then
    exit 1
fi
# <docker id>specifies the milvus server docker id of the master server
```

Create `start_docker.sh` and `stop_docker.sh` in `/etc/keepalived` of the backup server. `start_docker.sh` starts the Milvus server of the backup server when the master server stops and directs the virtual IP to the backup server. `stop server` stops the Milvus server of the backup server when the master server starts working.

Use the following code for `start_docker.sh`:

```bash
#!/bin/bash
RUNNING_STATUS=$(docker inspect --format '{{.State.Running}}' <docker id>)
if [[ "${RUNNING_STATUS}" != "true" ]];then
docker start <docker id>
fi
```

`<docker id>` specifies the milvus server docker id of the backup server.

Use the following code for `stop_docker.sh`:

```bash
#!/bin/bash
RUNNING_STATUS=$(docker inspect --format '{{.State.Running}}' <docker id>)
if [[ "${RUNNING_STATUS}" = "true" ]];then
docker start <docker id>
fi
```

`<docker id>` specifies the milvus server docker id of the backup server.

> Note: You must add execute permission to the previously created scripts.

```bash
chmod +x chk_milvus.sh
chmod +x start_docker.sh
chmod +x stop_docker.sh
```

**Launch keepalived for the master and backup machines**

```bash
service keepalived start
# Check the status of keepalived
service keepalived status
```

**Check keepalived logs**

```bash
# cat /var/log/syslog | grep Keepalived | tail
```

## Validate

After completing the previous steps, enter `ip a` in the master server. You can see that the virtual IP address `192.168.1.104` is displayed in the master server.

![1575010275666](pic/1575010275666.png)

Then you can use the client to connect to Milvus server via 192.168.1.104 to create a table and insert/query vectors.

Shutdown the master server or stop the Milvus server in the master server and enter `ip a` again, you can see that the virtual IP address has been removed from the master server.

![1575015544553](pic/1575015544553.png)

Enter `ip a` in the backup and you can see that the virtual IP address is transferred to the backup server.

![1575016581429](pic/1575016581429.png)

Enter `docker ps` and you can see that docker in the backup server is running. The Milvus server that connects to the client via 192.168.1.104 has been transferred to the backup server. Because the master server and the backup server share a storage device, the Milvus data is synchronized. In this solution, when the master server is down, the server connection can be recovered in seconds.

After recovering the Milvus server in  the master server, the virtual IP address is transferred to the master server, which connects to the client. The Milvus server in the backup server is stopped automatically.
