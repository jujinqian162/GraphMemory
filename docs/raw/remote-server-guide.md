# 对于访问远程服务器并运行训练命令的指南

# 准备工作
检查ssh tool工具是否正常
检查服务器可用cuda device
检查zellij位置（EXAMPLE: 在sensecore服务器，zellij无法直接访问，其在/mnt/afs/zhengmingkai/jjq/bootstrap/cargo/bin/zellij，使用绝对路径访问)
检查服务器环境（EXAMPLE: sensecore服务器没有uv，而是用conda，只需要先source /mnt/afs/zhengmingkai/jjq/GraphMemory/cloud_setup/setup.bash就可以直接用python运行脚本了)
# 执行需要运行的训练命令

## 默认行为
|> 若用户没有指明则使用以下默认行为

尽量让每个命令并行执行，并充分利用每个cuda device. (像sensecore服务器要用mx-smi查询可用国产显卡)
zellij 默认 session为 `agent`.

## 运行规范
跑命令不要直接使用ssh bash跑（train+retrieve可能耗时数十个小时，使用ssh直接跑易受网络波动影响）
所以**使用zellij**。
我给你准备了一个zellij session：`agent`。
**命令默认运行结构**：
```bash
zellij --session agent action new-tab \
    --cwd <project workspace dir> \
    --name <name for the task> \
    -- bash -lc 'source cloud_setup/setup.bash && <command needs to run>'
```

命令非阻塞且应该会返回tab-id (pane-id 可以通过 `zellij --session agent action list-panes` 得到). 
等一段时间后使用 `zellij --session agent action dump-screen --pane-id <pane_id>` 得到此时的日志快照，以此查看任务是否正常启动。若无输出则再等待一段时间再dump(等待程序启动)。

# 优化布局
对于几个训练命令，不必优化布局，可以都使用action new-tab将任务跑在不同的tab上。
但是假设需要并行运行的命令非常多，比如一些全量baselines+multiseed任务，可能有超过10甚至20个task需要运行。
**首先当然是评估服务器可以承受的并行task数量**：这部分我建议ask user。
如果当个task算力要求低，各种task各种multiseed任务可以大量并行运行，那就不要全部使用action new-tab了，需要稍微组织一下布局：
## 组织布局
举个例子，有4个baseline，每个baseline要跑5个seed。那么建议创建4个tab分别对应4个baseline name。
然后使用：
```bash
zellij --session agent run \                                                                                             
    --tab-id <tab-id> \
    --name <name for the task> \
    --cwd <project workspace dir>
    -- bash -lc 'source cloud_setup/setup.bash && <command needs to run>'
```
进行组织

