---
layout: post
title: Ubuntu 16.04.3 下安装 Apache Spark 和 Kafka
category: Machine Learning 
author: Hongtao Yu
tags: 
  - linux-unix 
  - big-data
comments: true
use_math: true
lang: zh
---

- TOC
{:toc}



# 安装 Scala 2.12 以及 SBT 

```bash
## Java
sudo apt-get update
sudo apt-get install default-jdk
echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64" >> ~/.bash_profile
source ~/.bash_profile

## Scala
sudo apt-get remove scala-library scala
sudo wget http://scala-lang.org/files/archive/scala-2.12.1.deb
sudo dpkg -i scala-2.12.1.deb
sudo apt-get update
sudo apt-get install scala

## SBT
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update
sudo apt-get install sbt
```

关于什么是 SBT，请参考[这里](https://github.com/CSUG/real_world_scala/blob/master/02_sbt.markdown)

# 安装 Kafka

```bash
## Kafka
wget https://www.apache.org/dyn/closer.cgi?path=/kafka/1.0.0/kafka_2.12-1.0.0.tgz
tar -zxvf kafka_2.12-1.0.0.tgz
mv kafka_2.12-1.0.0 /usr/local/kafka
echo "export KAFKA_HOME=/usr/local/kafka" >> ~/.bash_profile
source ~/.bash_profile

## Start Zookeeper
$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties

## Open another terminal and start the broker
$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties
```
如果看到信息 “INFO [KafkaServer id=0] started (kafka.server.KafkaServer)", 证明 Kafka broker/server 启动成功了。并且可以看到链接的端口号为9092.


## 创建 Topic

打开另一个 Terminal

```bash
$KAFKA_HOME/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test

$KAFKA_HOME/bin/kafka-topics.sh --list --zookeeper localhost:2181
```

## 创建 Producer 和 Consumer

新开一个 Terminal

```bash
$KAFKA_HOME/bin/kafka-console-producer.sh --broker-list localhost:9092--topic test
```

打开另一个 Terminal

```bash
$KAFKA_HOME/bin/kafka-console-consumer.sh --zookeeper localhost:2181 --topic test --from-beginning
```

# 安装 Hadoop 2.8.2

```bash
wget http://apache.mirrors.lucidnetworks.net/hadoop/common/hadoop-2.8.2/hadoop-2.8.2.tar.gz
tar -zxvf hadoop-2.8.2.tar.gz
mv hadoop-2.8.2 /usr/local/hadoop
echo "export HADOOP_HOME=/usr/local/hadoop" >> ~/.bash_profile
echo "export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$LD_LIBRARY_PATH" >> ~/.bash_profile
```

## 配置 Hadoop 的 Java 环境

使用 `which java` 命令可以看到，`/usr/bin/java` 是 `/etc/alternatives/java` 的一个符号链接，而后者又是 `/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java` 的一个符号链接。所以我们需要设置 `JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64`。可以通过带`-f` 选项的 `readlink` 命令直接找到 Java 的真实路径：

```bash
readlink -f /usr/bin/java | sed "s:bin/java::"
```
输出 `/usr/lib/jvm/java-8-openjdk-amd64/jre/`。

接下来在 `/usr/local/hadoop/etc/hadoop-env.sh`文件中写入下面任意一个就行：

**设置一个静态值**

```bash
 . . .
#export JAVA_HOME=${JAVA_HOME}
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre/
 . . . 
```
**或者使用 `Readlink` 设置一个动态值**

```bash
 . . .
#export JAVA_HOME=${JAVA_HOME}
export JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::")
 . . . 
```

然后运行 Hadoop

```bash
/usr/local/hadoop/bin/hadoop
``` 

将会看到如下信息：

```bash
Usage: hadoop [--config confdir] [COMMAND | CLASSNAME]
  CLASSNAME            run the class named CLASSNAME
 or
  where COMMAND is one of:
  fs                   run a generic filesystem user client
  version              print the version
  jar <jar>            run a jar file
                       note: please use "yarn jar" to launch
                             YARN applications, not this command.
  checknative [-a|-h]  check native hadoop and compression libraries availability
  distcp <srcurl> <desturl> copy file or directories recursively
  archive -archiveName NAME -p <parent path> <src>* <dest> create a hadoop archive
  classpath            prints the class path needed to get the
                       Hadoop jar and the required libraries
  credential           interact with credential providers
  daemonlog            get/set the log level for each daemon
  trace                view and modify Hadoop tracing settings

Most commands print help when invoked w/o parameters.
```

这就证明 Hadoop 的 **stand-alone 模式** 安装成功了！

## 测试 Hadoop

我们这里继续使用 Hadoop 自带的 MapReduce 例子做进一步测试，确保 Hadoop 正常运行。首先建立一个名为 input 文件夹，并将 Hadoop 的例子复制进去，然后运行 hadoop， 将结果输出到 grep_example 文件夹。

```bash
mkdir input
cp /usr/local/hadoop/etc/hadoop/*.xml input

/usr/local/hadoop/bin/hadoop jar /usr/local/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.8.2.jar grep input grep_example 'principal[.]*'
```

将会看到一下输出信息：

```bash
···
	File System Counters
		FILE: Number of bytes read=1275408
		FILE: Number of bytes written=2544028
		FILE: Number of read operations=0
		FILE: Number of large read operations=0
		FILE: Number of write operations=0
	Map-Reduce Framework
		Map input records=2
		Map output records=2
		Map output bytes=37
		Map output materialized bytes=47
		Input split bytes=132
		Combine input records=0
		Combine output records=0
		Reduce input groups=2
		Reduce shuffle bytes=47
		Reduce input records=2
		Reduce output records=2
		Spilled Records=4
		Shuffled Maps =1
		Failed Shuffles=0
		Merged Map outputs=1
		GC time elapsed (ms)=0
		Total committed heap usage (bytes)=1054867456
	Shuffle Errors
		BAD_ID=0
		CONNECTION=0
		IO_ERROR=0
		WRONG_LENGTH=0
		WRONG_MAP=0
		WRONG_REDUCE=0
	File Input Format Counters 
		Bytes Read=151
	File Output Format Counters 
		Bytes Written=37
```

查看输出结果 `cat ~/grep_example/*` 会显示以下信息：

```bash
6       principal
1       principal.
```

MapReduce 显示找到了一个后面带着点号的 principal 和六个不带点号的 principal。证明我们的 Hadoop 单机版安装一切正常。


# 安装 Spark 2.2.0

```bash
wget http://apache.osuosl.org/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz
tar -zxvf spark-2.2.0-bin-hadoop2.7.tgz
sudo mv spark-2.2.0-bin-hadoop2.7 /usr/local/spark
echo "SPARK_HOME=/usr/local/spark" >> ~/.bash_profile
source ~/.bash_profile
```

然后可以运行 `/usr/local/spark/bin/pyspark` 查看 Spark 是否正确安装。看到下面信息证明安装成功：

```bash
Python 2.7.12 (default, Nov 19 2016, 06:48:10) 
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
17/11/15 00:59:40 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
17/11/15 00:59:41 WARN Utils: Your hostname, htaoyu-ThinkPad-T400 resolves to a loopback address: 127.0.1.1; using 10.242.240.210 instead (on interface wlp3s0)
17/11/15 00:59:41 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
17/11/15 01:00:16 WARN ObjectStore: Version information not found in metastore. hive.metastore.schema.verification is not enabled so recording the schema version 1.2.0
17/11/15 01:00:16 WARN ObjectStore: Failed to get database default, returning NoSuchObjectException
17/11/15 01:00:19 WARN ObjectStore: Failed to get database global_temp, returning NoSuchObjectException
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.2.0
      /_/

Using Python version 2.7.12 (default, Nov 19 2016 06:48:10)
SparkSession available as 'spark'.
>>> 
```

## 启动 master 服务器

使用 `ifconfig` 命令查看网络信息：

```bash
lo        Link encap:Local Loopback  
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:377491 errors:0 dropped:0 overruns:0 frame:0
          TX packets:377491 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:26789953 (26.7 MB)  TX bytes:26789953 (26.7 MB)
wlp3s0    Link encap:Ethernet  HWaddr 00:23:4d:dc:e4:76  
          inet addr:10.242.240.210  Bcast:10.255.255.255  Mask:255.224.0.0
          inet6 addr: fe80::8a77:2d63:1795:a1bb/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:588888 errors:0 dropped:0 overruns:0 frame:0
          TX packets:413957 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:798159596 (798.1 MB)  TX bytes:38860902 (38.8 MB)
```

其中 `wlp3s0` 块中 `inet addr:10.242.240.210` 包含的就是我们的 IP 地址。有了这个信息，我们就可以用以下方式启动 master 服务器：

```bash
/usr/local/spark/sbin/start-master.sh -h 10.242.240.210
```
如果看到以下信息，就证明 Spark 运行成功了。

```bash
starting org.apache.spark.deploy.master.Master, logging to /usr/local/spark/logs/spark-htaoyu-org.apache.spark.deploy.master.Master-1-htaoyu-ThinkPad-T400.out
```

使用地址 `localhost:8080` 就可以在 IE 中查看 master 服务器的网页用户界面。在页面中你会看到形如 `spark://10.242.240.210:7077` 的 URL。这个 URL 就是你的 slaves 连接到集群的地址。


## 链接 Slaves 机器

```bash
/usr/local/spark/sbin/start-slave.sh spark://10.242.240.210:7077
```
运行后你将会看到和 master 服务器输出一样的信息。这时候刷新一下刚才 master 服务器的网页用户界面，你就可以看到一个 slave 加进来了。


# 安装 Sublime Text 3 编辑器（可选）

```bash
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
sudo apt-get update
sudo apt-get install sublime-text
```


# References


1. [Install Scala and SBT using apt-get on Ubuntu 16.04 or any Debian derivative using apt-get](https://gist.github.com/Frozenfire92/3627e38dc47ca581d6d024c14c1cf4a9)


2. [Step by Step of Installing Apache Kafka and Communicating with Spark](https://chongyaorobin.wordpress.com/2015/07/08/step-by-step-of-install-apache-kafka-on-ubuntu-standalone-mode/)

3. [How to Install Hadoop in Stand-Alone Mode on Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-hadoop-in-stand-alone-mode-on-ubuntu-16-04)

4. [How to configure an Apache Spark standalone cluster and integrate with Jupyter: Step-by-Step](https://www.davidadrian.cc/posts/2017/08/how-to-spark-cluster/)

