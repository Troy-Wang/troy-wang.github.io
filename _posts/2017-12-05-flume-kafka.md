---
layout:     post
title:      "flume + kafka"
subtitle:   "flume + kafka"
date:       2017-12-05
author:     "Troy Wang"
header-img: "img/post/default-dusk.jpg"
tags:
    - Flume
    - Kafka
---

#### 1. start zookeeper packaged with kafka

```bash
> bin/zookeeper-server-start.sh config/zookeeper.properties
```

zookeeper.properties:

```properties
clientPort=2181
```

#### 2. start kafka

```bash
> bin/kafka-server-start.sh config/server.properties
```

server.properties:

```properties
listeners=PLAINTEXT://127.0.0.1:9092
zookeeper.connect=localhost:2181
```

#### 3. create kafka topic

```bash
> bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic flume-test
```

check topic list:

```bash
> bin/kafka-topics.sh --list --zookeeper localhost:2181
```

#### 4. kafka consumer

```python
from confluent_kafka import Consumer, KafkaError

c = Consumer({'bootstrap.servers': '127.0.0.1:9092', 'group.id': 'mygroup',
              'default.topic.config': {'auto.offset.reset': 'smallest'}})
c.subscribe(['flume-test'])
running = True
while running:
    msg = c.poll()
    if not msg.error():
        print('Received message: %s' % msg.value().decode('utf-8'))
    elif msg.error().code() != KafkaError._PARTITION_EOF:
        print(msg.error())
        running = False
c.close()
```

#### 5. start flume

flume-conf.properties:

```properties
# The configuration file needs to define the sources, 
# the channels and the sinks.
# Sources, channels and sinks are defined per agent, 
# in this case called 'agent'

agent1.sources = tailSource
agent1.channels = memoryChannel
agent1.sinks = kafkaSink

# For each one of the sources, the type is defined
agent1.sources.tailSource.type = exec
agent1.sources.tailSource.command = tail -F /Users/troywang/log.file
agent1.sources.tailSource.channels = memoryChannel

# Each sink's type must be defined
agent1.sinks.kafkaSink.type = org.apache.flume.sink.kafka.KafkaSink
agent1.sinks.kafkaSink.kafka.topic = flume-test
agent1.sinks.kafkaSink.kafka.bootstrap.servers = localhost:9092
agent1.sinks.kafkaSink.kafka.flumeBatchSize = 20
agent1.sinks.kafkaSink.kafka.producer.acks = 1
agent1.sinks.kafkaSink.kafka.producer.linger.ms = 1
agent1.sinks.kafkaSink.kafka.producer.compression.type = snappy
agent1.sinks.kafkaSink.channel = memoryChannel

# Each channel's type is defined.
agent1.channels.memoryChannel.type = memory

# Other config values specific to each type of channel(sink or source)
# can be defined as well
# In this case, it specifies the capacity of the memory channel
agent1.channels.memoryChannel.capacity = 100
agent1.channels.memoryChannel.transactionCapacity = 100
```

```bash
> ./flume-ng agent -n agent1 -c ../conf -f ../conf/flume-conf.properties -Dflume.root.logger=DEBUG,console
```

#### 6. run test

```bash
> echo 'this is a flume test from troy' >> log.file
```
python console output: 
> Received message: test
> Received message: this is a flume test from troy
> Received message: this is a flume test from troyfasfdas
> Received message: this is a flume test from troyfasfdas111
