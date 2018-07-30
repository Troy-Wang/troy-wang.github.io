---
layout:     post
title:      "Jedis使用指南"
subtitle:   "翻译自Jedis Wiki"
date:       2018-07-30
author:     "Troy Wang"
header-img: "img/post/hobbiton1.jpg"
tags:
    - Redis
    - Jedis
    - Java
---

# Jedis使用指南

翻译自[Jedis Wiki](https://github.com/xetorthio/jedis/wiki)。

* TOC
{:toc}

## 快速入门

### 安装Jedis

你可以这样为你的应用添加Jedis的依赖：

**使用Jar文件**<br>
从[search.maven.org](http://search.maven.org/)或者其他的maven仓库下载最新的[Jedis](http://search.maven.org/#search%7Cgav%7C1%7Cg%3A%22redis.clients%22%20AND%20a%3A%22jedis%22)和[Apache Common Pool2](http://search.maven.org/#search%7Cgav%7C1%7Cg%3A%22org.apache.commons%22%20AND%20a%3A%22commons-pool2%22)的jar文件。

**使用源码编译**<br>
这样你可以得到最新的版本。

**从Github项目复制**<br>
很简单，你只需要在命令行中执行:
```powershell
git clone git://github.com/xetorthio/jedis.git
```

**使用Github编译**<br>
在使用maven打包之前需要通过验证。使用如下命令进行验证和打包：
```powershell
make package
```

**使用Maven依赖**<br>
可以在Sonatype上找到Jedis的Maven依赖，只需要在项目的pom.xml中加入以下XML片段：
```xml
<dependency>
    <groupId>redis.clients</groupId>
    <artifactId>jedis</artifactId>
    <version>2.9.0</version>
    <type>jar</type>
    <scope>compile</scope>
</dependency>
```

### 使用示例

#### 在多线程环境中使用Jedis
不应该在多线程中使用同一个Jedis实例，不然会产生很多奇怪的错误。但是创建多个Jedis示例又不是最优选择，因为这样意味着很多的socket和连接，同样会引起很多奇怪的错误。*Jedis实例并不是线程安全的！*为了避免以上问题，建议使用JedisPoll，这是一个线程安全的连接池。使用连接池可以安全地创建多个Jedis实例，并在使用结束后回收这些连接，进而避免那么奇怪的错误，取得更好的性能。

首先，初始化一个连接池：
```java
JedisPool pool = new JedisPool(new JedisPoolConfig(), "localhost");
```
你可以静态地存储连接池，它是线程安全的。

JedisPoolConfig针对Redis提供了不少很有用的默认配置。JedisPool基于Commons Pool2，详细内容请参阅[http://commons.apache.org/proper/commons-pool/apidocs/org/apache/commons/pool2/impl/GenericObjectPoolConfig.html](http://commons.apache.org/proper/commons-pool/apidocs/org/apache/commons/pool2/impl/GenericObjectPoolConfig.html)。

然后：
```java
/// Jedis implements Closeable. Hence, the jedis instance will be auto-closed after the last statement.
try (Jedis jedis = pool.getResource()) {
  /// ... do stuff here ... for example
  jedis.set("foo", "bar");
  String foobar = jedis.get("foo");
  jedis.zadd("sose", 0, "car"); jedis.zadd("sose", 0, "bike"); 
  Set<String> sose = jedis.zrange("sose", 0, -1);
}
/// ... when closing your application:
pool.close();
```

如果你无法使用try-with-resource，可以直接使用Jedis.close().
```java
Jedis jedis = null;
try {
  jedis = pool.getResource();
  /// ... do stuff here ... for example
  jedis.set("foo", "bar");
  String foobar = jedis.get("foo");
  jedis.zadd("sose", 0, "car"); jedis.zadd("sose", 0, "bike"); 
  Set<String> sose = jedis.zrange("sose", 0, -1);
} finally {
  // You have to close jedis object. If you don't close then
  // it doesn't release back to pool and you can't get a new
  // resource from pool.
  if (jedis != null) {
    jedis.close();
  }
}
/// ... when closing your application:
pool.close();
```

如果Jedis连接是从连接池中获取的，则在发生异常时，它会被连接池回收；而如果不是从连接池获取的，这个连接会被断开，并且关闭。

#### 主从配置

**打开复制功能**<br>
Redis设计时就提供了主/从的分布式架构。也就是说，写请求会被指定到一个主节点上，通过主从复制功能把变更再同步到从节点。读请求会被定位到从节点上，从而减轻主节点的压力。
你可以使用上面描述的方法建立主节点。可以使用如下两种方法来为一个主节点指定从节点，从而打开复制功能：
- 在Redis从节点的配置文件中指定主从关系
- 在一个指定的jedis示例上，请求slaveOf方法，提供主节点的IP和端口
```java
jedis.slaveof("localhost", 6379);  //  if the master is on the same PC which runs your code
jedis.slaveof("192.168.1.35", 6379); 
```

注意：Redis 2.6版本后，从节点默认是只读的，所以在从节点上执行写请求会报错。如果改变了该配置，则它们会变成一个正常的Redis节点，并且可以顺利地接受写请求，但是变更不会被复制到其他节点。这种情况下，如果你混淆了多个jedis实例，可能会导致一些内容在无意间被覆盖。

**关闭复制功能/在主节点宕机时将一个从节点提升为主节点**<br>
如果主节点宕机，你可以需要将一个从节点提升为主节点。你应该首先尝试停掉原主节点的复制功能，然后如果存在多个从节点，重新指定它们作为新主节点的从节点。

```java
slave1jedis.slaveofNoOne();
slave2jedis.slaveof("192.168.1.36", 6379); 
```

## 高级功能

### 事务

你需要在一个*事务块*中封装所有的操作，以便实现事务，类似一个“管道”：
```java
jedis.watch (key1, key2, ...);
Transaction t = jedis.multi();
t.set("foo", "bar");
t.exec();
```

注意：如果在事务块中用到了返回值，你可以这样：
```java
Transaction t = jedis.multi();
t.set("fool", "bar"); 
Response<String> result1 = t.get("fool");

t.zadd("foo", 1, "barowitch"); t.zadd("foo", 0, "barinsky"); t.zadd("foo", 0, "barikoviev");
Response<Set<String>> sose = t.zrange("foo", 0, -1);   // get the entire sortedset
t.exec();                                              // dont forget it

String foolbar = result1.get();                       // use Response.get() to retrieve things from a Response
int soseSize = sose.get().size();                      // on sose.get() you can directly call Set methods!

// List<Object> allResults = t.exec();                 // you could still get all results at once, as before
```

注意，一个Response对象中并不包含任何东西直到t.exe()被执行。忽略执行exec将会抛出异常。上述代码的最后几行中，你可以看到版本2之前是怎么处理事务的，现在你仍然可以这么做，只不过需要从一个列表中再获取对象，这个列表中同样包含了Redis的状态信息。

注意，Redis不允许在同一个事务中使用该事务中的中间结果，下面这样方式行不通：
```java
// this does not work! Intra-transaction dependencies are not supported by Redis!
jedis.watch(...);
Transaction t = jedis.multi();
if(t.get("key1").equals("something"))
   t.set("key2", "value2");
else 
   t.set("key", "value");
```

*但是，存在一些命令（比如setnx），本身就包含条件执行能力，这些命令在事务中当然是可以的*。当然，你也可以使用eval或者LUA脚本定制自己的命令。

### 管道
有些时候，你可能需要发送一系列命令到Redis，一种更好的实现方式是使用管道（pipelining）。这样应用层可以发送命令而不需要等待回复，在最后读取回复并处理即可，这样性能会更好，速度会更快：
```java
Pipeline p = jedis.pipelined();
p.set("fool", "bar"); 
p.zadd("foo", 1, "barowitch");  p.zadd("foo", 0, "barinsky"); p.zadd("foo", 0, "barikoviev");
Response<String> pipeString = p.get("fool");
Response<Set<String>> sose = p.zrange("foo", 0, -1);
p.sync(); 

int soseSize = sose.get().size();
Set<String> setBack = sose.get();
```
详细解释请参考“事务”一小节的代码注释。

### Publish/Subscribe 发布/订阅
为了在Redis中订阅一个频道，创建一个JedisPubSub实例，将其做为参数调用jedis实例的subscribe方法即可。

```java
class MyListener extends JedisPubSub {
        public void onMessage(String channel, String message) {
        }

        public void onSubscribe(String channel, int subscribedChannels) {
        }

        public void onUnsubscribe(String channel, int subscribedChannels) {
        }

        public void onPSubscribe(String pattern, int subscribedChannels) {
        }

        public void onPUnsubscribe(String pattern, int subscribedChannels) {
        }

        public void onPMessage(String pattern, String channel, String message) {
        }
}

MyListener l = new MyListener();

jedis.subscribe(l, "foo");
```

注意，订阅是一个阻塞式的操作，请求jedis实例subscribe方法的线程将一直轮询Redis。一个JedisPubSub实例可以订阅多个频道。你也可以针对同一个JedisPubSub实例调用多次subscribe和psubscribe方法来切换订阅的频道。

### ShardedJedis Jedis分片

##### 动机
在基本的Redis主从结构中，通常只有一个主节点接受写请求，多个从节点接受度请求。这意味着用户不得不关注如何才能将读请求有效地分散到多个从节点上。另外，只有读请求具备了扩展的能力，而写请求仍然是落到一台服务器上，从而造成了性能瓶颈。利用Jedis的分片功能（ShardedJedis）可以轻松实现写请求和读请求的扩展。分片功能使用了“一致性哈希”技术，使用一些哈希算法（md5和murmur，后者更快）使键值对比较均匀地分布到多台Redis服务器上。这样的其中一台机器我们称之为“一个分片”（A shard）。这样做的另外一个好处是，每一个分片只需要总数据量1/n的内存（n为参与的节点的数目）。

##### 缺点
因为每个分片都是一个单独的主节点，所以分片技术也有一些缺点：比如，不能使用事务、管道、发布订阅功能，尤其是跨分片的时候！但是，如果使用到的键值对都在同一个分片上，其实是可以使用上述功能的（可以去论坛获取更多的解决方案）。使用keytags来决定键值对去到哪一个分片。另外一个缺点是，当前的实现中，在一个正在运行的ShardedJedis上不可以增加和删除分片。如果你需要这个功能，这里有一个实验性的ShardedJedis再实现：[yaourt - dynamic sharding implementation](https://github.com/xetorthio/jedis/pull/174)。

##### 使用方法

1. **定义分片：**
```java
List<JedisShardInfo> shards = new ArrayList<JedisShardInfo>();
JedisShardInfo si = new JedisShardInfo("localhost", 6379);
si.setPassword("foobared");
shards.add(si);
si = new JedisShardInfo("localhost", 6380);
si.setPassword("foobared");
shards.add(si);
```
有两种方式使用ShardedJedis：直接建立连接，或者使用连接池ShardedJedisPool。为了保证安全，在多线程环境中必须使用后者。

2. **直接建立并使用连接：**
```java
ShardedJedis jedis = new ShardedJedis(shards);
jedis.set("a", "foo");
jedis.disconnect();
```

3. **使用连接池：**
```java
JedisPoolConfig jedisPoolConfig = new JedisPoolConfig();
ShardedJedisPool pool = new ShardedJedisPool(jedisPoolConfig, shards);
try (ShardedJedis jedis = pool.getResource()) {
    jedis.set("a", "foo");
}
try (ShardedJedis jedis2 = pool.getResource()) {
    jedis2.set("z", "bar");
}
pool.close();
```

4. **断开连接/返还连接资源**
在结束使用jedis时，需要调用pool.returnResource返还资源。如果不返回的话，连接池会在一段时间后变得很慢。获取和返还资源很快，因为不需要建立或者销毁连接。建立和销毁一个连接池很慢，因为需要建立真正的网络连接。忘记调用pool.close会让连接一直保持直到超时。

**获取某个键值的分片信息**
```java
ShardInfo si = jedis.getShardInfo(key);
si.getHost/getPort/getPassword/getTimeout/getName
```
**指定某些键值到同一个分片**
Jedis支持“keytags”来实现指定某些键值到同一个分片。使用keytags你需要在实例化ShardedJedis时设置一个patter，例如：
```java
ShardedJedis jedis = new ShardedJedis(shards,
ShardedJedis.DEFAULT_KEY_TAG_PATTERN); //Pattern.compile("\\{(.+?)\\}");
```
默认的pattern是{}，也就是说在括号中的内容会被用来决定该key属于哪个分片。当然你也可以指定自己的pattern。
所以：
```java
jedis.set("foo{bar}", "12345");
```
和
```java
jedis.set("car{bar}", "877878");
```
两个键值会走到同一个分片。

##### 混合方式

如果你即想使用ShardedJedis的分布式功能，又想使用事务/管道/发布订阅等功能，你可以混合使用普通功能和分片功能：定义一个节点为普通主节点，另外一批服务器使用分片方式实现，并做为主节点的从节点。在你的应用中，将写请求分发到主节点上，读请求发送到多个从节点上。这样的话，写请求不再具有扩展的能力，但是读请求获得了扩展的能力，而且你可以在主节点上使用事务/管道/发布订阅等功能。主节点的内存需要能够支撑整个数据集。记住，如果让从节点代替主节点做持久化的工作，主节点的性能可以得到很大的提升。

##### Redis Cluster
2011年后的某个时间，“redis cluster”将会发布第一个版本，它将会是一个增强版的Sharded Jedis，并且会提供一些ShardedJedis没有的功能。如果想了解更多关于redis cluster， 可以在youtube上找到redis创始人Salvatore Sanfilippo的演示视频。

##### 监控
如果想使用redis的监控功能，可以这样：
```java
new Thread(new Runnable() {
    public void run() {
        Jedis j = new Jedis("localhost");
        for (int i = 0; i < 100; i++) {
            j.incr("foobared");
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
            }
        }
        j.disconnect();
    }
}).start();

jedis.monitor(new JedisMonitor() {
    public void onCommand(String command) {
        System.out.println(command);
    }
});
```

### 杂项

**关于String和Binary**<br>
Redis/Jedis在很多场景下是关于String的。[这里](http://redis.io/topics/internals)说String是Redis最基本的编译单元。但是这里所说的String可能有点误导人。Redis的String实际上是指8位的C语言中的char类型，和Java 16位的String是不兼容的。Redis只能感知到8位的内容，所以通常情况下Redis并不会对数据内容进行“翻译”。在Java中，byte[]类型的数据是“native”的，所以字符串String在传输之前需要被编码，并且在收到后进行解码才能解析。这会对性能有些小影响。简单来说，如果你手里有二进制数据，不需要将它编码成字符串，直接使用即可。

**关于Redis的主从架构**<br>
一个Redis架构由多台redis服务器构成，这些服务器要么是主节点，要么是从节点。从节点通过复制功能与主节点保持同步。但是客户端感知不到主节点和从节点的区别。另外，一个从节点可以被其它下层的从节点试做主节点。

## 常见问题

### java.net.SocketTimeoutException: Read timed out异常
在使用如下方式建立JedisPool时，设置自定义的超时时间：
```java
JedisPool(GenericObjectPoolConfig poolConfig, String host, int port, int timeout)
```

*timeout*单位是毫秒，默认的超时时间是2秒。

### 在获取8个连接后JedisPool被阻塞

JedisPool默认支持8个连接，你可以再PoolConfig中变更：
```java
JedisPoolConfig poolConfig = new JedisPoolConfig();
poolConfig.setMaxTotal(maxTotal); // maximum active connections
poolConfig.setMaxIdle(maxIdle);  // maximum idle connections
```

注意JedisPool继承自common-pool的[BaseObjectPoolConfig](https://commons.apache.org/proper/commons-pool/api-2.3/org/apache/commons/pool2/impl/BaseObjectPoolConfig.html)，后者有大量的配置参数。我们为大部分场景设置了合适的参数值。在一些情况下，你对这些参数进行调优时可以会遇到[以下问题](https://github.com/xetorthio/jedis/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+JedisPool)。

## 参考链接

- [Redis: under the hood](http://pauladamsmith.com/articles/redis-under-the-hood.html)
- [Salvatore's Blog](http://antirez.com/)
- [Redis in Action](http://www.manning.com/carlson/)
- [Redis: The Definitive Guide](http://shop.oreilly.com/product/0636920014294.do)
