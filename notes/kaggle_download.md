# 从 Kaggle 下载数据集

Kaggle 是全球最大的数据科学社区，里面具有丰富的数据集和相关教程代码，是所有数据科学相关专业和从业人员必须熟悉的网站。

本文介绍如何使用终端/命令行工具从 Kaggle 下载数据（mac 和 win 系统）。官网地址 [https://www.kaggle.com/](https://www.kaggle.com/)

---

## 1 注册 kaggle 账户

进入官网地址 [https://www.kaggle.com/](https://www.kaggle.com/)

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1740215562957.png)

点击【Register】注册

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1740215612893.png)

可以选择 Google 账户登陆（【推荐】），或换邮箱注册。



## 2 下载 API Token

点击右侧头像，选择【Settings】，找到【API】，点击【Create New Token】

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1740215710310.png)

点击后会下载一个 `kaggle.json` 文件，保存好这个文件，等会使用。



## 3 安装 kaggle 包

新建虚拟环境，并在虚拟环境中下载安装 kaggle 包

> 虚拟环境是可选，但推荐在虚拟环境中操作，有利于后续管理。有关如何搭建虚拟环境可见教程 [Conda 创建虚拟环境全流程](https://zhuanlan.zhihu.com/p/21629604277)

使用 `pip` 下载，在终端中（激活环境后）输入，然后回车

```bash
pip install kaggle
```

完成后可以继续输入

```bash
kaggle competitions list
```

此时应该无法正常使用 kaggle ，需要我们去往用户目录下配置。



##  4 配置 `.kaggle`

### 4.1 MacOS 系统

对于 mac 系统，前往用户目录下。一般为 `~` 或 `/Users/<你的用户名>` ，在终端 (Terminal) 中输入

```bash
cd ~
```

> 一般默认打开终端就已经是位于用户目录下

此时显示如下，表示成功进入用户目录下

```bash
<你的用户名>@MacBook ~ % 
```

然后再输入如下指令，就可以打开文件夹 `.kaggle`

```bash
open .kaggle
```

此时再把第二步下载的 API Token 文件 `kaggle.json` 移入 `.kaggle` 文件夹。

>对于 mac 用户打开用户目录比较复杂。`open 文件夹名` 指令能直接以资源管理器的方式打开。
>
>与此同时，mac 还会默认隐藏一些文件夹，例如这里的 `.kaggle` ，可以在进入目录之后，同时按下键盘的 `cmd + shift + .` 就可以查看隐藏的文件和文件夹。

- 此时便可以正常使用 kaggle

```bash
kaggle competitions list
```

> 如果实在虚拟环境中 pip 安装的 kaggle ，需要激活虚拟环境才能正常使用。

### 4.2 Windows 系统

Windows 系统操作与 mac 相同，而且寻找用户文件夹更为简单。（类似地，【推荐】在虚拟环境里操作）

- 如果使用 PowerShell 终端的话，指令相同，只是用户目录一般为

```bash
cd C:\Users\<你的用户名>
```

然后打开文件夹

```bash
explorer .kaggle
```

相同地，将 `kaggle.json` 文件移入 `.kaggle` 文件夹。

- 或者直接点击【C 盘】，点击【Users】，进入【<你的用户名>】，找到【.kaggle】文件夹实行与上面相同的操作。Windows 电脑一般不会隐藏文件。



## 5 配置 kaggle 数据集下载路径

我们希望数据能下载在同一处文件夹内，方便以后程序的读取，而不用每次都在不同的地方下载数据，这会造成电脑存储空间的浪费。

这一步操作 mac 和 win 系统没有区别，只是路径的书写要注意。

- 打开【.kaggle】文件夹内的【kaggle.json】文件进行编辑。

原始 `kaggle.json` 文件默认为

```json
{
  "username": "用户名",
  "key": "密钥"
}
```

此时在文件中添加一行

```json
# mac
{
  "username": "用户名",
  "key": "密钥",
  "path": "/Users/用户名/AllData"
}

# win
{
  "username": "用户名",
  "key": "密钥",
  "path": "D:\\AllData"
}
```

如此以后所有的数据都会下载在 `/Users/用户名/AllData` AllData 文件夹里。（win 可以存放在 D 盘，注意要使用两个反斜杠 `\\` ）



## 6 从 kaggle 官网下载数据

在上面都配置好后，可以前往 kaggle 官网下载数据。下面以猫狗分类竞赛数据集为例，地址 [dog-vs-cat-classification](https://www.kaggle.com/competitions/dog-vs-cat-classification/data)

- 找到【Data】界面，然后向下翻找 kaggle 命令

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1740217509899.png)

例如此处输入如下，就可以正常下载了。

```bash
kaggle competitions download -c dog-vs-cat-classification
```

下载完成后去往我们设置的 AllData 文件夹查看数据集。

```bash
AllData
└── competitions
    └── dog-vs-cat-classification
        └── dog-vs-cat-classification.zip
```

会得到一个 `.zip` 文件，解压后即为数据集。



















