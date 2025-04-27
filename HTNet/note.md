python -m venv myenv 是用于创建 Python 虚拟环境的命令。下面为你详细解释这个命令以及相关操作：

命令解释
python：调用 Python 解释器。
-m venv：-m 选项用于将一个 Python 模块作为脚本运行，venv 是 Python 标准库中的虚拟环境模块。使用 venv 模块可以创建轻量级的虚拟环境。
myenv：指定虚拟环境的名称，你可以将其替换为自己想要的名称。执行该命令后，会在当前目录下创建一个名为 myenv 的文件夹，该文件夹包含虚拟环境的所有文件和配置。
执行示例
假设你在 D:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet 目录下执行此命令，操作步骤如下：

1. 打开命令行并切换到目标目录

bash
cd "D:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet"
2. 创建虚拟环境

bash
python -m venv myenv
执行完该命令后，会在当前目录下生成一个 myenv 文件夹，里面包含了虚拟环境所需的文件和目录。

3. 激活虚拟环境
Windows：

bash
myenv\Scripts\activate
激活后，命令行提示符前会显示虚拟环境的名称，例如 (myenv) D:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet>。

Linux/Mac：

bash
source myenv/bin/activate
4. 在虚拟环境中安装依赖
激活虚拟环境后，你可以使用 pip 安装项目所需的依赖，例如安装 requirements.txt 中的所有包：


bash
pip install -r requirements.txt
5. 退出虚拟环境
当你完成操作后，可以使用以下命令退出虚拟环境：


bash
deactivate
常见问题及解决办法
权限问题：如果在创建虚拟环境时遇到权限问题，你可以尝试以管理员身份运行命令行工具（Windows），或者使用 sudo 命令（Linux/Mac）。
Python 版本问题：确保你使用的 Python 版本支持 venv 模块。Python 3.3 及以上版本都包含 venv 模块。

# where.exe python
从 where.exe python 的输出可知，系统找到了两个 python.exe 文件路径，如果你想让系统不再指向 D:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\myenv\Scripts\python.exe，可以按以下方法处理：

1. 处理虚拟环境中的 Python
如果你项目里使用的 myenv 是虚拟环境，不想让系统默认调用它，可以停用这个虚拟环境，并且从环境变量里移除相关路径。

停用虚拟环境
在 PowerShell 里运行以下命令停用虚拟环境：


powershell
Apply
deactivate
移除环境变量中的路径
和之前操作类似，你可以通过图形界面或者 PowerShell 来移除 PATH 里的相关路径。

图形界面操作：

按下 Win + S 组合键，搜索“环境变量”，打开“编辑系统环境变量”。
在弹出窗口中，点击“环境变量”按钮。
在“系统变量”或者“用户变量”里找到 PATH 变量，选中后点击“编辑”。
查看列表，删除包含 D:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\myenv\Scripts 的路径项。
依次点击“确定”保存更改。

# 安装匹配的固定版本
4. 固定版本（可选）
如果上述方法还是无法解决问题，你可以尝试固定 numpy 和 opencv-python 的版本，选择经过验证的兼容版本组合：


powershell
Apply
# 安装特定版本的 numpy 和 opencv-python
pip install numpy==1.26.2 opencv-python==4.8.1.78
安装完成后，再次验证 cv2 模块是否能正常导入。
PS D:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition> python
Python 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> 
>>> print(cv2.__version__)
4.8.1

# 终于把opencv给装好了
注意运行环境和安装环境
venv 是 Python 标准库中的虚拟环境模块，用于创建独立的 Python 环境。
用where.exe python 会出现python的路径是项目的路径
而且有虚拟环境的时候不一定会显示env，deactivate 可以退出虚拟环境

主要numpy和opencv的版本要匹配，否则会报错人，如果requirements.txt里的版本太老了，安装一套新的匹配版本就可以。

# 图片无法正常显示
以无沙箱模式启动 VSCode：沙箱模式可能会阻止 Service Worker 的正常注册，导致该问题。

关闭 VSCode：确保 VSCode 已经完全关闭，没有任何进程在后台运行。
打开命令提示符：在 Windows 上，按下Win + R键，输入cmd并回车，打开命令提示符；在 macOS 或 Linux 上，可以打开终端。

使用特定命令启动 VSCode：在命令提示符或终端中，输入code --no-sandbox并回车，这将以无沙箱模式启动 VSCode。

# 图片找不到
用命令检查位置，再用os.path.join修改拼接格式
roycl@RoyThink14 MINGW64 /d/思扬/软件学院/三下/微表情识别/Micro-expression-recognition (main)
$ ls "d:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTnet\datasets\combined_datasets_whole\sub01_EP02_01f.jpg"
'd:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTnet\datasets\combined_datasets_whole\sub01_EP02_01f.jpg'

base_data_src = Path.cwd() /"HTnet" / "datasets" / "combined_datasets_whole" # 新加了/"HTnet" 

# 本地分支找不到关联的远端分支
新建并关联远端分支
git push -u origin main

验证关联分支
git branch -vv
若输出类似如下内容，表明关联成功：

* main  1234567 [origin/main] Initial commit


# 不是字符类型 将 Path 对象转换为字符串  train_face_image_apex = cv2.imread(str(img_path_apex)) 

[Running] python -u "d:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\main_HTNet.py"
lr=0.000050, epochs=800, device=cpu

Traceback (most recent call last):
  File "d:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\main_HTNet.py", line 424, in <module>
    main(config)
  File "d:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\main_HTNet.py", line 253, in main
    all_five_parts_optical_flow = crop_optical_flow_block() # 裁剪光流图像
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "d:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\main_HTNet.py", line 142, in crop_optical_flow_block
    face_block_coordinates_dict = whole_face_block_coordinates()
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "d:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\main_HTNet.py", line 93, in whole_face_block_coordinates
    train_face_image_apex = cv2.imread(img_path_apex) # (444, 533, 3)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Can't convert object to 'str' for 'filename'
Trae AI
从错误信息 TypeError: Can't convert object to 'str' for 'filename' 可知，cv2.imread 函数期望接收字符串类型的文件路径，然而实际传入的不是字符串类型，从而引发错误。下面为你分析可能的原因并给出解决办法。

可能的原因
img_path_apex 不是字符串类型：在拼接路径时，可能由于某些操作导致 img_path_apex 变成了非字符串类型，比如 Path 对象没有正确转换为字符串。
CSV 文件中包含非字符串数据：df['imagename'][i] 可能包含非字符串数据，在拼接路径时导致问题。

# 数据集找不到路径
FileNotFoundError: [WinError 3] 系统找不到指定的路径。: '.\\datasets\\STSNet_whole_norm_u_v_os'
改为固定路径whole_optical_flow_path = r'd:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\datasets\STSNet_whole_norm_u_v_os'

# 生成名字不匹配
[Running] python -u "d:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\main_HTNet.py"
is_cuda: False
lr=0.000050, epochs=800, device=cpu

Traceback (most recent call last):
  File "d:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\main_HTNet.py", line 431, in <module>
    main(config)
  File "d:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\main_HTNet.py", line 260, in main
    all_five_parts_optical_flow = crop_optical_flow_block() # 裁剪光流图像
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "d:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\main_HTNet.py", line 161, in crop_optical_flow_block
    four_part_coordinates = face_block_coordinates_dict[n_img]
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^
KeyError: '006_006_1_2 .png'

从错误信息 KeyError: '006_006_1_2 .png' 可知，在 crop_optical_flow_block 函数里，尝试从 face_block_coordinates_dict 字典中获取键为 '006_006_1_2 .png' 的值时，该键不存在。这一般是因为 whole_face_block_coordinates 函数生成的字典键和 crop_optical_flow_block 函数里使用的文件名不匹配。下面为你分析可能的原因和解决办法。

可能的原因
文件名生成规则不一致：whole_face_block_coordinates 函数生成图像文件名的方式和 crop_optical_flow_block 函数里使用的文件名不同。
文件扩展名问题：可能存在文件扩展名不一致的情况。
空格问题：在 whole_face_block_coordinates 函数里，生成的文件名 image_name 包含多余空格，如 '006_006_1_2 .png' 中间有多余空格。

对比两个函数生成的文件名
print(f"whole_face_block_coordinates 生成的 image_name: {image_name}")  # 打印生成的文件名
把不匹配的修改

# 预训练模型权重文件不存在
从错误信息 FileNotFoundError: [Errno 2] No such file or directory: 'ourmodel_threedatasets_weights/006.pth' 可知，程序在尝试加载预训练模型权重文件 ourmodel_threedatasets_weights/006.pth 时，发现该文件不存在。下面为你分析可能的原因及对应的解决办法。

可能的原因
权重文件未生成：在训练模式下，程序没有正确保存模型权重文件，或者你没有以训练模式运行过程序，导致权重文件根本没有生成。
路径错误：权重文件实际保存的路径和代码中指定的路径不一致。
文件被误删或移动：权重文件可能被手动删除或者移动到了其他位置。

解决办法

1. 检查训练模式
如果你想使用预训练的模型权重，需要先以训练模式运行程序，让程序生成权重文件。在命令行中运行以下命令开启训练模式：



```bash 
python d:/思扬/软件学院/三下/微表情识别/Micro-expression-recognition/HTNet/main_HTNet.py --train True
```

2. 添加文件存在检查
在加载权重文件之前，添加文件存在检查，避免因文件不存在而报错。修改 main 函数中的相关代码：


``` python
 ... 已有代码 ...

def main(config):
    # ... 已有代码 ...
    for n_subName in subName:
        print('Subject:', n_subName)
        # ... 已有代码 ...

        # 当前受试者模型权重保存路径
        weight_path = 'ourmodel_threedatasets_weights' + '/' + n_subName + '.pth' 

        # Reset or load model weigts
        model = HTNet(
            image_size=28, # 输入图像尺寸
            patch_size=7, # 分块尺寸
            dim=256,  # 256,--96, 56-, 192 模型维度
            heads=3,  # 3 ---- , 6- 多头注意力的头数
            num_hierarchies=3,  # 3----number of hierarchies 模型层次的数量
            block_repeats=(2, 2, 10),#(2, 2, 8),------ 每个层次中Transformer块的数量
            # the number of transformer blocks at each heirarchy, starting from the bottom(2,2,20) -
            num_classes=3 #分类的类别数量：正向，负向，惊讶
        )

        model = model.to(device) # 将模型移动到指定的设备（GPU 或 CPU）上进行计算

        if(config.train): # 训练模式
            # model.apply(reset_weights) # 重置模型权重
            print('train')
        else:
            if os.path.exists(weight_path):
                model.load_state_dict(torch.load(weight_path)) # 如果不处于训练模式，从指定路径加载预训练的模型权重
            else:
                print(f"权重文件 {weight_path} 不存在，使用初始模型权重。")
                # 这里可以选择重置模型权重
                model.apply(reset_weights)

        # ... 已有代码 ...
        ```
3. 检查权重文件路径
确保 ourmodel_threedatasets_weights 文件夹存在，并且权重文件确实保存在该文件夹下。你可以手动检查该文件夹，确认权重文件是否存在。

4. 打印调试信息
在加载权重文件之前，打印权重文件的路径，方便确认路径是否正确：


``` python
# ... 已有代码 ...

def main(config):
    # ... 已有代码 ...
    for n_subName in subName:
        print('Subject:', n_subName)
        # ... 已有代码 ...

        # 当前受试者模型权重保存路径
        weight_path = 'ourmodel_threedatasets_weights' + '/' + n_subName + '.pth' 
        print(f"尝试加载的权重文件路径: {weight_path}")

        # ... 已有代码 ...

        if(config.train): # 训练模式
            # model.apply(reset_weights) # 重置模型权重
            print('train')
        else:
            if os.path.exists(weight_path):
                model.load_state_dict(torch.load(weight_path)) # 如果不处于训练模式，从指定路径加载预训练的模型权重
            else:
                print(f"权重文件 {weight_path} 不存在，使用初始模型权重。")
                # 这里可以选择重置模型权重
                model.apply(reset_weights)

        # ... 已有代码 ...
        ```
通过以上步骤，你可以确保权重文件存在并正确加载，避免 FileNotFoundError 错误。
# 原来能跑后来又不行了：路径问题
D:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition>python d:/思扬/软件学院/三下/微表情识别/Micro-expression-recognition/HTNet/main_HTNet.py --train True