from os import path
import os 
import numpy as np # 数值计算

import time

import pandas
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool # 用于将字符串转换为布尔值
import torch # 导入pytorch
from Model import HTNet # 导入模型
import numpy as np
from facenet_pytorch import MTCNN # 人脸检测
import cv2
from pathlib import Path

# Some of the codes are adapted from STSNet
def reset_weights(m):  # Reset the weights for network to avoid weight leakage 
    # 重置神经网络中具有 reset_parameters 方法的层的权重，避免权重泄露。
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #             print(f'Reset trainable parameters of layer = {layer}') # 正在重置参数的层信息
            layer.reset_parameters() # 调用子模块的 reset_parameters 方法重置层的参数

# 混淆矩阵
def confusionMatrix(gt, pred, show=False): 
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN) # 计算F1分数，精确率和召回率的调和平均数
    num_samples = len([x for x in gt if x == 1]) # 计算正样本的数量
    average_recall = TP / num_samples   # 计算平均召回率
    return f1_score, average_recall

# 多分类模型评估
def recognition_evaluation(final_gt, final_pred, show=False): #评估多分类模型在不同类别上的性能，最终计算并返回平均 F1 分数（UF1）和平均召回率（UAR）
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    # Display recognition result
    f1_list = [] # 存储每个类别的 F1 分数
    ar_list = [] # 存储每个类别的平均召回率average
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            # 转化为二分类标签，符合为1，不符合为0
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog) # 在末尾添加一个新的元素
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list) # 计算平均 F1 分数
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''


# 1. get the whole face block coordinates
# 从 CSV 文件读取数据，然后遍历其中的每一条记录，读取对应的人脸图像，
# 使用 MTCNN 人脸检测模型检测人脸关键点（眼睛鼻子嘴角），最后将这些关键点坐标存储在字典中并返回
def whole_face_block_coordinates():
    # df = pandas.read_csv('combined_3_class2_for_optical_flow.csv')
    file_path = r"D:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\combined_3_class2_for_optical_flow.csv"
    df = pandas.read_csv(file_path) # 读取 CSV 文件 
    m, n = df.shape
    # base_data_src = './datasets/combined_datasets_whole'    # 原始图像的路径 # 原版
    # base_data_src = r"d:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\datasets\combined_datasets_whole'    # 原始图像的路径
    base_data_src = Path.cwd() /"HTnet" / "datasets" / "combined_datasets_whole" # 新加了/"HTnet" 
    total_emotion = 0   # 情感总数
    image_size_u_v = 28 # 图像大小
    
    
    # get the block center coordinates  
    face_block_coordinates = {} # 存储每个图像的人脸关键点坐标

    for i in range(0, m):
    # for i in range(0, 10):
        image_name = str(df['sub'][i]) + '_' + str(
            df['filename_o'][i]) + ' .png'
        print(f"whole_face_block_coordinates 生成的 image_name: {image_name}")  # 打印生成的文件名
        
        # print(image_name) 根据 CSV 文件中的 sub 和 filename_o 列生成图像文件名。
        # img_path_apex = base_data_src + '/' + df['imagename'][i] # 原版
        
         # 使用 pathlib 拼接路径
        img_path_apex = base_data_src / df['imagename'][i]
        # 检查文件是否存在
        if not os.path.exists(img_path_apex):
            print(f"文件不存在: {img_path_apex}")
            continue
      # 将 Path 对象转换为字符串
        train_face_image_apex = cv2.imread(str(img_path_apex)) 
        # train_face_image_apex = cv2.imread(img_path_apex) # (444, 533, 3) 原版
        

        
        # 检查图像是否成功读取
        if train_face_image_apex is None:
            print(f"图像读取失败: {img_path_apex}")
            continue
        
                
        face_apex = cv2.resize(train_face_image_apex, (28,28), interpolation=cv2.INTER_AREA)    # (28, 28, 3)
        # get face and bounding box 初始化人脸检测模型，边距为0
        # mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')#原版
        mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cpu')
        
        batch_boxes, _, batch_landmarks = mtcnn.detect(face_apex, landmarks=True)
        # mtcnn.detect：对调整后的图像进行人脸检测，返回人脸边界框、人脸置信度和人脸关键点坐标。
        # face_apex输入图像，要检测的图像，landmarks=True表示返回人脸关键点坐标，false表示只返回边框
        # batch_boxes: 人脸边界框坐标，左上角和右下角的坐标
        # _置信度分数，不需要，占位符
        # batch_landmarks: 人脸关键点坐标，包含眼睛、鼻子和嘴角的坐标
        # print(img_path_apex,batch_landmarks)
        
        
        # if not detecting face 没有检测到人来拿，使用预设的关键点坐标[28,28,3]，左上角为原点
        if batch_landmarks is None:
            # print( df['imagename'][i])
            batch_landmarks = np.array([[[9.528073, 11.062551]
                                            , [21.396168, 10.919773]
                                            , [15.380184, 17.380562]
                                            , [10.255435, 22.121233]
                                            , [20.583706, 22.25584]]])
            # print(img_path_apex)
            
        # 限制关键点坐标范围，否则后面14*14会越界    
        row_n, col_n = np.shape(batch_landmarks[0])
        # print(batch_landmarks[0])
        for i in range(0, row_n):
            for j in range(0, col_n):
                if batch_landmarks[0][i][j] < 7:
                    batch_landmarks[0][i][j] = 7
                if batch_landmarks[0][i][j] > 21:
                    batch_landmarks[0][i][j] = 21
        batch_landmarks = batch_landmarks.astype(int)
        # print(batch_landmarks[0])
        
        # get the block center coordinates 储存关键点坐标并返回结果，键为图像名称，值为关键点坐标
        face_block_coordinates[image_name] = batch_landmarks[0]
    # print(len(face_block_coordinates))
    return face_block_coordinates

# 2. crop the 28*28-> 14*14 according to i5 image centers 根据关键点坐标裁剪光流图像
def crop_optical_flow_block():
    face_block_coordinates_dict = whole_face_block_coordinates()
    # print(len(face_block_coordinates_dict))
    
    # Get train dataset获得训练集光流图像路径，遍历每个图像，根据关键点坐标裁剪光流图像并存储在字典中，
    # whole_optical_flow_path = './datasets/STSNet_whole_norm_u_v_os'
    whole_optical_flow_path = r'd:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\datasets\STSNet_whole_norm_u_v_os'
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)
    four_parts_optical_flow_imgs = {}
    
    
    # print(whole_optical_flow_imgs[0]) #spNO.189_f_150.png
    for n_img in whole_optical_flow_imgs:
        
        clean_n_img = n_img.replace(' ', ' ')
        if clean_n_img not in face_block_coordinates_dict:
            print(f"未找到对应的关键点坐标: {clean_n_img}，原始文件名: {n_img}")
            continue
        
        
        four_parts_optical_flow_imgs[n_img]=[]
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        four_part_coordinates = face_block_coordinates_dict[n_img]
        # 根据键 n_img 取出对应的值，并将其赋值给变量 four_part_coordinates，通常[0]左眼[1]左嘴唇[2]鼻子[3]右眼[4]右嘴唇
        
    
        
        # 左眼 左嘴唇 通过 OpenCV 对光流图像进行裁剪操作得到的，图像切片操作，返回原数组的一个视图
        # numpy.ndarray类型变量，左眼区域的像素数据
        l_eye = flow_image[four_part_coordinates[0][0]-7:four_part_coordinates[0][0]+7,
                four_part_coordinates[0][1]-7: four_part_coordinates[0][1]+7]
        l_lips = flow_image[four_part_coordinates[1][0] - 7:four_part_coordinates[1][0] + 7,
                four_part_coordinates[1][1] - 7: four_part_coordinates[1][1] + 7]
        
         # 从光流图像中裁剪出鼻子对应的区域，以关键点坐标为中心，向上下左右各扩展7个像素
        nose = flow_image[four_part_coordinates[2][0] - 7:four_part_coordinates[2][0] + 7,
                four_part_coordinates[2][1] - 7: four_part_coordinates[2][1] + 7]
        
        # 右眼 右嘴唇
        r_eye = flow_image[four_part_coordinates[3][0] - 7:four_part_coordinates[3][0] + 7,
                four_part_coordinates[3][1] - 7: four_part_coordinates[3][1] + 7]
        r_lips = flow_image[four_part_coordinates[4][0] - 7:four_part_coordinates[4][0] + 7,
                four_part_coordinates[4][1] - 7: four_part_coordinates[4][1] + 7]
        
        
        four_parts_optical_flow_imgs[n_img].append(l_eye)
        four_parts_optical_flow_imgs[n_img].append(l_lips)
        four_parts_optical_flow_imgs[n_img].append(nose)
        four_parts_optical_flow_imgs[n_img].append(r_eye)
        four_parts_optical_flow_imgs[n_img].append(r_lips)
        # print(np.shape(l_eye)) # 打印左眼裁剪图像的形状
    
    # 提示可以打印指定图像的第一个裁剪图像的形状，结果为(14,14,3)
    # print((four_parts_optical_flow_imgs['spNO.189_f_150.png'][0]))->(14,14,3)
    print(len(four_parts_optical_flow_imgs))
    return four_parts_optical_flow_imgs


# 定义了一个融合模型 Fusionmodel，继承自 nn.Module 类，用于将多个输入特征进行融合和处理。神经网络模型类

class Fusionmodel(nn.Module):
  def __init__(self):
    #  extend from original
    super(Fusionmodel,self).__init__()
    self.fc1 = nn.Linear(15, 3) # 15->3 fully connected layer 特征从15维映射到3维，一共五个部位，每个三维
    self.bn1 = nn.BatchNorm1d(3) # 批归一化层，用于加速训练过程
    self.d1 = nn.Dropout(p=0.5) # dropout层，防止过拟合
    # Linear 256 to 26
    self.fc_2 = nn.Linear(6, 2) # 6->3 第二个全连接层？
    # self.fc_cont = nn.Linear(256, 3)
    self.relu = nn.ReLU() # 激活函数

    # forward layers is to use these layers above
  def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
    #拼接五个特征，将五个特征向量拼接在一起，形成一个新的特征向量 fuse_five_features
    fuse_five_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
    
    # nn.linear - fc 通过第一个全连接层处理拼接后的特征
    fuse_out = self.fc1(fuse_five_features)
    # fuse_out = self.bn1(fuse_out) # 归一化
    fuse_out = self.relu(fuse_out) # 激活
    fuse_out = self.d1(fuse_out) # drop out # 丢弃层，防止过拟合
    
    # 沿着第0维拼接整体特征和处理后的局部特征 torch.cat拼接张量的函数，除了拼接维度之外，其他维度的大小必须相同
    # 拼接整体和局部信息，宏观态势运动模式+细节变化（眼睛的微眯、嘴唇的轻微抽动）
    fuse_whole_five_parts = torch.cat(
        (whole_feature,fuse_out), 0)
    # fuse_whole_five_parts = self.bn1(fuse_whole_five_parts)
    fuse_whole_five_parts = self.relu(fuse_whole_five_parts)
    fuse_whole_five_parts = self.d1(fuse_whole_five_parts)  # drop out
    out = self.fc_2(fuse_whole_five_parts)
    return out

def main(config):
    learning_rate = 0.00005
    batch_size = 256
    epochs = 800 # 原版
    epochs = 50
    all_accuracy_dict = {}
    is_cuda = torch.cuda.is_available()# 原版
    print('is_cuda:', is_cuda)
    # is_cuda = False 
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    loss_fn = nn.CrossEntropyLoss() # 交叉熵损失
    if (config.train):
        if not path.exists('ourmodel_threedatasets_weights'):
            os.mkdir('ourmodel_threedatasets_weights')
    # 保存训练中的模型权重
    print('lr=%f, epochs=%d, device=%s\n' % (learning_rate, epochs, device))
    # 学习率、训练轮数和计算设备
    
    total_gt = [] # 存储真实标签
    total_pred = [] # 存储预测标签
    best_total_pred = []    # 存储最佳预测标签

    t = time.time() # 记录开始时间

    # main_path = '../datasets/three_norm_u_v_os' #数据集主路径
    main_path = r"d:\思扬\软件学院\三下\微表情识别\Micro-expression-recognition\HTNet\datasets\three_norm_u_v_os"
    subName = os.listdir(main_path)
    all_five_parts_optical_flow = crop_optical_flow_block() # 裁剪光流图像
    print(subName) # 打印子文件夹名称

    for n_subName in subName:
        print('Subject:', n_subName)
        y_train = []
        y_test = []
        four_parts_train = []
        four_parts_test = []
        
        # Get train dataset 训练集
        expression = os.listdir(main_path + '/' + n_subName + '/u_train')
        for n_expression in expression:
            
            # 当前表情类别目录下的所有图像文件
            img = os.listdir(main_path + '/' + n_subName + '/u_train/' + n_expression)

            for n_img in img:
                # 将当前图像对应的表情类别（目录名）转换为整数并添加到训练集标签列表中
                y_train.append(int(n_expression)) 
                
                # 拼接左边 拼接右边 拼接全部，添加到训练集图像中
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips  =  cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                four_parts_train.append(lr_eye_lips)


        # Get test dataset 获得测试集，和获得训练集同理
        expression = os.listdir(main_path + '/' + n_subName + '/u_test')
        for n_expression in expression:
            # 当前表情类别目录下的所有图像文件
            img = os.listdir(main_path + '/' + n_subName + '/u_test/' + n_expression)

            for n_img in img: 
                y_test.append(int(n_expression)) # 对每一个图像，将其对应的表情类别（目录名）转换为整数并添加到测试集标签列表中
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                four_parts_test.append(lr_eye_lips)
                
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
            model.load_state_dict(torch.load(weight_path)) # 如果不处于训练模式，从指定路径加载预训练的模型权重
        
        # 优化器初始化与数据处理
        # 定义了一个优化器，用于更新模型的参数。这里使用的是Adam优化器，学习率为learning_rate。
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        y_train = torch.Tensor(y_train).to(dtype=torch.long)
        
        # 将标签转换为长整型张量+.permute(0, 3, 1, 2)：对张量的维度进行重排。在大多数深度学习框架中，图像数据的常见格式是 (batch_size, channels, height, width)，而 OpenCV 读取的图像数据格式通常是 (height, width, channels)。通过 permute(0, 3, 1, 2) 可以将数据格式从 (batch_size, height, width, channels) 转换为 (batch_size, channels, height, width)，以符合模型输入的要求。
        four_parts_train =  torch.Tensor(np.array(four_parts_train)).permute(0, 3, 1, 2)
        dataset_train = TensorDataset(four_parts_train, y_train) #训练集
        train_dl = DataLoader(dataset_train, batch_size=batch_size) #按批次加载数据
        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        four_parts_test = torch.Tensor(np.array(four_parts_test)).permute(0, 3, 1, 2)
        dataset_test = TensorDataset(four_parts_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size)
        
        
        # store best results 储存最佳结果
        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        for epoch in range(1, epochs + 1): # 训练轮数
            if (config.train):
                # Training 训练模式
                model.train()
                train_loss = 0.0 # 训练损失
                num_train_correct = 0 # 训练正确的样本数
                num_train_examples = 0 # 训练样本总数

                print(f"Subject: {n_subName}, Epoch {epoch}/{epochs} - Training started")
                # for batch in train_dl: # 原版
                for batch_idx, batch in enumerate(test_dl):
                    optimizer.zero_grad() # 清空梯度
                    x = batch[0].to(device) # 将输入数据移动到指定的设备（GPU 或 CPU）上进行计算
                    y = batch[1].to(device) 
                    yhat = model(x) # 输入数据传入模型，得到预测输出
                    loss = loss_fn(yhat, y) # 计算损失函数
                    loss.backward() # 反向传播计算梯度
                    optimizer.step() # 更新模型参数

                    train_loss += loss.data.item() * x.size(0) #累加当前批次的损失，乘以批次大小
                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item() # 计算当前批次的正确预测数量
                    # torch.max(yhat, 1)[1]返回每一行的最大值的索引，返回一个元组，[1]是预测的标签
                    # sum是当前批次中预测的正确数量，item()将其转换为Python整数
                    num_train_examples += x.shape[0]    #  计算当前批次的样本数量

                    # 训练提示
                    if batch_idx % 10 == 0:  # 每 10 个批次输出一次信息
                        batch_train_loss = loss.data.item()
                        batch_train_acc = (torch.max(yhat, 1)[1] == y).sum().item() / x.shape[0]
                        print(f"  Batch {batch_idx}/{len(train_dl)} - Train Loss: {batch_train_loss:.4f}, Train Acc: {batch_train_acc:.4f}")
            
                train_acc = num_train_correct / num_train_examples # 训练准确率
                train_loss = train_loss / len(train_dl.dataset) # 训练损失

            # Testing 评估模式
            model.eval()
            val_loss = 0.0
            num_val_correct = 0
            num_val_examples = 0
            
            for batch in test_dl:
                x = batch[0].to(device)
                y = batch[1].to(device)
                yhat = model(x)
                loss = loss_fn(yhat, y)
                val_loss += loss.data.item() * x.size(0)
                num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                num_val_examples += y.shape[0]

            val_acc = num_val_correct / num_val_examples
            val_loss = val_loss / len(test_dl.dataset)
            
            #### best result 这段代码的主要功能是在模型训练的每个轮次结束后，比较当前轮次的测试准确率 val_acc 与之前记录的该受试者的最佳准确率 best_accuracy_for_each_subject。如果当前测试准确率更高，则更新最佳准确率，并保存对应的预测结果。若处于训练模式，还会保存当前表现最佳的模型权重。
            temp_best_each_subject_pred = []
            if best_accuracy_for_each_subject <= val_acc:
                best_accuracy_for_each_subject = val_acc # 最佳准确率
                temp_best_each_subject_pred.extend(torch.max(yhat, 1)[1].tolist()) # .tolist()：将 PyTorch 张量转换为 Python 列表。
                best_each_subject_pred = temp_best_each_subject_pred
                # Save Weights
                if (config.train):
                    torch.save(model.state_dict(), weight_path)

        # For UF1 and UAR computation
        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred # 最佳预测结果
        accuracydict['truth'] = y.tolist()  # 真是标签转换成列表
        all_accuracy_dict[n_subName] = accuracydict

        print('Ground Truth :', y.tolist())
        print('Evaluation until this subject: ')
        total_pred.extend(torch.max(yhat, 1)[1].tolist()) # 将当前批次的模型预测结果添加到总的预测结果列表中
        total_gt.extend(y.tolist()) # 将当前批次的真实标签添加到总的真实标签列表中
        best_total_pred.extend(best_each_subject_pred)# 最佳预测结果添加到总的最佳预测结果列表中
        
        # 评估结果
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    print(np.shape(total_gt))
    print('Total Time Taken:', time.time() - t)
    print(all_accuracy_dict) # 每个预测和真实信息的字典


if __name__ == '__main__':
    # get_whole_u_v_os()
    # create_norm_u_v_os_train_test()
    parser = argparse.ArgumentParser() # 解析命令行参数
    # input parameters
    parser.add_argument('--train', type=strtobool, default=False)  # Train or use pre-trained weight for prediction
    # --train参数用于指定是否进行训练，类型为布尔值，默认值为False
    # 训练模式为True，使用预训练权重进行预测模式为False
    config = parser.parse_args() # 解析命令行参数，存入config里
    main(config)
