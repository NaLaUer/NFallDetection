import os
import cv2
from sklearn.model_selection import train_test_split
import shutil
import torch
import random
import  numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VideoDataset(Dataset):
    def __init__(self, args, split='train'):

        self.root_dir = args.root_dir
        self.label_dir = args.label_dir
        self.pic_dir = args.pic_dir
        self.args = args

        self.clip_len = args.clip_len

        self.resize_height = 224
        self.resize_width = 224

        # 1 判断原始video数据是否存在
        if self.check_integrity() == False:
            raise RuntimeError("Dataset is empty !")

        # 2 判断切分文件夹是否存在
        if self.check_preprocess() == True:
            print("数据集创建中，请稍等.....")
            self.preprocess()

        self.fnames = []

        for file in os.listdir(self.label_dir):
            file = os.path.join(self.label_dir, file)
            with open(file, 'r') as f:
                for fname in f:
                    self.fnames.append(fname.strip('\n'))

        self.transform = transforms.RandomHorizontalFlip()
        self.ToTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer, labels = self.load_frames(self.fnames[index])
        #   buffer = self.crop(buffer, self.clip_len)
        labels = np.array(labels)

        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def label_array(self, fname):
        type = fname.split('/')[-1].split('_')[0]
        if type == 'normal':
            return 1
        else:
            return 0

    def load_frames(self, file_info):
        file_dir = file_info.split(' ')[0]
        file_start = int(file_info.split(' ')[1])
        file_end = int(file_info.split(' ')[2])
        file_label = int(file_info.split(' ')[3])

        frames = sorted(
            [os.path.join(file_dir, img) for img in os.listdir(file_dir)])

        clip_frames = sorted(random.sample(range(file_start, file_end), self.clip_len))
        chose_frnames = [frames[i] for i in clip_frames]
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3),
                          np.dtype('float32'))

        for i, frame_name in enumerate(chose_frnames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer, file_label

    def crop(self, buffer, clip_len):

        time_index = np.random.randint(buffer.shape[0] - clip_len)
        buffer = buffer[time_index:time_index + clip_len, :, :, :]

        return buffer

    def randomflip(self, buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((0, 3, 1, 2))

    def check_integrity(self):
        # function : 判断原始数据集是否存在
        if os.path.exists(self.root_dir):
            return True
        else:
            return False

    """
        root_dir    /home/sky/PycharmProjects/data/Video
        pic_dir     /home/sky/PycharmProjects/data/Pic
        out_dir     /home/sky/PycharmProjects/data/Video_split
    """

    def check_preprocess(self):
        # function : 判断切分文件夹是否存在
        if os.path.exists(self.pic_dir):
            return False
        if os.path.exists(self.label_dir):
            return False
        return True

    def preprocess(self):
        """
        function : 数据集创建， 创建文件夹， 将Video => 图片
        格式 ：
            data
                -train
                    - fall
                    - normal
                -test
                -val
        """
        # 1 创建相关文件夹
        self.mkdir_begin()

        # 2 将所有的图片切分出来
        self.Convert_video()

        print("数据创建完毕")

    def Convert_video(self):
        # function : 将video 切换为图片
        for file_dir in os.listdir(self.root_dir):
            file_dir = os.path.join(self.root_dir, file_dir)
            # file_dir = /home/sky/PycharmProjects/data/Video/Coffee_room_02
            # 2.1 取出每个图像的摔倒帧数位置，格式为：{"video.txt" : [95, 125]}
            dic = self.Statpos_of_fall(file_dir, 'Annotation_files')
            # 2.2 保存图像，图像名字 label_video_帧数.jpg
            self.Video_to_Pic(file_dir, 'Videos', dic)

    def mkdir_begin(self):
        # function : 创建相关文件夹
        # 0 切割图片文件存储
        if not os.path.exists(self.pic_dir):
            os.mkdir(self.pic_dir)
        # 1 创建label存储文件
        if not os.path.exists(self.label_dir):
            os.mkdir(self.label_dir)
            self.train_bg = os.path.join(self.label_dir, "train_bg.txt")
            self.train_fall = os.path.join(self.label_dir, "train_fall.txt")

            train_bg = open(self.train_bg, 'a')
            train_fall = open(self.train_fall, 'a')
            train_fall.close()
            train_bg.close()

    def Statpos_of_fall(self, file_dir, video_dir):
        # function : 从 Annotation_files 文件中提取跌倒帧数的起始于终止位置
        video_dir = os.path.join(file_dir, video_dir)
        dic = {}

        for file in os.listdir(video_dir):
            annotation_dir = os.path.join(video_dir, file)
            with open(annotation_dir, 'r') as f:
                line = f.readlines()[:2]
                f.close()
            # 排除 没有帧数的表
            if len(line[0].split(',')) == 1:
                dic[file] = line

        return dic

    def Video_to_Pic(self, file_dir, video_dir, dic):

        video_dir = os.path.join(file_dir, video_dir)

        for video in os.listdir(video_dir):

            if video.split('.')[0] + '.txt' in dic:
                line = dic[video.split('.')[0] + '.txt']

                video_path = os.path.join(video_dir, video)

                index_list = [int(val.strip()) for val in line]

                if index_list[0] == index_list[1]:
                    continue

                video_path_dir = video_path.split('/')[-3]
                video_name = video.split('.')[0].replace(' ', '').replace('(', '').replace(')', '')

                save_address = os.path.join(self.pic_dir, video_path_dir + video_name)
                os.mkdir(save_address)
                temp = save_address + " " + str(index_list[0]) + " " + str(index_list[1]) + " 1"

                with open(self.train_fall, "a") as f:
                    f.writelines(temp)
                    f.write('\n')
                f.close()

                clip = index_list[1] - index_list[0]
                end = index_list[1] - clip
                start = np.random.randint(0, end)
                end = start + clip
                temp = save_address + " " + str(start) + " " + str(end) + " 0"

                with open(self.train_bg, "a") as f:
                    f.writelines(temp)
                    f.write('\n')
                f.close()

                self.Read_Video(video_path, save_address, index_list)

    def Read_Video(self, video_path, save_address, index_list):
        # function ： 读取视频文件
        capture = cv2.VideoCapture(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = 1

        while (count < frame_count):
            retaining, frame = capture.read()

            if frame is None:
                continue

            if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))

            cv2.imwrite(filename=os.path.join(save_address, '{}.jpg'.format(str(count).zfill(4))),
                        img=frame)

            count += 1


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # __file__获取执行文件相对路径，整行为取上一级的上一级目录
    sys.path.append(BASE_DIR)
    print (BASE_DIR)
    from config.args import setting_args

    parser = setting_args()
    args = parser.parse_args()

    train_data = VideoDataset(args,split='test')
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break