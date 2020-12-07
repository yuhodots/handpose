from model import CPM2DPose
import torch
import numpy as np
import torchvision
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2

from glob import glob
import pandas

device = 'cuda:0'
num_joints = 21

class ObmanDataset(Dataset):
    def __init__(self, method=None):
        self.root = './' #Change this path
        self.x_data = []
        self.y_data = []
        if method == 'train':
            self.root = self.root + 'dataset/train/'
            self.img_path = sorted(glob(self.root + 'rgb/*.jpg'))

        elif method == 'test':
            self.root = self.root + 'dataset/test/'
            self.img_path = sorted(glob(self.root + 'rgb/*.jpg'))

        for i in tqdm.tqdm(range(len(self.img_path))):
            img = cv2.imread(self.img_path[i], cv2.IMREAD_COLOR)
            # print(self.img_path[i])
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            self.x_data.append(img)

            num = self.img_path[i].split('.')[1].split('/')[-1]
            img_pkl = self.root + 'meta/' + str(num) + '.pkl'
            pkl = pandas.read_pickle(img_pkl)
            coords_2d = pkl['coords_2d']
            self.y_data.append(coords_2d)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        transform1 = torchvision.transforms.ToTensor()
        new_x_data = transform1(self.x_data[idx])

        return new_x_data, self.y_data[idx]


class Trainer(object):
    def __init__(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self._build_model()

        dataset = ObmanDataset(method='train')
        self.root = './'
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Load of pretrained_weight file
        weight_root = self.root.split('/')
        del weight_root[-2]
        weight_root = "/".join(weight_root)
        weight_PATH = weight_root + 'pretrained_weight.pth'
        self.poseNet.load_state_dict(torch.load(weight_PATH))

        print("Training...")

    def _build_model(self):
        # 2d pose estimator
        poseNet = CPM2DPose()
        self.poseNet = poseNet.to(device)
        self.poseNet.train()

        print('Finish build model.')

    def skeleton2heatmap(self, _heatmap, keypoint_targets):
        heatmap_gt = torch.zeros_like(_heatmap, device=_heatmap.device)

        keypoint_targets = (((keypoint_targets)) // 8)
        for i in range(keypoint_targets.shape[0]):
            for j in range(21):
                x = int(keypoint_targets[i, j, 0])
                y = int(keypoint_targets[i, j, 1])
                heatmap_gt[i, j, x, y] = 1

        heatmap_gt = heatmap_gt.detach().cpu().numpy()
        for i in range(keypoint_targets.shape[0]):
            for j in range(21):
                heatmap_gt[i, j, :, :] = cv2.GaussianBlur(heatmap_gt[i, j, :, :], ksize=(3, 3), sigmaX=2, sigmaY=2) * 9 / 1.1772
        heatmap_gt = torch.FloatTensor(heatmap_gt).to(device)
        return heatmap_gt

    def train(self):

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.poseNet.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)

        date = '201201'
        for epoch in tqdm.tqdm(range(self.epochs + 1)):
            if epoch % 20 == 0:
                torch.save(self.poseNet.state_dict(), "_".join([self.root, date, str(epoch), 'model.pth']))

            for batch_idx, samples in enumerate(self.dataloader):
                x_train, y_train = samples
                heatmapsPoseNet = self.poseNet(x_train.cuda())
                gt_heatmap = self.skeleton2heatmap(heatmapsPoseNet, y_train)

                loss = criterion(heatmapsPoseNet, gt_heatmap)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Write train result
                if batch_idx % 20 == 0:
                    with open('train_result_' + date + '.txt', 'a') as f:
                        f.write('Epoch {:4d}/{} Batch {}/{}\n'.format(
                            epoch, self.epochs, batch_idx, len(self.dataloader)
                        ))
                    print('Epoch {:4d}/{} Batch {}/{}'.format(
                        epoch, self.epochs, batch_idx, len(self.dataloader)
                    ))

        print('Finish training.')


class Tester(object):
    def __init__(self, batch_size, flag):
        self.batch_size = batch_size
        self._build_model()

        dataset = ObmanDataset(method='test')
        self.root = './'
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.datalen = dataset.__len__()
        self.mse_all_img = []

        self.flag = flag    # flag means the type of model (pretrained or fine-tuned)

        # Load of pretrained_weight file
        weight_root = self.root.split('/')
        del weight_root[-2]
        weight_root = "/".join(weight_root)
        if flag == 0:
            weight_PATH = weight_root + 'pretrained_weight.pth'     # pretrained model (problem 1)
        else:
            weight_PATH = weight_root + 'finetunedweight.pth'     # fine-tuned model (problem 3)
        self.poseNet.load_state_dict(torch.load(weight_PATH))

        print("Testing...")

    def _build_model(self):
        # 2d pose estimator
        poseNet = CPM2DPose()
        self.poseNet = poseNet.to(device)

        print('Finish build model.')

    def heatmap2skeleton(self, heatmapsPoseNet):
        skeletons = np.zeros((heatmapsPoseNet.shape[0], heatmapsPoseNet.shape[1], 2))
        for m in range(heatmapsPoseNet.shape[0]):
            for i in range(heatmapsPoseNet.shape[1]):
                u, v = np.unravel_index(np.argmax(heatmapsPoseNet[m][i]), (32, 32))
                skeletons[m, i, 0] = u * 8
                skeletons[m, i, 1] = v * 8
        return skeletons

    def test(self):
        err = 0

        for batch_idx, samples in enumerate(self.dataloader):
            x_test, y_test = samples
            heatmapsPoseNet = self.poseNet(x_test.cuda()).cpu().detach().numpy()
            skeletons_in = self.heatmap2skeleton(heatmapsPoseNet)

            """ batch_idx = 0 & heatmapsPoseNet[9] is '106.jpg'
                because option of dataloader is shuffle=False """

            if batch_idx == 0:
                self.plot_heatmap(x_test[9].numpy(), heatmapsPoseNet[9], self.flag)
                self.plot_hand(x_test[9].numpy(), skeletons_in[9], self.flag)

            for i in range(x_test.shape[0]):
                err += self.calc_error(y_test[i].numpy(), skeletons_in[i])

        avg_err = err/500

        if self.flag == 0:
            print('Average error of pretrained model = {}'.format(avg_err))     # pretrained model (problem 1)
        else:
            print('Average error of fine-tuned model = {}'.format(avg_err))     # fine-tuned model (problem 3)

    def plot_heatmap(self, img_np, heatmap_np, flag):
        img_np = 255 * np.transpose(img_np, (1, 2, 0))
        save_file = self.root + 'save/test/original_flag{}.png'.format(flag)
        cv2.imwrite(save_file, cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))   # just for checking '106.jpg'

        heatmap_np = 255 * np.transpose(heatmap_np, (1, 2, 0))
        for i in range(heatmap_np.shape[2]):
            save_file = self.root + 'save/test/heatmap_flag{0}_{1}.png'.format(flag, i + 1)
            cv2.imwrite(save_file, heatmap_np[:, :, i])

    def plot_hand(self, img_np, coords, flag):
        """ Thumb (blue), Index finger (pink), Middle finger (red), Ring finger (green), Pinky (Orange)).
            <RGB> Blue: (0,0,255), Pink: (255,0,255), Red: (255,0,0), Green: (0,255,0), Orange: (255,122,0)"""

        img_np = 255 * np.transpose(img_np, (1, 2, 0))
        color_table = [(0, 0, 255), (255, 0, 255), (255, 0, 0), (0, 255, 0), (255, 122, 0)]

        for i in range(5):  # Iteration of each five different colors
            init_joint_idx = 4*i

            for j in range(4):  # Iteration of joints which is in the same color
                joint_a = coords[init_joint_idx+j]
                if j == 0:
                    joint_a = coords[0]
                joint_b = coords[init_joint_idx+j+1]
                joints = np.array([[joint_a[1], joint_a[0]], [joint_b[1], joint_b[0]]], dtype=np.int32)

                img_np = cv2.polylines(img_np, [joints], False, color_table[i], thickness=1)

        save_file = self.root + 'save/test/skeleton_flag{}.png'.format(flag)
        cv2.imwrite(save_file, cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

    def calc_error(self, GT, Pred):
        err = 0
        for i in range(num_joints):
            err += np.sqrt(np.square(Pred[i, 0] - GT[i, 0]) + np.square(Pred[i, 1] - GT[i, 1]))
        return err/num_joints


def main():

    epochs = 100
    batchSize = 16
    learningRate = 1e-5

    """ Fine-tuning the model is required in the problem 2
        Please remove these annotations to run the code of problem 2 """
    # trainer = Trainer(epochs, batchSize, learningRate)
    # trainer.train()

    tester_pretrained = Tester(batchSize, flag=0)   # 'flag=0' means pretrained model
    tester_pretrained.test()

    tester_finetuned = Tester(batchSize, flag=1)    # 'flag=1' means fine-tuned model
    tester_finetuned.test()


if __name__ == '__main__':
    main()
