import torch
from tensorboardX import SummaryWriter # for logging training information
from tqdm import tqdm # for displaying progress bars during training
from metrics import EvalMetrics # import accuracy, recall and F1
from utils import save_checkpoint, use_optimizer
import pandas as pd
import os


class Engine(object):
    """Meta Engine for Training and Evaluating the model"""

    def __init__(self, config):
        self.config = config # storing current training config
        self._evaluate = EvalMetrics()
        self._writer = SummaryWriter(log_dir="/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/runs/{}".format(config["alias"])) 
        self._writer.add_text("config", str(config), 0)
        self.opt = use_optimizer(self.model, config) # set up optimizer 
        self.crit = torch.nn.BCELoss() # defining loss function as Binary cross entropy, used for binary classification tasks

    # for a single batch that is passed to the function, move tensors to GPU, perform one training step, clip grads, and return loss
    def train_single_batch(self, users, items, rel_int, interest, embds_path):
        if self.config["use_cuda"] is True: # move all tensors onto GPU
            users, items, rel_int, interest = (
                users.cuda(),
                items.cuda(),
                rel_int.cuda(),
                interest.cuda(),
            )
        self.opt.zero_grad() # clear grads of tensors 
        rating_pred, _ = self.model(users, items, rel_int, embds_path=embds_path) # forward pass to get interest
        loss = self.crit(rating_pred.view(-1), interest) # compare predicted with actual interest values

        loss.backward() # grad backprop

        # clip gradient to avoid exploding grads, scales down too large grads, which leads to unstable training or divergence
        clip_grad = 3.0
        for _, p in self.model.named_parameters():
            if p.grad is not None: # if there are gradients computed for that model parameter, clip them
                p.grad.data = torch.nan_to_num(p.grad.data) # replace all nan's with 0
                param_norm = p.grad.data.norm(2) # calculate L2 norm of the param's grads -> L2 regularization
                # 1e-6 helps against divisions by 0
                # if norm of param exceeds threshold of 3, then division leads to clip_coef < 1
                clip_coef = clip_grad / (param_norm + 1e-6)
                # gradients with L2 norm exceeding threshold will be scaled down
                if clip_coef < 1: 
                    p.grad.data.mul_(clip_coef) # grads are scaled down by multiplying them with a number smaller than 1


        self.opt.step() # update params
        loss = loss.item() # converts loss value to python scalar and return it
        return loss

    def train_an_epoch(self, train_loader, epoch_id, embds_path): # train all the batches for one epoch
        self.model.train() # put model into training mode, enabling grad computation
        total_loss = 0
        for batch_id, batch in tqdm(enumerate(train_loader),total=len(train_loader)):
            assert isinstance(batch[0], torch.LongTensor) # WHY
            user, item, rel_int, interest = batch[0], batch[1], batch[2], batch[3]
            interest = interest.float()
            loss = self.train_single_batch(user, item, rel_int, interest, embds_path=embds_path)
            #print("[epoch {}] batch {}, loss: {}".format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar("model/loss", total_loss, epoch_id) # log total loss for one whole epoch

    def evaluate(self, eval_data, epoch_id, embds_path):
        self.model.eval()
        with torch.no_grad(): 
            # move all test data to GPU
            test_users, test_items, test_rel_int, test_y = (
                eval_data[0],
                eval_data[1],
                eval_data[2],
                eval_data[3],
            )
            if self.config["use_cuda"] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                test_rel_int = test_rel_int.cuda()
                test_y = test_y.cuda()
            #print(test_users.size())
            #print(test_items.size())
            #print(test_rel_int.size())
            #print(embds_path)
            #print(test_rel_int)
            test_scores, distance = self.model(test_users, test_items, test_rel_int, embds_path) #forward pass with test set to get interest scores

            if self.config["use_cuda"] is False: # move to cpu if cuda not available
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_rel_int = test_rel_int.cpu()
                test_scores = test_scores.cpu()
                test_y = test_y.cpu()

            self._evaluate.subjects = [
                test_users.data.view(-1).tolist(),
                test_items.data.view(-1).tolist(),
                test_scores.data.view(-1).tolist(),
                test_y.data.view(-1).tolist(),
            ]

            # calculate accuracy, recall, f1 for all tensors
            accuracy, recall, f1, bacc = (
                self._evaluate.cal_acc(),
                self._evaluate.cal_recall(),
                self._evaluate.cal_f1(),
                self._evaluate.cal_balanced_acc(),
            )
            # log metrics in the tensorboard writer
            self._writer.add_scalar("performance/ACC", accuracy, epoch_id)
            self._writer.add_scalar("performance/RECALL", recall, epoch_id)
            self._writer.add_scalar("performance/F1", f1, epoch_id)
            self._writer.add_scalar("performance/B_ACC", bacc, epoch_id)

            print(
                "[Evaluating Epoch {}] ACC = {:.4f}, B_ACC = {:.4f}, RECALL = {:.4f}, F1 = {:.4f}".format(
                    epoch_id, accuracy, bacc, recall, f1
                )
            )
            return accuracy, recall, f1, bacc

    def save(self, alias, epoch_id: int, f1, param_str: str, metric_str: str, embds_path):
        if epoch_id == self.config["num_epoch"]-1: # save result of last epoch as trained model
          model_dir = self.config["model_dir"].format(alias, epoch_id, f1)
          print('Saving final model to: ', model_dir)
          save_checkpoint(self.model, model_dir)

          # write model data (alias, params, metrics) to table
          new_row = {
              'model_name': [alias],
              'item_embds': [os.path.basename(embds_path).split('.')[0]] if embds_path != '' else ['ex2vec'],
              'params': [param_str],
              'results': [metric_str]
          }

          new_row_df = pd.DataFrame(new_row)
          log_path = '/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/results/best_models.csv'
          
          if os.path.exists(log_path): # append contents without header
              new_row_df.to_csv(log_path, mode='a', header=False, index=False)
          else: # create header and then append contents
              new_row_df.to_csv(log_path, mode='w', header=True, index=False)

          print('Final model saved.')
        else:
            if (epoch_id + 1) % 10 == 0: # save model at every 10th epoch
                print('Saving model at epoch ', (epoch_id+1))
                chckpt_dir = self.config["chckpt_dir"].format(alias, epoch_id, f1)
                print('Saving model to ', chckpt_dir)
                save_checkpoint(self.model, chckpt_dir)

