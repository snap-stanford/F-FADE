from utils import *
from model import *
from sklearn import metrics
from arguments import parse_arguments
import os
import json
import numpy as np

if __name__ == '__main__':
  args = parse_arguments()
  device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

  if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)

  with open(args.model_dir + 'arg_list.txt', 'w') as file:
    json.dump(args.__dict__, file, indent=2)

  dataset = Dataset(args.dataset)

  model = Model(num_nodes = dataset.num_nodes, embedding_size = args.embedding_size).to(device = device)
  print(model.h_static)

  F_FADE = F_FADE(model, dataset, args.t_setup, args.W_upd, args.alpha, args.M, args.T_th, args.epochs, args.online_train_steps, args.batch_size, device)

  np.savetxt(args.model_dir + 'score.txt', np.array(F_FADE).reshape((-1, 1)))

  F_FADE = np.array(F_FADE)

  for i in range(len(F_FADE)):
    if np.isnan(F_FADE[i]):
      F_FADE[i] = 0

  fpr, tpr, thresholds = metrics.roc_curve(dataset.label[-len(F_FADE):], F_FADE, pos_label = 1)
  AUC = metrics.auc(fpr, tpr)
  print("AUC: {}".format(AUC))

  with open(args.model_dir + 'AUC.txt', 'w') as file:
    file.write(json.dumps(AUC))
