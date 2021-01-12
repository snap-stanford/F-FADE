import argparse

def parse_arguments():
  parser = argparse.ArgumentParser('')
  parser.add_argument('--dataset', type = str, default = './data/darpa.txt')
  parser.add_argument('--embedding_size', type = int, default = 200)
  parser.add_argument('--batch_size', type = int, default = 32)
  parser.add_argument('--t_setup', type = int, default = 8000)
  parser.add_argument('--W_upd', type = int, default = 720)
  parser.add_argument('--alpha', type = float, default = 0.999)
  parser.add_argument('--T_th', type = int, default = 120)
  parser.add_argument('--epochs', type = int, default = 5)
  parser.add_argument('--online_train_steps', type = int, default = 10)
  parser.add_argument('--gpu', type = int, default = 0)
  parser.add_argument('--M', type = int, default = 100)
  parser.add_argument('--model_dir', type = str, required = True)
  args = parser.parse_args()
  return args
