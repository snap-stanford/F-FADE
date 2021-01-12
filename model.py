import torch
import torch.nn as nn
import torch.nn.functional as FF
import numpy as np
import torch.optim as optim
from scipy.stats import f as f_dis
import random
from itertools import product


class Model(nn.Module):
  def __init__(self, num_nodes, embedding_size):
    super(Model, self).__init__()
    h_initial = FF.normalize(torch.rand(num_nodes + 1, embedding_size), dim = 1)
    self.h_static = nn.Parameter(h_initial)
    self.embedding_size = embedding_size
    self.num_nodes = num_nodes
    self.Q = nn.Parameter(torch.rand(embedding_size, embedding_size))
    self.Q.requires_grad_(False)

  def forward(self, s_id, d_id):
    h_s = self.h_static[s_id, :]
    h_d = self.h_static[d_id, :]
    lambda_sd = torch.exp((torch.matmul(h_s, self.Q) * h_d).sum(dim = -1))
    return lambda_sd

def UNION(F, Act_S, delta_F, alpha, M, t_0, f_th):
  for (s, d), (t, f) in delta_F.items():
    Act_S.add((s, d))
    if (s, d) in F:
      t_, f_ = F[(s, d)]
      F[(s, d)] = (t, alpha ** (t - t_) * f_ + (1 - alpha) * f)
    else:
      F[(s, d)] = (t, (1 - alpha) * f)
  # Step 6
  if M > 0:
    F_ = {k:v for k, v in sorted(F.items(), key = lambda item: -1 * alpha ** (t_0 - item[1][0]) * item[1][1])}
    F_value_list = list(F_.values())
    if len(F_value_list) >= M:
      f_th = alpha ** (t_0 - F_value_list[M - 1][0]) * F_value_list[M - 1][1]
    # Step 7 ~ Step 9
  F_ = F.copy()
  for (s, d), (t, f) in F.items():
    if alpha ** (t_0 - t) * f < f_th:
      del F_[(s, d)]
      Act_S.discard((s, d))
  F = F_

  return F, Act_S, f_th

def DETECT(model, F, delta_F, H, f_th, device):
  Sc_list = list()
  Xi_out = dict()
  Xi_in = dict()
  for (s, d) in delta_F.keys():
    # Setup Xi_out[s]
    if s in Xi_out:
      Xi_out[s].add((s, d))
    else:
      Xi_out[s] = set()
      Xi_out[s].add((s, d))
    # Setup Xi_in[d]
    if d in Xi_in:
      Xi_in[d].add((s, d))
    else:
      Xi_in[d] = set()
      Xi_in[d].add((s, d))

  s_set = set()
  d_set = set()
  for (s, d), (t, f) in delta_F.items():
    #print(s, d)
    s_set.add(s)
    d_set.add(d)
    with torch.no_grad():
      # Compute f_sd
      f_sd_first = 0.0
      f_sd_others = 0.0
      if (s, d) not in F:
        f_sd_first = f_th
      else:
        t_prime = F[(s, d)][0]
        f_sd_first = 1 / (t - t_prime - 1 + (1 / f))
      if f > 1:
        f_sd_others = f

      # Compute lambda_sd and Sc_sd
      lambda_sd = 0.0
      Sc_sd_first = 0.0
      Sc_sd_others = 0.0
      if s in H and d in H:
        source_ids = torch.from_numpy(np.array([s], dtype = np.int)).to(device)
        target_ids = torch.from_numpy(np.array([d], dtype = np.int)).to(device)
        lambda_sd = model(source_ids, target_ids).item()
      else:
        lambda_sd = f_th
      Sc_sd_first = 1 * f_sd_first / lambda_sd
      if f > 1:
        Sc_sd_others = 1 * f_sd_others / lambda_sd

      # Compute f_Xi_out_s
      f_Xi_out_s_first = 0.0
      f_Xi_out_s_others = 0.0
      f_Xi_out_s_total = 0
      t_Xi_out_s_prime = 0
      in_F = False
      for (s_, d_) in Xi_out[s]:
        f_Xi_out_s_total += delta_F[(s_, d_)][1]
        if (s_, d_) in F:
          in_F = True
          t_Xi_out_s_prime = max(t_Xi_out_s_prime, F[(s_, d_)][0])

      if s in s_set:
        f_Xi_out_s_first = f_Xi_out_s_total
        if f > 1:
          f_Xi_out_s_others = f_Xi_out_s_total
      else:
        if in_F == False:
          if s not in s_set:
            f_Xi_out_s_first = f_th
          else:
            f_Xi_out_s_first = f_Xi_out_s_total
        else:
          if s not in s_set:
            f_Xi_out_s_first = 1 / (t - t_Xi_out_s_prime - 1 + (1 / f_Xi_out_s_total))
          else:
            f_Xi_out_s_first = f_Xi_out_s_total

      if f > 1:
        f_Xi_out_s_others = f_Xi_out_s_total

      # Comput f_Xi_in_d
      f_Xi_in_d_first = 0.0
      f_Xi_in_d_others = 0.0
      f_Xi_in_d_total = 0
      t_Xi_in_d_prime = 0
      in_F = False
      for (s_, d_) in Xi_in[d]:
        f_Xi_in_d_total += delta_F[(s_, d_)][1]
        if (s_, d_) in F:
          in_F = True
          t_Xi_in_d_prime = max(t_Xi_in_d_prime, F[(s_, d_)][0])

      if d in d_set:
        f_Xi_in_d_first = f_Xi_in_d_total
        if f > 1:
          f_Xi_in_d_others = f_Xi_in_d_total
      else:
        if in_F == False:
          if d not in d_set:
            f_Xi_in_d_first = f_th
          else:
            f_Xi_in_d_first = f_Xi_in_d_total
        else:
          if d not in d_set:
            f_Xi_in_d_first = 1 / (t - t_Xi_in_d_prime - 1 + (1 / f_Xi_in_d_total))
          else:
            f_Xi_in_d_first = f_Xi_in_d_total

      if f > 1:
        f_Xi_in_d_others = f_Xi_in_d_total

      # Compute lambda_Xi_out_s and Sc_Xi_out_s
      lambda_Xi_out_s = 0.0
      Sc_Xi_out_s_first = 0.0
      Sc_Xi_out_s_others = 0.0
      for (s_, d_) in Xi_out[s]:
        if s_ in H and d_ in H:
          source_ids = torch.from_numpy(np.array([s_], dtype = np.int)).to(device)
          target_ids = torch.from_numpy(np.array([d_], dtype = np.int)).to(device)
          lambda_Xi_out_s += model(source_ids, target_ids).item()
        else:
          lambda_Xi_out_s += f_th
      Sc_Xi_out_s_first = 1 * f_Xi_out_s_first / lambda_Xi_out_s
      if f > 1:
        Sc_Xi_out_s_others = 1 * f_Xi_out_s_others / lambda_Xi_out_s

      # Compute lambda_Xi_in_d and Sc_Xi_in_d
      lambda_Xi_in_d = 0.0
      Sc_Xi_in_d_first = 0.0
      Sc_Xi_in_d_others = 0.0
      for (s_, d_) in Xi_in[d]:
        if s_ in H and d_ in H:
          source_ids = torch.from_numpy(np.array([s_], dtype = np.int)).to(device)
          target_ids = torch.from_numpy(np.array([d_], dtype = np.int)).to(device)
          lambda_Xi_in_d += model(source_ids, target_ids).item()
        else:
          lambda_Xi_in_d += f_th
      Sc_Xi_in_d_first = 1 * f_Xi_in_d_first / lambda_Xi_in_d
      if f > 1:
        Sc_Xi_in_d_others = 1 * f_Xi_in_d_others / lambda_Xi_in_d

      Sc_first = max(Sc_sd_first, Sc_Xi_out_s_first, Sc_Xi_in_d_first)
      Sc_others = max(Sc_sd_others, Sc_Xi_out_s_others, Sc_Xi_in_d_others)
      for i in range(f):
        if i == 0:
          Sc_list.append(Sc_first)
        else:
          Sc_list.append(Sc_others)

  return Sc_list

def F_FAC(model, F, Act_S, H, V, f_th, epochs, max_batch_size, flag, device):
  # Step 1
  H_ = H.copy()
  for v in H:
    if v not in V:
      H_.discard(v)

  # Step 2
  for v in V:
    if v not in H:
      H_.add(v)
      h_tmp = FF.normalize(torch.rand(1, model.embedding_size), dim = 1).view(-1)
      model.h_static[v].data.copy_(nn.Parameter(h_tmp))

  H = H_

  # Step 3 ~ Step 10
  if flag == "global":
    optimizer = optim.RMSprop([model.h_static], lr = 1e-3)
    # Setup F_c
    V_square = list(product(list(V), list(V)))
    for (s, d) in F.keys():
      V_square.remove((s, d))
    F_c = V_square
    # Setup training data structure
    train_data = []
    for (s, d), (t, f) in F.items():
      train_data.append((s, d, f))
    for (s, d) in F_c:
      if s == d:
        continue
      train_data.append((s, d, f_th))
    # Start training
    train_num = len(train_data)
    for epoch in range(1, epochs + 1):
      loss_epoch = 0.0
      random.shuffle(train_data)
      batch_s = 0
      while batch_s < train_num:
        optimizer.zero_grad()
        batch_t = min(batch_s + max_batch_size, train_num)
        batch = train_data[batch_s: batch_t]
        batch_size = len(batch)
        batch_source_ids_np = np.array([x[0] for x in batch], dtype = np.int)
        batch_source_ids = torch.from_numpy(batch_source_ids_np).to(device = device)
        batch_target_ids_np = np.array([x[1] for x in batch], dtype = np.int)
        batch_target_ids = torch.from_numpy(batch_target_ids_np).to(device = device)
        batch_freq = torch.from_numpy(np.array([x[2] for x in batch], dtype = np.float32)).to(device).view(batch_size, 1)
        batch_lambda_sd = model(batch_source_ids, batch_target_ids)
        batch_loss = (torch.log(batch_lambda_sd) + batch_freq / batch_lambda_sd).sum() / batch_size
        batch_loss.backward()
        optimizer.step()
        loss_epoch += batch_loss.item()
        batch_s = batch_t
      print("Epoch {} | Total Loss {}".format(epoch, loss_epoch))

  elif flag == "local":
    optimizer = optim.RMSprop([model.h_static, ], lr = 2e-3)
    # Setup V_Omega_p
    V_Omega_p = set()
    for (s, d) in Act_S:
      V_Omega_p.add(s)
      V_Omega_p.add(d)
    # Setup V_prime
    V_prime = set()
    for s in V:
      if s not in V_Omega_p:
        V_prime.add(s)

    for epoch in range(1, epochs + 1):
      V_Omega_p_tmp = set()
      for i in range(max_batch_size):
        if len(V_Omega_p) > 0:
          s = random.choice(list(V_Omega_p))
          V_Omega_p_tmp.add(s)
      for i in range(int(4 * max_batch_size)):
        if len(V_prime) > 0:
          s = random.choice(list(V_prime))
          V_Omega_p_tmp.add(s)
      V_square = list(product(list(V_Omega_p_tmp), list(V_Omega_p_tmp)))
      train_data = []
      for (s, d) in V_square:
        if s == d:
          continue
        if (s, d) in F:
          train_data.append((s, d, F[(s, d)][1]))
        else:
          train_data.append((s, d, f_th))
      optimizer.zero_grad()
      batch_size = len(train_data)
      batch_source_ids_np = np.array([x[0] for x in train_data], dtype = np.int)
      batch_source_ids = torch.from_numpy(batch_source_ids_np).to(device = device)
      batch_target_ids_np = np.array([x[1] for x in train_data], dtype = np.int)
      batch_target_ids = torch.from_numpy(batch_target_ids_np).to(device = device)
      batch_freq = torch.from_numpy(np.array([x[2] for x in train_data], dtype = np.float32)).to(device).view(batch_size, 1)
      batch_lambda_sd = model(batch_source_ids, batch_target_ids)
      batch_loss = (torch.log(batch_lambda_sd) + batch_freq / batch_lambda_sd).sum() / batch_size
      batch_loss.backward()
      optimizer.step()

  return H

def F_FADE(model, dataset, t_setup, W_upd, alpha, M, T_th, epochs, online_train_steps, max_batch_size, device):
  # Step 1
  edge_stream = dataset.data
  f_th = 1 / T_th
  k = 0
  Act_S = set()
  F = dict()
  H = set()
  V = set()
  N_in = set()
  N_out = set()
  T_max = dataset.T_max
  num_edges = dataset.num_edges
  num_nodes = dataset.num_nodes
  label = dataset.label
  data_idx = 0
  Sc = list()

  # Step 2
  for t in range(1, T_max + 1):
    # Step 3
    delta_F = dict()
    while edge_stream[data_idx].curr_time == t:
      s = edge_stream[data_idx].s_id
      d = edge_stream[data_idx].d_id
      if (s, d) not in delta_F:
        delta_F[(s, d)] = (t, 1)
      else:
        t_, f_ = delta_F[(s, d)]
        delta_F[(s, d)] = (t, f_ + 1)
      data_idx += 1
      if data_idx == num_edges:
        break

    # Step 4
    if t > t_setup:
      Sc += DETECT(model, F, delta_F, H, f_th, device)

    # Step 5
    F, Act_S, f_th = UNION(F, Act_S, delta_F, alpha, M, t, f_th)

    # Step 6 ~ Step 9
    if t == (t_setup + k * W_upd):
      # Setup V(F)
      V = set()
      for (s, d) in F.keys():
        V.add(s)
        V.add(d)
      if k == 0:
        H = F_FAC(model, F, Act_S, H, V, f_th, epochs, max_batch_size, "global", device)
      else:
        H = F_FAC(model, F, Act_S, H, V, f_th, online_train_steps, max_batch_size, "local", device)
      k += 1
      Act_S = set()

  return Sc
