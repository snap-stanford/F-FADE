class Dataset():
  def __init__(self, file_name):
    class data_point():
      def __init__(self):
        self.s_id = None
        self.d_id = None
        self.weight = None
        self.curr_time = None


    self.data = []
    self.label = []
    self.num_nodes = 0
    self.num_edges = 0
    self.T_max = 0

    with open(file_name, 'r') as dataset:
      for LINE in dataset:
        line = LINE.split(' ')
        temp_data_point = data_point()
        temp_data_point.curr_time = float(line[0])
        temp_data_point.s_id = int(line[1])
        temp_data_point.d_id = int(line[2])
        temp_data_point.weight = 1.0
        self.data.append(temp_data_point)
        self.label.append(int(line[3]))
        self.num_nodes = max(max(temp_data_point.s_id, temp_data_point.d_id), self.num_nodes)
        self.num_edges = self.num_edges + 1
        self.T_max = int(line[0])
