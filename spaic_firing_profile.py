import numpy as np
import spaic


class SpaicFiringProfile:
    '''统计 SPAIC 的 SNN 模型中每个神经元的平均发放频率'''

    def __init__(self):
        # 元素 ID 映射到 batch_sum(time_mean(counts)), 除以批数就是平均发放频率
        self.firing_counts: dict[str, np.ndarray] = {}
        self.batch_size = 0
        # 下面的变量只在运行网络时用到
        self.backend = None
        self.count_list: list[np.ndarray] = [] # 存储值
        self.id_out_list: list[str] = [] # 存储输出变量名称

    def set_net(self, net: spaic.Network):
        '''设置要运行的网络'''

        backend = net._backend
        if not backend or not backend.builded:
            raise ValueError('网络未构建。')

        firing_counts: dict[str, np.ndarray] = {}
        count_list: list[np.ndarray] = []
        id_out_list: list[str] = []

        for ele in net.get_groups():
            if isinstance(ele, spaic.Encoder):
                var = ele.coding_var_name
            elif isinstance(ele, spaic.NeuronGroup):
                var = 'O'
            else:
                continue
            id_ = ele.id
            id_out = ''.join((id_, ':{', var, '}'))
            if not backend.has_variable(id_out): # 有的神经元没有输出
                continue
            count = np.zeros((ele.num,))

            firing_counts[id_] = count
            count_list.append(count)
            id_out_list.append(id_out)

        self.firing_counts = firing_counts
        self.batch_size = 0
        self.backend = backend
        self.count_list = count_list
        self.id_out_list = id_out_list

    def run(self, time: float | int):
        '''运行一段时间，统计每个神经元的平均发放次数，可以多次运行不同批次'''

        backend = self.backend
        if not backend:
            raise Exception('未设置网络，无法运行。')

        self.batch_size += backend.get_batch_size() # 累计运行的所有批次
        count_list = self.count_list
        id_out_list = self.id_out_list
        tmp_count_list = [np.zeros_like(count) for count in count_list]

        # 用 spaic.Network.run 的方式运行
        backend.set_runtime(time)
        backend.initial_step()
        while (backend.runtime > backend.time - backend.last_time):
            backend.update_step()
            for id_out, tmp_count in zip(id_out_list, tmp_count_list):
                out = backend.to_numpy(backend.get_varialble(id_out))
                # 对于该时间点的输出，沿批次求和
                tmp_count += \
                    out.astype(np.float64, copy=False).sum(axis=0).ravel()

        for count, tmp_count in zip(count_list, tmp_count_list):
            count += tmp_count / time

    def reset(self):
        '''重置统计数据'''

        for count in self.count_list:
            count[:] = 0.
        self.batch_size = 0

    def get_num_time_step(self) -> int:
        '''
        获取时间步数量。

        若发放情况保存各个时间步的瞬时发放频率，则本方法返回时间步数量；
        若发放情况保存各个时间步的发放频率均值，则本方法返回 0。
        '''
        return 0

    def get_firing_rate(self, label: str, neuron_id: np.ndarray) -> np.ndarray:
        '''
        获取发放频率。

        Args:
            lable (str): Encoder 或 NeuronGroup 的 ID。
            neuron_id (np.ndarray): 神经元序号的数组，若为 None 则为所有神经元。

        Returns:
            np.ndarray: 输入神经元的平均发放频率。

        Raises:
            KeyError: 节点发放情况不存在。
            ValueError: 神经元序号不存在。
        '''
        batch_size = self.batch_size
        if not batch_size:
            raise Exception('网络未运行，无法计算频率。')

        try:
            count = self.firing_counts[str(label)]
        except KeyError:
            raise KeyError(f'节点发放情况不存在: "{label}"。')

        if neuron_id is not None:
            try:
                count = count[neuron_id]
            except IndexError as e:
                raise ValueError(f'神经元序号不存在。\n{e}')

        return count / batch_size

    def dump(self, filename: str):
        '''以 HDF5 格式保存统计数据'''
        import h5py

        batch_size = self.batch_size
        if not batch_size:
            raise Exception('网络未运行，无法计算频率。')

        with h5py.File(filename, 'w') as f:
            for id_, count in self.firing_counts.items():
                f.create_dataset(id_, data=count / batch_size)

    def load(self, filename: str):
        '''以 HDF5 格式读取统计数据'''
        import h5py

        with h5py.File(filename, 'r') as f:
            self.firing_counts = {id_: data[:] for id_, data in f.items()}
        self.batch_size = 1
        self.backend = None
        self.count_list = []
        self.id_out_list = []


def test():
    '''测试'''

    net = spaic.Network('test')
    net.input = spaic.Encoder(shape=(3, 4, 5), coding_method='poisson')
    net.neg = spaic.NeuronGroup(shape=(1, 3, 3), model='LIF')
    net.con = spaic.Connection(pre=net.input, post=net.neg, link_type='conv',
        w_mean=0.5, in_channels=3, kernel_size=(2, 3), padding=1, stride=2)

    b = spaic.Torch_Backend()
    b.dt = 1
    net.build(b)

    sfp = SpaicFiringProfile()
    sfp.set_net(net)

    net.input(np.random.random((10, *net.input.shape[1:])))
    sfp.run(500)

    print(f'input: {sfp.get_firing_rate(net.input.id, np.array([5, 20, 40]))}\n'
          f'neg: {sfp.get_firing_rate(net.neg.id, None)}')


if __name__ == '__main__':
    test()
