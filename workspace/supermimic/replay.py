"""按环境时间对齐的 CPU 半精度环形序列回放池。"""
import torch


FIELDS = ('state', 'next_state', 'observation', 'next_observation',
          'clean_next_observation', 'goal', 'action', 'reward',
          'terminated', 'boundary', 'context')


class SequenceReplay:
    """一次写入所有并行环境的一步，用批量张量索引采样连续窗口。

    论文容量是每环境 1000 步。长期数据放 CPU float16；只把本次 batch
    转回 float32/GPU。boundary 包含失败和 time-out，窗口不能穿过 reset。
    预分配 ring 避免 2048 batch × 多时间步的 Python 逐样本堆叠。
    """

    def __init__(self, capacity_steps, num_envs, storage_dtype=torch.float16):
        if capacity_steps < 2 or num_envs < 1:
            raise ValueError('Replay dimensions must be positive')
        self.capacity_steps, self.num_envs = capacity_steps, num_envs
        self.storage_dtype = storage_dtype
        self.storage = {}
        self.cursor = self.size = 0

    def __len__(self):
        return self.size

    def _allocate(self, values):
        for name in FIELDS:
            value = values[name]
            dtype = torch.bool if name in ('terminated', 'boundary') else self.storage_dtype
            self.storage[name] = torch.empty(
                (self.capacity_steps, self.num_envs, *value.shape[1:]), dtype=dtype, device='cpu')

    def append(self, **values):
        """复制到 CPU ring，避免仿真下一步原位改写已记录 Tensor。"""
        missing = set(FIELDS)-set(values)
        if missing:
            raise ValueError(f'Missing replay fields: {sorted(missing)}')
        if not self.storage:
            self._allocate(values)
        for name in FIELDS:
            value = values[name].detach()
            if value.shape[0] != self.num_envs:
                raise ValueError(f'{name} has {value.shape[0]} envs, expected {self.num_envs}')
            self.storage[name][self.cursor].copy_(value, non_blocking=False)
        self.cursor = (self.cursor+1) % self.capacity_steps
        self.size = min(self.size+1, self.capacity_steps)

    def sample(self, length, batch_size, device, *, allow_final_boundary=True, generator=None):
        """返回 `[B,T,...]`；批量拒绝内部含 episode boundary 的候选。"""
        if length < 1 or self.size < length:
            raise RuntimeError(f'Replay has {self.size} steps; need {length}')
        max_start = self.size-length
        oldest = (self.cursor-self.size) % self.capacity_steps
        chosen_starts, chosen_envs, selected_count, attempts = [], [], 0, 0
        stop = length-1 if allow_final_boundary else length
        offsets = torch.arange(length, dtype=torch.long)
        while selected_count < batch_size and attempts < 40:
            attempts += 1
            needed = batch_size-selected_count
            count = max(needed*4, 64)
            starts = torch.randint(max_start+1, (count,), generator=generator)
            envs = torch.randint(self.num_envs, (count,), generator=generator)
            indices = (oldest+starts[:, None]+offsets) % self.capacity_steps
            valid = ~self.storage['boundary'][indices[:, :stop], envs[:, None]].any(dim=1)
            selected = valid.nonzero(as_tuple=False).squeeze(-1)[:needed]
            if selected.numel():
                chosen_starts.append(starts[selected])
                chosen_envs.append(envs[selected])
                selected_count += selected.numel()
        if selected_count != batch_size:
            raise RuntimeError('Not enough contiguous replay windows; collect more post-reset data')
        starts, envs = torch.cat(chosen_starts), torch.cat(chosen_envs)
        indices = (oldest+starts[:, None]+offsets) % self.capacity_steps
        result = {}
        for name in FIELDS:
            value = self.storage[name][indices, envs[:, None]]
            if name not in ('terminated', 'boundary'):
                value = value.float()
            result[name] = value.to(device)
        return result

    def state_dict(self, tail_steps=0):
        """默认只存元数据；需要调试时可保存按时间排序的少量尾部。"""
        tail = []
        if tail_steps and self.size:
            count = min(tail_steps, self.size)
            oldest = (self.cursor-self.size) % self.capacity_steps
            logical = torch.arange(self.size-count, self.size)
            indices = (oldest+logical) % self.capacity_steps
            tail = [{name: self.storage[name][index].clone() for name in FIELDS}
                    for index in indices]
        return dict(capacity_steps=self.capacity_steps, num_envs=self.num_envs,
                    stored_steps=self.size, tail=tail)

    def load_state_dict(self, state):
        if state['capacity_steps'] != self.capacity_steps or state['num_envs'] != self.num_envs:
            raise ValueError('Replay shape differs from checkpoint')
        self.storage, self.cursor, self.size = {}, 0, 0
        for item in state.get('tail', ()):
            self.append(**item)
