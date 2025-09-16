import torch

# 检查是否有可用的GPU
if not torch.cuda.is_available():
    print("CUDA is not available. This script requires a GPU to run.")
    exit()

# 定义元素的数量
# const unsigned n = 1 << 30;
n = 1 << 30

print(f"Preparing to calculate softmax for {n} elements on the GPU...")

# 将设备设置为cuda
device = torch.device('cuda')

try:
    # 在GPU上直接创建一个包含n个单精度浮点数的张量
    # 直接在GPU上生成，以避免测量CPU到GPU的数据传输时间
    x = torch.randn(n, dtype=torch.float32, device=device)
    print(f"Successfully allocated a tensor with {n} elements on the GPU.")
except torch.cuda.OutOfMemoryError:
    gb_required = n * 4 / (1024**3)
    print(f"Error: Not enough GPU memory.")
    print(f"To store {n} float32 numbers, you need at least {gb_required:.2f} GB of GPU memory.")
    exit()


# 使用CUDA Events来精确测量GPU操作的执行时间
# 这是在PyTorch中测量GPU代码段执行时间的标准方法
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 预热(Warm-up)
# 第一次在GPU上执行操作时，可能会有一些额外的初始化开销。
# 我们先运行一次，确保后续的计时是准确的。
print("Performing a warm-up run...")
_ = torch.nn.functional.softmax(x, dim=0)

# 确保所有GPU操作都已完成，以便我们从一个干净的状态开始计时
torch.cuda.synchronize()

print("Starting the benchmark...")

# --- 开始计时 ---
start_event.record()

# 执行Softmax计算
# 这是我们唯一要计时的部分
softmax_output = torch.nn.functional.softmax(x, dim=0)

# --- 结束计时 ---
end_event.record()


# 等待所有CUDA核心完成它们的工作
# 这一点至关重要，因为CUDA的调用是异步的
torch.cuda.synchronize()

# 计算并打印消耗的时间
elapsed_time_ms = start_event.elapsed_time(end_event)
print("-" * 50)
print(f"Softmax calculation time on GPU: {elapsed_time_ms:.4f} ms")
print("-" * 50)
