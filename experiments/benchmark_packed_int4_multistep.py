import torch

# Assuming your device and dtype setup
device = 'cuda'
dtype = torch.bfloat16

# Define your model for 7B-class geometry
class SampleModel(torch.nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.linear = torch.nn.Linear(1024, 1024)  # Example geometry

    def forward(self, x):
        return self.linear(x)

model = SampleModel().to(device).to(dtype)

# Function to benchmark decoding performance
def benchmark_decoding(steps=100):
    with torch.no_grad():
        total_time = 0.0
        for _ in range(steps):
            input_data = torch.randn(1, 1024, device=device, dtype=dtype)  # Example input
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            _ = model(input_data)  # Forward pass
            end_time.record()
            end_time.synchronize()
            total_time += start_time.elapsed_time(end_time)

        average_time = total_time / steps
        print(f"Average decoding time over {steps} steps: {average_time:.4f}ms")

if __name__ == '__main__':
    benchmark_decoding()