import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file_path):
    # Data structures to store extracted information
    steps = []
    losses = []
    
    # Memory usage data
    memory_data = {}  # Dictionary to store memory data by step
    
    # Regular expressions for pattern matching
    loss_pattern = re.compile(r'Step (\d+), Loss: (\d+\.\d+)')
    step_pattern = re.compile(r'Step (\d+):')
    rss_pattern = re.compile(r'- RSS: (\d+\.\d+)MB')
    vms_pattern = re.compile(r'- VMS: (\d+\.\d+)MB')
    gpu_pattern = re.compile(r'- GPU: (\d+\.\d+)MB allocated, (\d+\.\d+)MB reserved')
    
    current_step = None
    collecting_memory = False
    memory_info = {}
    
    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # First check for step loss information
            loss_match = loss_pattern.search(line)
            if loss_match:
                step = int(loss_match.group(1))
                loss = float(loss_match.group(2))
                steps.append(step)
                losses.append(loss)
                continue
            
            # Check for lines indicating a step number before memory info
            step_match = step_pattern.search(line)
            if step_match:
                # If we found a new step, save the previous step's memory data
                if collecting_memory and current_step is not None and memory_info:
                    memory_data[current_step] = memory_info
                
                # Start collecting for the new step
                current_step = int(step_match.group(1))
                collecting_memory = "Memory usage" in line
                memory_info = {} if collecting_memory else {}
                continue
            
            # Check for memory information if we're in a collecting state
            if collecting_memory and current_step is not None:
                rss_match = rss_pattern.search(line)
                if rss_match:
                    memory_info['rss'] = float(rss_match.group(1))
                    continue
                    
                vms_match = vms_pattern.search(line)
                if vms_match:
                    memory_info['vms'] = float(vms_match.group(1))
                    continue
                    
                gpu_match = gpu_pattern.search(line)
                if gpu_match:
                    memory_info['gpu_allocated'] = float(gpu_match.group(1))
                    memory_info['gpu_reserved'] = float(gpu_match.group(2))
                    
                    # Save this memory data since we've collected all fields
                    memory_data[current_step] = memory_info.copy()
                    collecting_memory = False
                    continue
    
    # Extract memory values into lists aligned with steps
    memory_steps = sorted(memory_data.keys())
    rss_values = [memory_data[step].get('rss', 0) for step in memory_steps]
    vms_values = [memory_data[step].get('vms', 0) for step in memory_steps]
    gpu_allocated = [memory_data[step].get('gpu_allocated', 0) for step in memory_steps]
    gpu_reserved = [memory_data[step].get('gpu_reserved', 0) for step in memory_steps]
    
    # Create a dictionary to return all collected data
    data = {
        'steps': steps,
        'losses': losses,
        'memory_steps': memory_steps,
        'rss': rss_values,
        'vms': vms_values,
        'gpu_allocated': gpu_allocated,
        'gpu_reserved': gpu_reserved
    }
    
    return data

def plot_data(data):
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot loss over time
    ax1.plot(data['steps'], data['losses'], 'b-', marker='o', markersize=3, label='Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Training Steps')
    ax1.grid(True)
    ax1.legend()
    
    # Plot RSS and GPU memory on one plot (similar scale)
    ax2.plot(data['memory_steps'], data['rss'], 'r-', marker='o', markersize=3, label='RSS Memory (MB)')
    ax2.plot(data['memory_steps'], data['gpu_allocated'], 'm-', marker='o', markersize=3, label='GPU Allocated (MB)')
    ax2.plot(data['memory_steps'], data['gpu_reserved'], 'y-', marker='o', markersize=3, label='GPU Reserved (MB)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('RSS and GPU Memory Usage')
    ax2.grid(True)
    ax2.legend()
    
    # Plot VMS separately (different scale)
    ax3.plot(data['memory_steps'], data['vms'], 'g-', marker='o', markersize=3, label='VMS Memory (MB)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Memory (MB)')
    ax3.set_title('VMS Memory Usage')
    ax3.grid(True)
    ax3.legend()
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    
    # Create additional plots showing the relationship between loss and memory
    if len(data['memory_steps']) > 0 and len(data['losses']) > 0:
        # Create a map of step to loss
        step_to_loss = {step: loss for step, loss in zip(data['steps'], data['losses'])}
        
        # Get corresponding loss values for memory steps
        valid_memory_steps = []
        valid_loss_values = []
        valid_gpu_allocated = []
        valid_rss = []
        
        for i, step in enumerate(data['memory_steps']):
            if step in step_to_loss:
                valid_memory_steps.append(step)
                valid_loss_values.append(step_to_loss[step])
                valid_gpu_allocated.append(data['gpu_allocated'][i])
                valid_rss.append(data['rss'][i])
        
        if valid_memory_steps:
            # Create loss vs. memory plots
            fig2, (ax4, ax5) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot GPU allocated vs Loss
            ax4.scatter(valid_loss_values, valid_gpu_allocated, c='purple', alpha=0.7)
            ax4.set_xlabel('Loss')
            ax4.set_ylabel('GPU Memory Allocated (MB)')
            ax4.set_title('Loss vs. GPU Memory Allocation')
            ax4.grid(True)
            
            # Plot RSS vs Loss
            ax5.scatter(valid_loss_values, valid_rss, c='red', alpha=0.7)
            ax5.set_xlabel('Loss')
            ax5.set_ylabel('RSS Memory (MB)')
            ax5.set_title('Loss vs. RSS Memory')
            ax5.grid(True)
            
            plt.tight_layout()
            plt.savefig('loss_vs_memory.png')
    
    plt.show()

if __name__ == "__main__":
    log_file_path = "refat_run.log"
    data = parse_log_file(log_file_path)
    
    # Print summary statistics
    print(f"Parsed {len(data['steps'])} steps with loss values")
    print(f"Parsed {len(data['memory_steps'])} steps with memory information")
    
    if len(data['steps']) > 0:
        print(f"Loss range: {min(data['losses']):.4f} to {max(data['losses']):.4f}")
    
    if len(data['memory_steps']) > 0:
        print(f"RSS Memory range: {min(data['rss']):.2f}MB to {max(data['rss']):.2f}MB")
        print(f"VMS Memory range: {min(data['vms']):.2f}MB to {max(data['vms']):.2f}MB")
        print(f"GPU Allocated range: {min(data['gpu_allocated']):.2f}MB to {max(data['gpu_allocated']):.2f}MB")
        print(f"GPU Reserved range: {min(data['gpu_reserved']):.2f}MB to {max(data['gpu_reserved']):.2f}MB")
    
    # Plot the data
    plot_data(data) 