import matplotlib.pyplot as plt
import numpy as np

# Use data obtained from the result calculations.
# Using hard-coded values here for simplicity and clarity.
tasks = {
    'ABR': {
        'title': 'Adaptive Bitrate Streaming (ABR)',
        'ylabel': 'Quality of Experience (Higher is Better)',
        'original': 1.00,
        'repro1': 0.98,
        'repro1_err': 0.03,
        'repro2': 0.75,
        'repro2_err': 0.04,
        'labels': ['Original', 'Reproduction (Checkpoint)', 'Reproduction (Fine-tuning)']
    },
    'CJS': {
        'title': 'Cluster Job Scheduling (CJS)',
        'ylabel': 'Job Completion Time, Seconds (Lower is Better)',
        'original': 48.08,
        'repro1': None,  
        'repro1_err': 0,
        'repro2': 45.51, 
        'repro2_err': 5.95,
        'labels': ['Original', 'Reproduction (Checkpoint)', 'Reproduction (Fine-tuning)']
    },
    'VP': {
        'title': 'Viewport Prediction (VP)',
        'ylabel': 'Mean Absolute Error, Degrees (Lower is Better)',
        'original': 11.88,
        'repro1': 12.86,
        'repro1_err': 1.31,
        'repro2': 11.51, 
        'repro2_err': 1.16,
        'labels': ['Original', 'Reproduction (Checkpoint)', 'Reproduction (Fine-tuning)']
    },
        'ABR_3': {
        'title': 'Adaptive Bitrate Streaming (ABR)',
        'ylabel': 'Quality of Experience (Higher is Better)',
        'original': 1.00,
        'repro1': 0.75,
        'repro1_err': 0.04,
        'repro2': 0.76,
        'repro2_err': 0.04,
        'labels': ['Original', 'Llama2', 'Llama3']
    },
    'CJS_3': {
        'title': 'Cluster Job Scheduling (CJS)',
        'ylabel': 'Job Completion Time, Seconds (Lower is Better)',
        'original': 48.08,
        'repro1': 45.51,  
        'repro1_err': 5.95,
        'repro2': 53.92, 
        'repro2_err': 6.49,
        'labels': ['Original', 'Llama2', 'Llama3']
    },
    'VP_3': {
        'title': 'Viewport Prediction (VP)',
        'ylabel': 'Mean Absolute Error, Degrees (Lower is Better)',
        'original': 11.88,
        'repro1': 11.51,
        'repro1_err': 1.16,
        'repro2': 12.48, 
        'repro2_err': 0.55,
        'labels': ['Original', 'Llama 2', 'Llama 3']
    }
}

def create_task_chart(task_name, data):
    means = [data['original'], data['repro1'], data['repro2']]
    errors = [0, data['repro1_err'], data['repro2_err']]
    

    # Add check for None
    valid_indices = [i for i, x in enumerate(means) if x is not None]
    plot_labels = [data['labels'][i] for i in valid_indices]
    plot_means = [means[i] for i in valid_indices]
    plot_errors = [errors[i] for i in valid_indices]

    x = np.arange(len(plot_labels))
    width = 0.6

    _, ax = plt.subplots(figsize=(7, 5))
    
    color = '#3498db'
    
    bars = ax.bar(x, plot_means, width, yerr=plot_errors, 
                  capsize=8, color=color, alpha=0.9, edgecolor='black')

    # Formatting
    ax.set_title(data['title'], fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel(data['ylabel'], fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, fontsize=11)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    filename = f"{task_name}_comparison.pdf"
    plt.savefig(filename, dpi=300)
    plt.close()

# Generate charts
for task_id, task_data in tasks.items():
    create_task_chart(task_id, task_data)