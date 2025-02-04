import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import psutil
import os
import logging
from typing import Dict, Any

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.start_cpu_freq = None
        self.start_memory = None
        
    def start(self):
        self.start_time = time.time()
        self.start_cpu_freq = psutil.cpu_freq().current
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss
        
    def end(self):
        end_time = time.time()
        end_cpu_freq = psutil.cpu_freq().current
        end_memory = psutil.Process(os.getpid()).memory_info().rss
        
        metrics = {
            'execution_time': end_time - self.start_time,
            'cpu_frequency_mhz': {
                'start': round(self.start_cpu_freq, 2),
                'end': round(end_cpu_freq, 2),
                'average': round((self.start_cpu_freq + end_cpu_freq) / 2, 2)
            },
            'memory_usage_mb': round((end_memory - self.start_memory) / (1024 * 1024), 2)
        }
        return metrics

def setup_logging(log_file: str = 'ecg_processing.log') -> logging.Logger:
    """Configure logging with file and stream handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_ecg_image(image_path: str, output_dir: str = "output") -> Dict[str, Any]:
    """
    Process ECG image from file path.
    
    Args:
        image_path (str): Path to the ECG image file
        output_dir (str): Directory to save output files
    
    Returns:
        Dict containing processing status, metrics, and output file paths
    """
    logger = setup_logging()
    monitor = PerformanceMonitor()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Start performance monitoring
        monitor.start()
        logger.info(f"Starting ECG processing for image: {image_path}")
        
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Image preprocessing
        img = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(img, 50, 150)
        
        # Extract points from edges
        points = np.column_stack(np.where(edges > 0))
        
        # Normalize data
        y_values = max(points[:, 0]) - points[:, 0]
        x_values = points[:, 1]
        
        x_values = (x_values - min(x_values)) / (max(x_values) - min(x_values))
        y_values = (y_values - min(y_values)) / (max(y_values) - min(y_values))
        
        # Remove duplicate X-values and sort
        unique_x, indices = np.unique(x_values, return_index=True)
        unique_y = y_values[indices]
        
        sorted_indices = np.argsort(unique_x)
        x_values_sorted = unique_x[sorted_indices]
        y_values_sorted = unique_y[sorted_indices]
        
        # Save processed data
        output_csv = os.path.join(output_dir, "ecg_data.csv")
        np.savetxt(output_csv, np.column_stack((x_values_sorted, y_values_sorted)), 
                   delimiter=",", header="time,amplitude", comments="")
        
        # Reconstruct ECG signal with interpolation
        f = interp1d(x_values_sorted, y_values_sorted, kind='cubic')
        x_new = np.linspace(0, 1, 500)
        y_new = f(x_new)
        
        # Plot and save visualization
        plt.figure(figsize=(15, 5))
        plt.plot(x_values_sorted, y_values_sorted, 'ro', label="Extracted Points", markersize=2)
        plt.plot(x_new, y_new, 'b-', label="Reconstructed Curve")
        plt.xlabel("Time (Normalized)")
        plt.ylabel("Amplitude (Normalized)")
        plt.title("Extracted & Reconstructed ECG Signal")
        plt.legend()
        plt.grid()
        
        output_plot = os.path.join(output_dir, "ecg_plot.png")
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Get performance metrics
        metrics = monitor.end()
        
        # Log metrics
        logger.info("\nPerformance Metrics:")
        logger.info(f"Total Execution Time: {metrics['execution_time']:.2f} seconds")
        logger.info(f"CPU Frequency (Start): {metrics['cpu_frequency_mhz']['start']} MHz")
        logger.info(f"CPU Frequency (End): {metrics['cpu_frequency_mhz']['end']} MHz")
        logger.info(f"CPU Frequency (Average): {metrics['cpu_frequency_mhz']['average']} MHz")
        logger.info(f"Memory Usage: {metrics['memory_usage_mb']:.1f} MB")
        
        return {
            'status': 'success',
            'metrics': metrics,
            'outputs': {
                'data': output_csv,
                'plot': output_plot
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing ECG: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }

def main():
    """Main execution function."""
    image_path = "single_line_ecg.jpg"  # Replace with your image path
    result = process_ecg_image(image_path)
    
    if result['status'] == 'success':
        print("\nProcessing completed successfully!")
        print(f"Data saved to: {result['outputs']['data']}")
        print(f"Plot saved to: {result['outputs']['plot']}")
        print("\nPerformance Metrics:")
        print(f"Execution Time: {result['metrics']['execution_time']:.2f} seconds")
        print(f"CPU Frequency (Average): {result['metrics']['cpu_frequency_mhz']['average']} MHz")
        print(f"Memory Usage: {result['metrics']['memory_usage_mb']:.1f} MB")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()