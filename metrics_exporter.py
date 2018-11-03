
import os
import csv
from datetime import datetime


class MetricsExporter():
    def __init__(self):
        self.model = None
        self.model_name = None
        self.dataset_name = None
        self.output_dir = 'results'
        self.run_id = None
        self.num_factors = None
        self.num_negatives = None


        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def export(self, data):

        path = f'metrics.csv'
        output_path = os.path.join(self.output_dir, path)

        metrics = []

        if not os.path.exists(output_path):
            metrics.append([
                'timestamp',
                'model',
                'dataset',
                'factors',
                'negatives',
                'epoch',
                'metric',
                'value'
                ]
            )

        for metric_name, values in data.items():
            for epoch, metric_value in enumerate(values):
                metrics.append([
                    self.run_id,
                    self.model_name,
                    self.dataset_name,
                    self.num_factors,
                    self.num_negatives,
                    epoch,
                    metric_name,
                    metric_value])

        with open(output_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(metrics)