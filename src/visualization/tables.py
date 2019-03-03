"""
Generating tables.
"""
import io

import tabulate

from src.evaluation import metrics
from src.pipeline.train_pipeline import flat


def build_results_table(results_list):
    out = io.StringIO()

    metrics_ = [metrics.mean_absolute_error, metrics.root_mean_squared_error]
    for results, title in results_list:
        out.write(title + '\n')
        out.write('=====================' + '\n')

        for metric in metrics_:
            out.write(metric.__name__ + '\n')
            table = []
            for k, v in results.items():
                values = list(map(lambda x: round(metric(*flat(x)), 5), results[k]))
                table.append([k] + values)
            out.write(tabulate.tabulate(table))

    return out.getvalue()
