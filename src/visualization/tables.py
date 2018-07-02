"""
Generating tables.
"""
import StringIO

def build_results_table(results_list):
    out = StringIO.StringIO()

    metrics_ = [metrics.mean_absolute_error, metrics.root_mean_squared_error]
    for results,title in results_list:
        out.write(title + '\n')
        out.write('=====================' + '\n')

        for metric in metrics_:
            out.write(metric.__name__ + '\n')
            table = []
            for k,v in results.iteritems():
                vals = map(lambda x: round(metric(*flat(x)),5), results[k])
                table.append([k]+vals)
            out.write(tabulate.tabulate(table))

    return out.getvalue()

