"""
Wraps multiple datasets (e.g. one for active fire and one for ignitions) so they can be split
by the cross-validation easily.
"""


class MultidataWrapper(object):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, key):
        return self.datasets[key]

    def remove_year(self, year):
        datasets_excl = []
        datasets_incl = []
        for ds in self.datasets:
            if ds is not None:
                ds_excl = ds.sel(time=ds.time.dt.year != year)
                ds_incl = ds.sel(time=ds.time.dt.year == year)
            else:
                ds_excl, ds_incl = None, None

            datasets_excl.append(ds_excl)
            datasets_incl.append(ds_incl)

        return MultidataWrapper(datasets_excl), MultidataWrapper(datasets_incl)
