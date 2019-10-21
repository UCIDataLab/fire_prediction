import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from .zero_inflated_models import PoissonHurdleRegression

class ResidualPoissonHurdleRegression(PoissonHurdleRegression):
    def fit(self, X, y=None):
        print('=== Start fit')
        ret = super().fit(X, y)

        pred = ret.predict(X, use_residual_update=False)

        if not y:
            y = X[self.response_var]

        residual = y - pred

        residual_x = []
        residual_y = []

        for index, row in X.iterrows():
            time = row['time']
            time = datetime.strptime(time, '%Y-%m-%d')
            time += timedelta(days=1)
            time = time.strftime('%Y-%m-%d')

            same_date = X.loc[(X['time'] == time) & (X['x'] == row['x']) & (X['y'] == row['y'])]

            if len(same_date) > 0:
                residual_x.append(residual[index])
                residual_y.append(residual[same_date.index[0]])

        print('=== Mean fit residual', np.shape(residual), np.mean(residual))

        self.residual_model = LinearRegression().fit(np.array(residual_x).reshape(-1, 1), residual_y)

        return ret


    def predict(self, X, choice=None, use_residual_update=True):
        pred = super().predict(X, choice)
        y = X[self.response_var]

        residual = y - pred

        print(X)
        print('=== Mean pred residual', np.shape(residual), np.mean(residual))

        if use_residual_update:
            ind = pred != 0
            update_residual = self.residual_model.predict(residual[:-1])
            pred -= update_residual

        return pred




