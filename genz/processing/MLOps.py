# %%
import os.path as op
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_flavor as pf
import seaborn as sns
from genz import defaults
from numpy import mean, std
from sklearn.compose import make_column_transformer
from sklearn.experimental import enable_hist_gradient_boosting  #noqa
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor)
from sklearn.experimental import enable_hist_gradient_boosting  #noqa
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LassoCV, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, StandardScaler)
from sklearn.utils import shuffle

# %%
sns.set(style='ticks', color_codes=True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('precision', 2)
seed = np.random.seed(42)

features = ['id',
            'gender',
            'peermindset',
            'persmindset',
            'needforapproval',
            'needforbelonging',
            'rejection',
            'coping_mad',
            'coping_sad',
            'coping_worried',
            'rsqanxiety',
            'rsqanger',
            'cdimean',
            'moodgood',
            'moodhappy',
            'moodrelaxed',
            'stateanxiety',
            'roi',
            'hem',
            'deg']
target = ['traitanxiety']


# %%
tuned_parameters = [{}]
n_folds = 5

plt.figure().set_size_inches(8, 6)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)


# alpha=0.2 controls the translucency of the fill color
plt.ylabel('CV score +/- std error')
plt.axhline(np.max(scores), linestyle='--', color='.5')

# %%

# rf_pipeline = make_pipeline(processor_nlin,
#                             RandomForestRegressor(random_state=seed))

# gradient_pipeline = make_pipeline(
#     processor_nlin,
#     HistGradientBoostingRegressor(random_state=seed))

estimators = [#('Random Forest', rf_pipeline),
              ('Lasso', lasso_pipeline),
              #('Gradient Boosting', gradient_pipeline)
              ]

stacking_regressor = StackingRegressor(estimators=estimators,
                                       final_estimator=RidgeCV())# %%
# %%
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_predict


def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title = title + '\n Evaluation in {:.2f} seconds'.format(elapsed_time)
    ax.set_title(title)
    

# %%
fig, axs = plt.subplots(2, 2, figsize=(9, 7))
axs = np.ravel(axs)

for ax, (name, est) in zip(axs, estimators + [('Stacking Regressor',
                                               stacking_regressor)]):
    start_time = time.time()
    score = cross_validate(est, X, y,
                           scoring=['r2', 'neg_mean_absolute_error'],
                           n_jobs=-1, verbose=0)
    elapsed_time = time.time() - start_time

    y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0)

    plot_regression_results(
        ax, y, y_pred,
        name,
        (r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$MAE={:.2f} \pm {:.2f}$')
        .format(np.mean(score['test_r2']),
                np.std(score['test_r2']),
                -np.mean(score['test_neg_mean_absolute_error']),
                np.std(score['test_neg_mean_absolute_error'])),
        elapsed_time)

plt.suptitle('Single predictors versus stacked predictors')
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# %%


#                                        



# %%
lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=10000)
k_fold = KFold(3)

print("Alpha parameters maximising the generalization score on different")
print("subsets of the data:")
for k, (train, test) in enumerate(k_fold.split(X, y)):
    lasso_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
print()

# %%
