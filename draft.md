Problem statement:


Data set


Comment: 

Technically, the data set . Thus, the estimated causal effect and not. Since, all   Applying this  relies on the assumption that. 
# setup
import numpy as np
import pandas as pd
import scipy.stats as stats
import doubleml as dml
import statsmodels.api as sm
import patsy
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
EDA:

First, some basic checks. The data does not contain missing values and the covariates are already standardized.
# load data
df = pd.read_csv("data/uplift_data_final.csv", index_col=0)

# missing values? standardized data? balance treatment and control?
print(df.head())
print(df.isnull().sum())
print(df.mean())
print(df.std())
For illustration, the true . Note that naive, without causal considerations. 

We obsvere, correlation (imbalance in ). 
# true average treatment effect?  without causal analysis?
print("true average treatment effect:", df['ite'].mean()) 

treated = df[df['coupon'] == 1]
control = df[df['coupon'] == 0]
print("naive difference in conversion rates:", treated['conversion'].mean()- control['conversion'].mean())

# potential confounding? imbalance? correlation with treatment?
covariates = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10',
              'X11', 'X12', 'X13', 'X14']
for col in covariates:
    print(f"correlation between {col} and coupon:", np.corrcoef(df[col], df['coupon'])[0, 1])
The first step of causal inference is to reason about underlying causal structure. Could be done using causal discovery methods, however here not necessary due to time constraints (pre- /post-treatment). 

Modeling Choices/Assumptions:

- causal sufficiency 
- causal markov condition


We choose the DAG corresponding to minimal assumptions (missing edges correspond to conditional independence assumptions). It might be possible to choose a smaller more efficient (see e.g. ) adjustment set, however, including all pre-treatment variables is always a valid adjustement set (without bias) assuming causal sufficiency.

Post-treatment variables are 

Comments:

This is th
# causal analysis, first step causal graph
from graphviz import Digraph
from IPython.display import Image

dot = Digraph()
dot.attr(rankdir='TB')

with dot.subgraph() as top:
    top.attr(rank='same')
    top.node('X', 'Customer Features (X)', style='filled', fillcolor='lightcoral')
    top.node('M', 'Membership Level (M)', style='filled', fillcolor='lightcoral')

dot.node('C', 'Coupon')
dot.node('Y', 'Conversion')

with dot.subgraph() as bottom:
    bottom.attr(rank='same')
    bottom.node('L', 'Loyalty Points (L)', style='filled', fillcolor='lightblue')
    bottom.node('Z', 'Activity Score (Z)', style='filled', fillcolor='lightblue')

dot.edges([('X', 'C'), ('M', 'C'),
           ('X', 'Y'), ('M', 'Y'),
           ('C', 'Y'),
           ('C', 'L'), ('C', 'Z'),
           ('Y', 'L'), ('Y', 'Z'),
           ('X', 'L'), ('M', 'L'),
           ('X', 'Z'), ('M', 'Z')])

dot.render('dag', format='png', cleanup=True)
Image('dag.png')


Comments:

This is the major 'causal' step in the analysis (translation across worlds, relating interventional distribtution to observational distribution). Afterwards its mainly back about classical statistics to estimate ..
# backdoor adjustment set
adjustment_set = covariates + ['membership_level_0', 'membership_level_1', 'membership_level_2']


The main idea of ipw is to 

Modeling Choices/Assumptions:

- 
# ATE estimation using different methods

# ipw 
prop_model = LogisticRegression(max_iter=1000)
prop_model.fit(df[adjustment_set], df['coupon'])  
propensity = prop_model.predict_proba(df[adjustment_set])[:, 1]

plt.hist(propensity[df['coupon']==1], alpha=0.5, label='Treated')
plt.hist(propensity[df['coupon']==0], alpha=0.5, label='Control')
plt.legend() 
plt.show()


plt.hist(propensity[df['coupon']==1], alpha=0.5, weights=1/propensity[df['coupon']==1], label='Treated')
plt.hist(propensity[df['coupon']==0], alpha=0.5, weights=1/(1-propensity[df['coupon']==0]),label='Control')
plt.title('reweighted sample')
plt.legend() 
plt.show()

ate_iwp = np.mean((df['coupon'] * df['conversion']) / propensity - ((1 - df['coupon']) * df['conversion']) / (1 - propensity))
print("ipw estimated ATE:", ate_iwp)
print("true average treatment effect:", df['ite'].mean()) 
The main idea

Modeling Choices/Assumptions:

-
# logistic regression 
outcome_model = LogisticRegression(max_iter=1000)
outcome_model.fit(df[adjustment_set + ['coupon'] ], df['conversion'])

y_hat_1 = outcome_model.predict_proba(df[adjustment_set + ['coupon'] ].assign(coupon=1))[:, 1]
y_hat_0 = outcome_model.predict_proba(df[adjustment_set + ['coupon'] ].assign(coupon=0))[:, 1]
ate_lr = (y_hat_1-y_hat_0).mean()
print("logistic regression estimated ATE:", ate_lr)
print("true average treatment effect:", df['ite'].mean()) 
To some extend, doubly robust estimation combines both worlds. As name suggests, . Doubleml approach uses data splitting to allow without overfitting.

Modeling Choices/Assumptions:
# doubly robust estimation (here usimg doubleml)

# logistic regression (simple)
treatment_model = LogisticRegression(max_iter=1000)
outcome_model = LogisticRegression(max_iter=1000)

dml_data = dml.DoubleMLData(df, y_col='conversion', d_cols='coupon',
                        x_cols=adjustment_set)

dml_irm_lr = dml.DoubleMLIRM(dml_data,
                          ml_g=outcome_model,
                          ml_m=treatment_model,
                          n_folds=5)

dml_irm_lr.fit()
print("double logistic regression estimated ATE:\n", dml_irm_lr.summary)
print("true average treatment effect:", df['ite'].mean()) 
Modeling Choices/Assumptions:

# random forest (more involved)
treatment_model = RandomForestClassifier(n_estimators=500)
outcome_model = RandomForestRegressor(n_estimators=500)

dml_irm_rf = dml.DoubleMLIRM(dml_data,
                          ml_g=outcome_model,
                          ml_m=treatment_model,
                          n_folds=5)

dml_irm_rf.fit()
print("double random forest estimated ATE:\n", dml_irm_rf.summary)
print("true average treatment effect:", df['ite'].mean()) 
This can be used to
# GATE estimation by membership level 
membership_level = df[['membership_level_0', 'membership_level_1', 'membership_level_2']].astype('bool')
gate = dml_irm_rf.gate(groups=membership_level)
print("double random forest estimated GATE:\n", gate)

for level in ['membership_level_0', 'membership_level_1', 'membership_level_2']:
    print(f"true GATE {level}: {df['ite'][df[level] == 1].mean()}")
More involved 
# CATE estimation by X10 (linear projection)
design_matrix=patsy.dmatrix("x", {"x": df['X10']})
spline_basis = pd.DataFrame(design_matrix)
cate = dml_irm_rf.cate(spline_basis)
print(cate)

# visualization 
x_grid = {"x": np.linspace(min(df['X10']), max(df['X10']), 100)}
spline_grid = pd.DataFrame(patsy.build_design_matrices([design_matrix.design_info], x_grid)[0])
df_cate = cate.confint(spline_grid, level=0.95, joint=True, n_rep_boot=2000)

df_cate['x'] = x_grid['x']
true_fitted_spline = sm.OLS(df['ite'], spline_basis).fit()
df_cate['ite_fitted'] = true_fitted_spline.predict(spline_grid)

fig, ax = plt.subplots()
ax.plot(df_cate['x'],df_cate['effect'], color='#238B45', label='Estimated Effect')
ax.plot(df_cate['x'],df_cate['ite_fitted'], color='#C51B8A', label='Fitted True Effect')
ax.scatter(df['X10'], df['ite'], color='#969696', alpha=0.1, s=10, label='Observed ITE', zorder=1)
ax.fill_between(df_cate['x'], df_cate['2.5 %'], df_cate['97.5 %'], color='#A1D99B', alpha=.3, label='Confidence Interval')

plt.legend()
plt.title('CATE')
plt.xlabel('x')
_ =  plt.ylabel('Effect and 95%-CI')

plt.show()

# CATE estimation by X11 (quadratic projection)
design_matrix=patsy.dmatrix("x + I(x**2)", {"x": df['X11']})
spline_basis = pd.DataFrame(design_matrix)
cate = dml_irm_rf.cate(spline_basis)
print(cate)

# visualization 
x_grid = {"x": np.linspace(min(df['X11']), max(df['X11']), 100)}
spline_grid = pd.DataFrame(patsy.build_design_matrices([design_matrix.design_info], x_grid)[0])
df_cate = cate.confint(spline_grid, level=0.95, joint=True, n_rep_boot=2000)

df_cate['x'] = x_grid['x']
true_fitted_spline = sm.OLS(df['ite'], spline_basis).fit()
df_cate['ite_fitted'] = true_fitted_spline.predict(spline_grid)

fig, ax = plt.subplots()
ax.plot(df_cate['x'],df_cate['effect'], color='#238B45', label='Estimated Effect')
ax.plot(df_cate['x'],df_cate['ite_fitted'], color='#C51B8A', label='Fitted True Effect')
ax.scatter(df['X11'], df['ite'], color='#969696', alpha=0.1, s=10, label='Observed ITE', zorder=1)
ax.fill_between(df_cate['x'], df_cate['2.5 %'], df_cate['97.5 %'], color='#A1D99B', alpha=.3, label='Confidence Interval')

plt.legend()
plt.title('CATE')
plt.xlabel('x')
_ =  plt.ylabel('Effect and 95%-CI')

plt.show()
Conclusion



Suggestion:

Comment:

All underestimate