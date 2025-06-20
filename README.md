## Causal Analysis: Targeted Coupon Campaigns

### Problem Statement

We aim to evaluate the causal effect of coupon usage on product conversion using observational data from an online shop's newsletter subscribers.

### Dataset Overview

- **Size**: 10,000 observations
- **Outcome**: `conversion` (binary)
- **Treatment**: `coupon` (binary)
- **Pre-treatment covariates**: `X1`–`X14` (numeric), `membership_level` (categorical, dummy-coded)
- **Post-treatment variables**: `Z` (activity score), `loyalty_points` (numeric)



**Comment**: All customers were offered a coupon, but only some chose to use it. Therefore, the estimated causal effect pertains to the impact of using a coupon on the conversion rate. As there is no data about customers that did not receive any coupons, drawing causal conclusion about the effect of receiving a coupon critically relies on the assumption that
the offer itself does not influence conversion except through its use.

**References**: Most of this analysis follows standard causal inference literature, for details we refer to, e.g., Causal Inference: What if by Miguel Hernán and James Robins or Causal Inference: The Mixtape by Scott Cunningham.  We use the DoubleML package to implement the main model based on Chernozhukov et al. (2018). For details on DAGs, and in particular the backdoor criterion, we refer to Judea Pearl's Causality book.



### EDA

First, some basic checks. The data does not contain missing values and the covariates appear to be already standardized.


```python
# load data
df = pd.read_csv("data/uplift_data_final.csv", index_col=0)

# missing values? standardized data? balance treatment and control?
print(df.head())
print(df.isnull().sum())
print(df.mean())
print(df.std())
```

For illustration, we compute the true average treatment effect. Note that the naive difference of treated vs control, without causal considerations, would imply a different sign for the effect. This is not a randomized study, thus, we have to adjust for confounding or imbalance in the groups, i.e., use causal inference approaches.

We highlight the (marginal) correlation between treatment and the covariates. 


```python
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
```

    true average treatment effect: 0.03908395919621665
    naive difference in conversion rates: -0.020071941191220577
    correlation between X0 and coupon: 0.018481828739076077
    correlation between X1 and coupon: 0.008034546295127641
    correlation between X2 and coupon: 0.02487746175698864
    correlation between X3 and coupon: 0.044094632953325226
    correlation between X4 and coupon: 0.06008973524142825
    correlation between X5 and coupon: 0.11569101545631695
    correlation between X6 and coupon: 0.09585159094710013
    correlation between X7 and coupon: 0.1498084290646118
    correlation between X8 and coupon: 0.20689439487009248
    correlation between X9 and coupon: 0.12702288360936814
    correlation between X10 and coupon: 0.14751202127925872
    correlation between X11 and coupon: 0.10992197370724054
    correlation between X12 and coupon: 0.14475091349319172
    correlation between X13 and coupon: 0.12338138760930167
    correlation between X14 and coupon: 0.06124696264329803


### Causal Framework

The first step of causal inference is to reason about the underlying causal structure. Here, causal discovery is not necessary; we specify a DAG based on domain knowledge (e.g. time constraints pre-/post-treatment) and minimal assumptions (not necessarily minimal causal dag), where missing edges encode hypothesized absences of direct causal effects. We emphasize that specifying such a causal DAG assumes 
- **causal sufficiency** (no unobserved confounding)
- **causal markov condition** (independent mechanism, conditional independence implied by d-separation)
- **and acyclicity.**

**Comment**: The set of all pre-treatment covariates satisfies the backdoor criterion, guaranteeing unbiased effect estimation. However, smaller adjustment sets (e.g., a subset of covariates) may improve efficiency, but crucially, this relies on correctly identifying a minimal sufficient adjustment set (specifying a minimal causal DAG), which requires additional conditional independence assumptions.


![png](coupon_files/dag.png)

Finding a valid adjustment set is the major causal step in the analysis that relates the interventional quantity of interest to observational data. We emphasize that adjusting for colliders d-connects and introduces bias (similarly, adjusting for mediators would block a causal path and introduce bias). Thus, we do not adjust for the post-treatment variables.

**Comment**: There exist other (graphical) criteria, such as the front-door criterion, that also provide sufficient conditions to identify causal effects from observational data.


```python
# backdoor adjustment set
adjustment_set = covariates + ['membership_level_0', 'membership_level_1', 'membership_level_2']
```

We are interested in estimating the average treatment effect (ATE) of coupon usage on conversion rate. Formally, this is given as

$$
\text{ATE} = \mathbb{E}[conversion \mid do(coupon=1)] - \mathbb{E}[conversion \mid do(coupon=0)].
$$

Using the backdoor criterion this can be expressed as

$$
\text{ATE} = \mathbb{E}_{X,M} \left[ \mathbb{E}[conversion \mid coupon=1, X, M] - \mathbb{E}[conversion \mid coupon=0, X, M] \right].
$$

### Estimation Approaches

The main remaining challenge is to estimate these conditional expectations from data. We will first highlight some important concepts using simple approaches before employing more sophisticated models to continue with our analysis. The main assumptions that ensure validity of the following approaches are
- **positivity** (common support, overlap)
- **conditional** exchangeability (implied by causal sufficiency/no unobserved confounding)
- **consistency** (implicit in the specification of the causal dag)

#### 1. Inverse Probability Weighting (IPW)

The main idea of ipw is to rebalance the data to mimic a randomized control trial by fitting a treatment model (here logistic regression).
- validity additionally relies on well-specified treatment model



```python
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
```


    
![png](coupon_files/coupon_10_0.png)
    



    
![png](coupon_files/coupon_10_1.png)
    


    ipw estimated ATE: 0.010870724339120993
    true average treatment effect: 0.03908395919621665


We show that the positivity assumption seems to be satisfied, however, the plot also shows the imbalance in treatment vs control. The second plot shows the idea of ipw of reweighting the data to mimic a RCT.

While still underestimating the causal effect, ipw at least identifies the correct sign compared to the naive approach.


#### 2. Outcome Modeling

Another straightforward approach is to fit directly an outcome model to estimate the conditional expectation (here, again, simple logistic regression).
- validity additionally relies on well-specified outcome model


```python
# logistic regression 
outcome_model = LogisticRegression(max_iter=1000)
outcome_model.fit(df[adjustment_set + ['coupon'] ], df['conversion'])

y_hat_1 = outcome_model.predict_proba(df[adjustment_set + ['coupon'] ].assign(coupon=1))[:, 1]
y_hat_0 = outcome_model.predict_proba(df[adjustment_set + ['coupon'] ].assign(coupon=0))[:, 1]
ate_lr = (y_hat_1-y_hat_0).mean()
print("logistic regression estimated ATE:", ate_lr)
print("true average treatment effect:", df['ite'].mean()) 
```

    logistic regression estimated ATE: 0.011909259494919312
    true average treatment effect: 0.03908395919621665


#### 3. Doubly Robust Estimation

To some extent, the logical next step is to employ doubly robust estimation to combine both worlds. As name suggests, the idea is to fit both a treatment and outcome model in an orthogonal way to ensure validity even if one model is misspecified. The double ml approach extends this idea using data splitting to allow the use of more sophisticated models without worrying about overfitting.


```python
# doubly robust estimation (here using doubleml)

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
```

    double logistic regression estimated ATE:
                 coef  std err        t    P>|t|     2.5 %   97.5 %
    coupon  0.011841  0.00953  1.24248  0.21406 -0.006838  0.03052
    true average treatment effect: 0.03908395919621665


To highlight the idea, we first employ double robust estimation using logistic regression for both, treatment and outcome model.


```python
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
```

    double random forest estimated ATE:
                 coef   std err         t     P>|t|     2.5 %    97.5 %
    coupon  0.028855  0.009354  3.084947  0.002036  0.010523  0.047188
    true average treatment effect: 0.03908395919621665


Finally, we estimate the average treatment effect using double ml by fitting a random forest to both, treatment and outcome model. As a sanity check, note, that the true average treatment effect is contained in the 95% confidence interval. For details on other common choices of models and hyperparameter tuning, we refer to the recent CLeaR paper 'Hyperparameter Tuning for Causal Inference with Double Machine Learning: A Simulation Study' by Philipp Bach et al.


### Treatment Effect Heterogeneity

#### Group Average Treatment Effects (GATE)

Moreover, we can now use our fitted model to guide the decision on which customers should receive a coupon. First, we compute the ATE grouped by membership level (GATE).


```python
# GATE estimation by membership level 
membership_level = df[['membership_level_0', 'membership_level_1', 'membership_level_2']].astype('bool')
gate = dml_irm_rf.gate(groups=membership_level)
print("double random forest estimated GATE:\n", gate)

for level in ['membership_level_0', 'membership_level_1', 'membership_level_2']:
    print(f"true GATE {level}: {df['ite'][df[level] == 1].mean()}")
```

    double random forest estimated GATE:
     ================== DoubleMLBLP Object ==================
    
    ------------------ Fit summary ------------------
                            coef   std err         t     P>|t|    [0.025    0.975]
    membership_level_0  0.045372  0.014976  3.029727  0.002448  0.016020  0.074724
    membership_level_1  0.012151  0.014396  0.844035  0.398650 -0.016065  0.040367
    membership_level_2  0.010817  0.021094  0.512788  0.608100 -0.030527  0.052161
    true GATE membership_level_0: 0.06297299239315964
    true GATE membership_level_1: 0.025669719591681534
    true GATE membership_level_2: 0.018707844852900623


While there seems to be a positive effect in all groups, targeting customers with membership level zero seems particularly valuable.

#### Conditional Average Treatment Effects (CATE)

We can similarly investigate the effect of specific (continuous) covariates by estimating the conditional treatment effect (CATE). The idea here is to fit a basis expansion (e.g., splines) to the covariate of interest and project the treatment effect onto this basis.


```python
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
plt.title('CATE X10')
plt.xlabel('x')
_ =  plt.ylabel('Effect and 95%-CI')

plt.show()
```

    ================== DoubleMLBLP Object ==================
    
    ------------------ Fit summary ------------------
           coef   std err         t         P>|t|    [0.025    0.975]
    0  0.028391  0.009330  3.043032  2.342078e-03  0.010105  0.046678
    1 -0.055564  0.009405 -5.907719  3.468773e-09 -0.073998 -0.037130



    
![png](coupon_files/coupon_20_1.png)
  


```python
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
plt.title('CATE X11')
plt.xlabel('x')
_ =  plt.ylabel('Effect and 95%-CI')

plt.show()
```

    ================== DoubleMLBLP Object ==================
    
    ------------------ Fit summary ------------------
           coef   std err         t         P>|t|    [0.025    0.975]
    0  0.047347  0.011593  4.084164  4.423577e-05  0.024626  0.070069
    1 -0.043476  0.008802 -4.939460  7.833932e-07 -0.060727 -0.026225
    2 -0.019102  0.006135 -3.113364  1.849676e-03 -0.031127 -0.007077



    
![png](coupon_files/coupon_23_1.png)
    


The covariate X11 seems to have a non-linear effect on the conversion rate.


### Conclusion

This analysis demonstrates a pipeline for estimating causal effects from observational data. By leveraging backdoor adjustments and DoubleML, we obtain interpretable and policy-relevant results, despite the limitations of non-randomized designs.

Further steps could include investigating counterfactuals (would the customer have been converted even without a coupon?) and evaluating different policies of assigning coupons based on pre-treatment information (e.g., no coupon if X10>2 or X11>1).


**Comment**: All methods seem to underestimate the true average treatment effect. To some extent, this is not necessarily surprising and might indicate confounding/selection bias by purchase intent. In fact, highly motivated buyers (who would convert anyway) might be less likely to use a coupon, because they don’t need the discount.  Conversely, people who choose to use the coupon might be more hesitant or price-sensitive customers,
