import json
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(
    font_scale=1.5,
    style="whitegrid",
    rc={'figure.figsize':(20,7)}
)

print("Row/Column Count = ",result.shape)
result.head() #.head(10)

result['HR_xn'].describe()
result.describe()
include =['object', 'float', 'int']
result.describe(include = include)

result.count()

result.info()


#another test comment
#test

#nulls

test = result[['Tm','HR_xn','W_xn']]
add= test.groupby('Tm').agg({'HR_xn':['sum','min','max'],'W_xn':['mean']})
add.columns= ["_".join(x) for x in add.columns.ravel()]
list(add.columns) 
print(add)


#rename columns result.rename(columns={"Area": "place_name"}, inplace=True)
# data.rename(columns=str.lower)
test['HR_xn'].max()
test['HR_xn'].min()

result.dtypes
result.Year.dtype
result['HR_xn']=result['HR_xn'].astype(float)
result['R_xn']=result['R_xn'].astype(float)

result['Lg'].apply(pd.value_counts).fillna(0)

result['HR_xn'].value_counts(bins=10)


# https://towardsdatascience.com/how-to-explore-and-visualize-a-dataset-with-python-7da5024900ef
#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

result.columns
result['HR_xn'].describe()
result['Lg'].describe()

sns.distplot(result['HR_xn'])

# https://help.gooddata.com/doc/en/reporting-and-dashboards/maql-analytical-query-language/maql-expression-reference/aggregation-functions/statistical-functions/predictive-statistical-use-cases/normality-testing-skewness-and-kurtosis
#Left Large = Postive, right large = Negative
print("Skewness: %f" % result['HR_xn'].skew())

#negative = heavy tail, postiive = light tails
print("Kurtosis: %f" % result['HR_xn'].kurt())


#scatter plot grlivarea/saleprice
var = 'HR_xn'
data = pd.concat([result['W_xn'], result[var]], axis=1)
data.plot.scatter(x=var, y='W_xn', ylim=(0,110))

#box plot overallqual/saleprice
var = 'Lg'
data = pd.concat([result['HR_xn'], result[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="HR_xn", data=data)
fig.axis(ymin=0, ymax=350)

var = 'Year'
data = pd.concat([result['HR_xn'], result[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="HR_xn", data=data)
fig.axis(ymin=0, ymax=700)
plt.xticks(rotation=30)

#correlation matrix - THIS IS AWESOME
corrmat = result.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'W_xn')['W_xn'].index
cm = np.corrcoef(result[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#scatterplot
sns.set()
cols = ['W_xn', 'HR_xn', 'BA_xn', 'R_xn']
sns.pairplot(result[cols], size = 2.5)
plt.show()

_ = result.set_index(pd.DatetimeIndex(result.Year)).groupby(
    pd.Grouper(freq='Y')
)['H_xn'].mean()

ax = _.rolling(5).mean().plot(figsize=(20,7),title='Hit Rate Over Time')
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0f}%'.format(x*100) for x in vals])
sns.despine()


# orders_with_sales_team = pd.merge(order_leads,sales_team,on=['Company Id','Company Name'])
ax = sns.distplot(result.groupby('H_xn')['HR_xn'].mean(),kde=False)
vals = ax.get_xticks()
ax.set_xticklabels(['{:,.0f}%'.format(x*100) for x in vals])
ax.set_title('Number of sales reps by conversion rate')
sns.despine()


def vertical_mean_line(x, **kwargs):
    ls = {"0":"-","1":"--"}
    plt.axvline(x.mean(), linestyle =ls[kwargs.get("label","0")], 
                color = kwargs.get("color", "r"))
    txkw = dict(size=15, color = kwargs.get("color", "r"))
    tx = "mean: {:.1f}%\n(std: {:.1f}%)".format(x.mean()*100,x.std()*100)
    label_x_pos_adjustment = 0.015 
    label_y_pos_adjustment = 20
    plt.text(x.mean() + label_x_pos_adjustment, label_y_pos_adjustment, tx, **txkw)

sns.set(
    font_scale=1.5,
    style="whitegrid"
)

_ = result.groupby('Year').agg({
    'H_xn': np.mean,
    'HR_xn': pd.Series.nunique
})
_.columns = ['Hits','HRs']

g = sns.FacetGrid(_, col="Tm", height=2, aspect=0.9, col_wrap=5)
g.map(sns.kdeplot, "Hits", shade=True)
g.set(xlim=(0, 0.35))
g.map(vertical_mean_line, "Hits")


def vertical_mean_line(x, **kwargs):
    ls = {"0":"-","1":"--"}
    plt.axvline(x.mean(), linestyle =ls[kwargs.get("label","0")], 
                color = kwargs.get("color", "r"))
    txkw = dict(size=15, color = kwargs.get("color", "r"))
    tx = "mean: {:.1f}%\n(std: {:.1f}%)".format(x.mean()*100,x.std()*100)
    label_x_pos_adjustment = 0.015 
    label_y_pos_adjustment = 20
    plt.text(x.mean() + label_x_pos_adjustment, label_y_pos_adjustment, tx, **txkw)


sns.set(
    font_scale=1.5,
    style="whitegrid"
)

aa = result.groupby('Year').agg({
    'H_xn': np.mean,
    'HR_xn': pd.Series.nunique
})
aa.columns = ['Hits','HRs']

g = sns.FacetGrid(aa, col='HRs', height=4, aspect=0.9, col_wrap=4)
g.map(sns.kdeplot, "Hits", shade=True)
g.set(xlim=(1000, 2000))
# g.map(vertical_mean_line, "conversion rate")

result['Date of Meal'] = result['HR_x']
result['Date of Meal'].value_counts().sort_index()

result['Bucket'] = pd.cut(
    result['HR_xn'],
    bins=[10,100,150,200,400],
    labels=['breakfast','lunch','dinner','Other']
)

result.head(2)


def plot_bars(data,x_col,y_col):
    data = data.reset_index()
    sns.set(
        font_scale=1.5,
        style="whitegrid",
        rc={'figure.figsize':(25,8)}
    )
    g = sns.barplot(x=x_col, y=y_col, data=data, color='royalblue')

    for p in g.patches:
        g.annotate(
            format(p.get_height(), '.5'),
            (p.get_x() + p.get_width() / 2, p.get_height()), 
            ha = 'center', 
            va = 'center', 
            xytext = (0, 10), 
            textcoords = 'offset points'
        )
        
    vals = g.get_yticks()
    g.set_yticklabels(['{:,.0f}'.format(x) for x in vals])

    sns.despine()

_ = result.groupby('Bucket').agg({'HR_xn': np.mean})
plot_bars(_,x_col='Bucket',y_col='HR_xn')


_ = result.groupby(['Lg']).agg(
{'HR_xn': np.median} #mean sum 
)
plot_bars(data=_,x_col='Lg',y_col='HR_xn')


def draw_heatmap(data,inner_row, inner_col, outer_row, outer_col, values):
    sns.set(font_scale=1)
    fg = sns.FacetGrid(
        data, 
        row=outer_row,
        col=outer_col, 
        margin_titles=True
    )

    position = left, bottom, width, height = 1.4, .2, .1, .6
    cbar_ax = fg.fig.add_axes(position) 

    fg.map_dataframe(
        draw_heatmap_facet, 
        x_col=inner_col,
        y_col=inner_row, 
        values=values, 
        cbar_ax=cbar_ax,
        vmin=0, 
        vmax=250
    )

    fg.fig.subplots_adjust(right=1.3)  
    plt.show()

def draw_heatmap_facet(*args, **kwargs):
    data = kwargs.pop('data')
    x_col = kwargs.pop('x_col')
    y_col = kwargs.pop('y_col')
    values = kwargs.pop('values')
    d = data.pivot(index=y_col, columns=x_col, values=values)
    annot = round(d,4).values
    cmap = sns.color_palette("RdYlGn",30)
    # cmap = sns.color_palette("PuBu",30) alternative color coding
    sns.heatmap(d, **kwargs, annot=annot, center=0, fmt=".1%", cmap=cmap, linewidth=.5)

draw_heatmap(
    data=result, 
    outer_row='Bucket',
    outer_col='Lg',
    inner_row='Bucket',
    inner_col='Lg',
    values='HR_xn'
)

#intersting#
print(result.to_html()) 