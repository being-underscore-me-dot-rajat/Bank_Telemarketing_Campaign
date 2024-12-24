import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


train_df=pd.read_csv('train.csv')
imp_cols=train_df.isnull().sum()[train_df.isnull().sum()>0].index
Mode_impute=SimpleImputer(strategy='most_frequent')
train_df[imp_cols]=pd.DataFrame(Mode_impute.fit_transform(train_df[imp_cols]),columns=imp_cols,index=train_df.index)

X=train_df[train_df.columns[:-1]]
y=pd.DataFrame(train_df[train_df.columns[-1]])

X['duration_log'] = np.log(X['duration'] + 1)

X['campaign_inv'] = 1/(X['campaign'])

X['previous_log'] = np.log(X['previous']+2)
eng_cols=[]
def age_binner(age):
    if age < 30:
        return 'Young'
    elif 30 <= age < 60:
        return 'Middle-aged'
    else:
        return 'Old'
X['age_binned'] = X['age'].map(age_binner)
eng_cols.append('age_binned')


X['ongoing_loan']=((X['housing']=='yes')|(X['loan']=='yes')).astype(int)
eng_cols.append('ongoing_loan')

def balance_binner(balance):
    if balance <= 0:
        return 'Negative'
    elif 0<= balance < 3000:
        return 'Low'
    elif 3000 <= balance < 10000:
        return 'Average'
    elif 10000<= balance < 50000:
        return 'Upper Average'
    else:
        return 'High'
X['balance_binned'] = X['balance'].map(balance_binner)
eng_cols.append('balance_binned')

X['financially_weak']=((X['balance']<1000)&X['ongoing_loan']==1 ).astype(int)
X['financially_strong']=((X['balance']>3000)&X['ongoing_loan']==0 ).astype(int)
eng_cols.append('financially_weak')
eng_cols.append('financially_strong')

X['risky_customers']=((X['default']=='yes')&(X['ongoing_loan']==1)).astype(int)
eng_cols.append('risky_customers')

def campaign_binner(no):
    if no <= 5:
        return 'Short'
    else:
        return 'Long'
X['campaign_binned'] = X['campaign'].map(campaign_binner)
X['previous_binned'] = X['previous'].map(campaign_binner)
eng_cols.append('campaign_binned')
eng_cols.append('previous_binned')

def duration_binner(days):
    if days <= 150:
        return 'Short'
    else:
        return 'Long'

X['duration_binned'] = X['duration'].map(duration_binner)
eng_cols.append('duration_binned')

X['possibly_interested']=(((X['campaign']>1)|(X['previous']>1))&(X['duration']>100))
eng_cols.append('possibly_interested')

X['ageXmarital']=X['age_binned']+" "+X['marital']
eng_cols.append('ageXmarital')

def job_binner(job):
    if job in ['blue-collar','technician']:
        return 'Labour_class'
    elif job in ['unemployed','student','retired','housemaid']:
        return 'Non-Working'
    elif job in ['admin.','management','services']:
        return 'White_collar'
    else:
        return 'self_employed'

X['job_binned'] = X['job'].map(job_binner)
eng_cols.append('job_binned')



X['last contact date'] = pd.to_datetime(X['last contact date'])
X['day'] = X['last contact date'].dt.day_name()  
X['date'] = X['last contact date'].dt.day        
X['month'] = X['last contact date'].dt.month


def day_binner(day):
    if day in ['Saturday','Sunday']:
        return 'Weekend'
    else:
        return 'Weekday'

X['day_binned'] = X['day'].map(day_binner)
eng_cols.append('day_binned')

def date_binner(date):
    if date < 10:
        return 'Month_start'
    elif 10<date<25:
        return 'Mid_month'
    else:
        return 'Month_end'

X['date_binned'] = X['date'].map(date_binner)
eng_cols.append('date_binned')

def month_binner(month):
    if month <= 3:
        return 'First_Quarter'
    elif 4<=month<7:
        return 'Second_Quarter'
    elif 7<=month<10:
        return 'Third_Quarter'
    else:
        return 'Fourth_Quarter'

X['month_binned'] = X['month'].map(month_binner)

drp_cols=['duration','campaign','previous','last contact date']
X.drop(drp_cols,axis=1,inplace=True)

# Grouping the Object and Integer columns

objcols=X.select_dtypes(include='object').columns.tolist()
numcols=X.select_dtypes(exclude='object').columns.tolist()

#Grouping the columns into one hot and ordinal categories

oh_cols=X[objcols].nunique()[X[objcols].nunique()>=3].index.tolist()
ord_cols=[x for x in objcols if x not in oh_cols]

oh_encoder=OneHotEncoder(sparse_output=False)
ord_encoder=OrdinalEncoder()

#Applying One Hot encoding and then grouping the Datasets
oh_df_train=pd.DataFrame(oh_encoder.fit_transform(X[oh_cols]),columns=oh_encoder.get_feature_names_out(),index=X.index)
X=X.join(oh_df_train).drop(oh_cols,axis=1)

# Applying ordinal encoding
X[ord_cols]=pd.DataFrame(ord_encoder.fit_transform(X[ord_cols]),columns=ord_cols,index=X.index)

#Label encoding the target variable
lb_encoder=LabelEncoder()
y=pd.DataFrame(lb_encoder.fit_transform(y),index=X.index)


xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8,random_state=8)
ytrain=np.array(ytrain[0])
ytest=np.array(ytest[0])


xgbclf = XGBClassifier(
    scale_pos_weight=len(ytrain) / (2 * sum(ytrain)),
    random_state=132
)

param_grid = {
    'n_estimators': [700,200,500],
    'max_depth': [3, 6, 9,12],
    'learning_rate': [0.001,0.002,0.01, 0.1, 0.2,0.5],
    'subsample': [0.2,0.5, 0.7, 1],
    'colsample_bytree': [0.2,0.5, 0.7, 1],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
}

model = RandomizedSearchCV(
    estimator=xgbclf,
    param_distributions=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='f1_macro',
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ('rbs',RobustScaler(),numcols)
    ]
)

xgbclf_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', model)
])

xgbclf_pipeline.fit(xtrain, ytrain)