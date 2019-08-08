# for basic mathematical operations
import numpy as np 
import pandas as pd 

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# for defining a path for the dataset
import os
print(os.listdir("../input"))

# reading the dataset

train = pd.read_csv('../input/X_train.csv')
sub = pd.read_csv('../input/sample_submission.csv')
test = pd.read_csv('../input/X_test.csv')
y_train = pd.read_csv('../input/y_train.csv')

# getting the shapes of the datasets
print("Shape of train :", train.shape)
print("Shape of test :", test.shape)
print("Shape of y_train :", y_train.shape)

# checking the x_train

train.head()

# checking the test head

test.head()


# describing the train dataset

train.describe()



# checking if x_train and x_test contains any NULL values in the dataset

print("Null Values in the training set :")
print(train.isnull().sum())
print("NULL Values in the testing set :")
print(test.isnull().sum())



# checking the head of the y_train set

y_train.head()




# checking the unique group_id 

print("Unique Elements present in the Group-id :", y_train['group_id'].nunique())
print("Unique Elements present in the Series-id :",y_train['series_id'].nunique())



# checking the different types of surface

y_train['surface'].value_counts()


# plotting a pie chart

size = [779, 732, 607, 514, 363, 308, 297, 189, 21]
colors = ['pink', 'yellow', 'lightgreen', 'lightblue', 'purple', 'violet', 'crimson', 'darkred',
          'blue']
labels = 'concrete', 'soft_pvc', 'wood','tiled','fine_concrete','hard_tiles_large_space','soft_tiles','carpet','hard_tiles'

my_circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(size, colors = colors, labels = labels, autopct = '%.2f%%')
plt.title('Donut Chart to Represent different Surface Types', fontsize = 30)
plt.axis('off')
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.legend()
plt.show()



series_1 = train.head(128)
series_1.head()

# plotting Time-Series Graph for all the attributes present in the training set

plt.rcParams['figure.figsize'] = (15, 15)
for i, col in enumerate(series_1.columns[3:]):
    plt.subplot(3, 4, i + 1)
    plt.plot(series_1[col])
    plt.title(col)




series_2 = train.head(128)
series_2.head()

# plotting Time-Series Graph for all the attributes present in the training set

plt.rcParams['figure.figsize'] = (15, 15)
for i, col in enumerate(series_2.columns[3:]):
    plt.subplot(3, 4, i + 1)
    plt.plot(series_2[col])
    plt.title(col)


# checking the distribution of all the attributes in the training and testing data

plt.rcParams['figure.figsize'] = (15, 15)
for i, col in enumerate(train.columns[3:]):
    plt.subplot(3, 4, i + 1)
    plt.hist(train[col], color='blue', bins=100)
    plt.hist(test[col], color='green', bins=100)
    plt.title(col)


# heatmap of correlation for the attributes of the training set

f,ax = plt.subplots(figsize=(12,6))
m = train.iloc[:,3:].corr()
sns.heatmap(m, annot=True, linecolor='darkblue', linewidths=.1, cmap = 'Reds', fmt= '.1f',ax=ax)




def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,5,figsize=(16,8))

    for feature in features:
        i += 1
        plt.subplot(2,5,i)
        sns.distplot(df1[feature], hist=False, label=label1)
        sns.distplot(df2[feature], hist=False, label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();

def plot_feature_class_distribution(classes,tt, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(5,2,figsize=(16,24))
    
    for feature in features:
        i += 1
        plt.subplot(5,2,i)
        for clas in classes:
            ttc = tt[tt['surface']==clas]
            sns.distplot(ttc[feature], hist=False,label=clas)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();



# https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1
def quaternion_to_euler(x, y, z, w):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z


def perform_euler_factors_calculation(df):
    df['total_angular_velocity'] = np.sqrt(np.square(df['angular_velocity_X']) + np.square(df['angular_velocity_Y']) + np.square(df['angular_velocity_Z']))
    df['total_linear_acceleration'] = np.sqrt(np.square(df['linear_acceleration_X']) + np.square(df['linear_acceleration_Y']) + np.square(df['linear_acceleration_Z']))
    df['total_xyz'] = np.sqrt(np.square(df['orientation_X']) + np.square(df['orientation_Y']) +
                              np.square(df['orientation_Z']))
    df['acc_vs_vel'] = df['total_linear_acceleration'] / df['total_angular_velocity']
    
    x, y, z, w = df['orientation_X'].tolist(), df['orientation_Y'].tolist(), df['orientation_Z'].tolist(), df['orientation_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)
    
    df['euler_x'] = nx
    df['euler_y'] = ny
    df['euler_z'] = nz
    
    df['total_angle'] = np.sqrt(np.square(df['euler_x']) + np.square(df['euler_y']) + np.square(df['euler_z']))
    df['angle_vs_acc'] = df['total_angle'] / df['total_linear_acceleration']
    df['angle_vs_vel'] = df['total_angle'] / df['total_angular_velocity']
    return df



def perform_feature_engineering(df):
    df_out = pd.DataFrame()
    
    def mean_change_of_abs_change(x):
        return np.mean(np.diff(np.abs(np.diff(x))))

    def mean_abs_change(x):
        return np.mean(np.abs(np.diff(x)))
    
    for col in df.columns:
        if col in ['row_id', 'series_id', 'measurement_number']:
            continue
        df_out[col + '_mean'] = df.groupby(['series_id'])[col].mean()
        df_out[col + '_min'] = df.groupby(['series_id'])[col].min()
        df_out[col + '_max'] = df.groupby(['series_id'])[col].max()
        df_out[col + '_std'] = df.groupby(['series_id'])[col].std()
        df_out[col + '_mad'] = df.groupby(['series_id'])[col].mad()
        df_out[col + '_med'] = df.groupby(['series_id'])[col].median()
        df_out[col + '_skew'] = df.groupby(['series_id'])[col].skew()
        df_out[col + '_range'] = df_out[col + '_max'] - df_out[col + '_min']
        df_out[col + '_max_to_min'] = df_out[col + '_max'] / df_out[col + '_min']
        df_out[col + '_mean_abs_change'] = df.groupby('series_id')[col].apply(mean_abs_change)
        df_out[col + '_mean_change_of_abs_change'] = df.groupby('series_id')[col].apply(mean_change_of_abs_change)
        df_out[col + '_abs_max'] = df.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))
        df_out[col + '_abs_min'] = df.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))
        df_out[col + '_abs_mean'] = df.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(x)))
        df_out[col + '_abs_std'] = df.groupby('series_id')[col].apply(lambda x: np.std(np.abs(x)))
        df_out[col + '_abs_avg'] = (df_out[col + '_abs_min'] + df_out[col + '_abs_max'])/2
        df_out[col + '_abs_range'] = df_out[col + '_abs_max'] - df_out[col + '_abs_min']

    return df_out



train = perform_euler_factors_calculation(train)
test = perform_euler_factors_calculation(test)

# checking the shapes of the datset
print("Shape of train:", train.shape)
print("Shape of test:", test.shape)


features = train.columns.values[13:23]
plot_feature_distribution(train, test, 'train', 'test', features)

classes = (y_train['surface'].value_counts()).index
tt = train.merge(y_train, on='series_id', how='inner')
plot_feature_class_distribution(classes, tt, features)



# performing feature engineering on the datasets

x_train = perform_feature_engineering(train)
x_test = perform_feature_engineering(test)

# checking the shapes
print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_train :", y_train.shape)



# applying label encoder to the surface attribute

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train['surface'] = le.fit_transform(y_train['surface'])





# replacing all the nan, and infinities with zero

x_train.fillna(0, inplace = True)
x_train.replace(-np.inf, 0, inplace = True)
x_train.replace(np.inf, 0, inplace = True)
x_test.fillna(0, inplace = True)
x_test.replace(-np.inf, 0, inplace = True)
x_test.replace(np.inf, 0, inplace = True)





# splitting the data into train and validation sets

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.3,
                                                      random_state = 0)

print("Shape of x_train :", x_train.shape)
print("Shape of x_valid :", x_valid.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_valid :", y_valid.shape)



# deleting series-id and group-id from y-train and y-valid

y_train = y_train.drop(['series_id', 'group_id'], axis = 1)
y_valid = y_valid.drop(['series_id', 'group_id'], axis = 1)


# modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)

y_pred_rf = model_rf.predict(x_valid)

# evaluating the model
print("Training Accuracy :", model_rf.score(x_train, y_train))
print("Validation Accuracy :", model_rf.score(x_valid, y_valid))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred_rf)
print(cm)

# classification report
cr = classification_report(y_valid, y_pred_rf)
print(cr)



# modelling
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model_xgb = XGBClassifier()
model_xgb.fit(x_train, y_train)

y_pred_xgb = model_xgb.predict(x_valid)

# evaluating the model
print("Training Accuracy :", model_xgb.score(x_train, y_train))
print("Validation Accuracy :", model_xgb.score(x_valid, y_valid))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred_xgb)
print(cm)

# classification report
cr = classification_report(y_valid, y_pred_xgb)
print(cr)


# boosting the predictions of the model

boosted_predictions = 0.4*y_pred_rf + 0.6*y_pred_xgb
boosted_predictions
