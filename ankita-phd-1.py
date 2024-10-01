import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Multiply
import keras_tuner as kt

# ================================
# Load and Preprocess the Datasets
# ================================
def load_and_preprocess_datasets():
    # Load datasets
    bot_iot_df = pd.read_csv('BoT_IOT.csv')
    unsw_nb15_df = pd.read_csv('UNSW_NB15.csv')

    # Preprocess BoT IOT
    bot_iot_df.fillna(bot_iot_df.mean(), inplace=True)
    if bot_iot_df.select_dtypes(include=['object']).shape[1] > 0:
        bot_iot_df = pd.get_dummies(bot_iot_df)

    # Preprocess UNSW-NB15
    unsw_nb15_df.fillna(unsw_nb15_df.mean(), inplace=True)
    if unsw_nb15_df.select_dtypes(include=['object']).shape[1] > 0:
        unsw_nb15_df = pd.get_dummies(unsw_nb15_df)

    # Normalize both datasets
    scaler = StandardScaler()
    
    # Scaling the BoT IOT dataset (excluding the target column if present)
    X_bot_iot = bot_iot_df.iloc[:, :-1]
    y_bot_iot = bot_iot_df.iloc[:, -1]  # Adjust if target column is elsewhere
    X_bot_iot_scaled = scaler.fit_transform(X_bot_iot)

    # Scaling the UNSW-NB15 dataset (excluding the target column)
    X_unsw_nb15 = unsw_nb15_df.iloc[:, :-1]
    y_unsw_nb15 = unsw_nb15_df.iloc[:, -1]  # Adjust if target column is elsewhere
    X_unsw_nb15_scaled = scaler.fit_transform(X_unsw_nb15)

    # Merging datasets
    X_combined = pd.concat([pd.DataFrame(X_bot_iot_scaled), pd.DataFrame(X_unsw_nb15_scaled)], axis=0)
    y_combined = pd.concat([pd.Series(y_bot_iot), pd.Series(y_unsw_nb15)], axis=0)

    # Split the UNSW-NB15 and  dataset into training and test sets
    Xtrn, Xtst, ytrn, ytst = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    return Xtrn, Xtst, ytrn, ytst

# ================================
# PART 1: APPLY PCA FOR LOW-RANK FACTORIZATION FOR ML MODELS
# ================================
def apply_pca(Xtrn, Xtst, rank):
    """
    Applies PCA to reduce the dimensionality of the input data.
    """
    pca = PCA(n_components=rank)
    Xtrn_pca = pca.fit_transform(Xtrn)
    Xtst_pca = pca.transform(Xtst)
    
    return Xtrn_pca, Xtst_pca

# ================================
# PART 2: HYPERPARAMETER TUNING FOR MACHINE LEARNING MODELS USING GRIDSEARCHCV
# ================================
def hyperparameter_tuning_ml_models(Xtrn, Xtst, ytrn, ytst):
    # Hyperparameter grid for each model
    prm_grds = {
        'Logistic Regression': {'C': [0.1, 1, 10]},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
        'Low-Rank SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'k-NN': {'n_neighbors': [3, 5, 7]},
        'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
    }

    # usins the python dictionary with objective of storing best performing models
    bst_mdls = {}

    # Train and evaluate each model using GridSearchCV
    for mdl_nm, prm_grd in prm_grds.items():
        if mdl_nm == 'Logistic Regression':
            mdl = LogisticRegression()
        elif mdl_nm == 'Random Forest':
            mdl = RandomForestClassifier(random_state=42)
        elif mdl_nm == 'Low-Rank SVM':
            mdl = SVC()
        elif mdl_nm == 'k-NN':
            mdl = KNeighborsClassifier()
        elif mdl_nm == 'Gradient Boosting':
            mdl = GradientBoostingClassifier(random_state=42)
        
        grd_srch = GridSearchCV(mdl, prm_grd, cv=5, n_jobs=-1, scoring='accuracy')
        grd_srch.fit(Xtrn, ytrn)
        bst_mdls[mdl_nm] = grd_srch.best_estimator_

        prdy = grd_srch.best_estimator_.predict(Xtst)
        acc = accuracy_score(ytst, prdy)
        print(f'{mdl_nm} Best Params: {grd_srch.best_params_}, Accuracy: {acc * 100:.2f}%')

# ================================
# PART 3: CNN HYPERPARAMETER TUNING USING KERASTUNER
# ================================
def bldCNN_mdl(hp):
    inp_lyr = Input(shape=(28, 28, 1))
    
    # CNN Part with Tuning
    x = Conv2D(filters=hp.Int('filters', min_value=32, max_value=128, step=32),
               kernel_size=(3, 3), activation='relu')(inp_lyr)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    
    # Dense Layer
    x = Dense(units=hp.Int('units', min_value=64, max_value=256, step=64), activation='relu')(x)
    
    # Final Output Layer
    opt_lyr = Dense(1, activation='sigmoid')(x)  # Binary classification
    
    # Building the attack classification-based model optimized file 
    mdl = Model(inputs=inp_lyr, outputs=opt_lyr)
    mdl.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [0.001, 0.01, 0.1])),
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    return mdl

def tnnCNN_mdl(Xtrn_reshaped, ytrn):
    tuner = kt.RandomSearch(
        bldCNN_mdl,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='cnn_tuning',
        project_name='cnn_project')

    tuner.search(Xtrn_reshaped, ytrn, epochs=10, validation_split=0.2)
    mdl_bst = tuner.get_bst_mdls(num_models=1)[0]
    return mdl_bst

# ================================
# PART 4: CNN-MLP WITH LOW-RANK FACTORIZATION TUNING USING KERASTUNER
# ================================
def bldCNNMLP_mdl(hp):
    inp_lyr = Input(shape=(28, 28, 1))

    # CNN Part with Tuning
    x = Conv2D(filters=hp.Int('filters', min_value=32, max_value=128, step=32),
               kernel_size=(3, 3), activation='relu')(inp_lyr)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    # Low-rank factorization in MLP
    dns_inpdim = x.shape[1]
    rank = hp.Int('rank', min_value=32, max_value=128, step=32)
    W1 = Dense(rank, activation='relu', use_bias=False)(x)
    W2 = Dense(dns_inpdim, activation='relu', use_bias=False)(W1)
    factorized_output = Multiply()([x, W2])

    # Final dense layer for classification
    opt_lyr = Dense(1, activation='sigmoid')(factorized_output)

    mdl = Model(inputs=inp_lyr, outputs=opt_lyr)
    mdl.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [0.001, 0.01, 0.1])),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return mdl

def tunCNNMLP_mdl(Xtrn_reshaped, ytrn):
    tuner = kt.RandomSearch(
        bldCNNMLP_mdl,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='cnn_mlp_tuning',
        project_name='cnn_mlp_project')

    tuner.search(Xtrn_reshaped, ytrn, epochs=10, validation_split=0.2)
    mdl_bst = tuner.get_bst_mdls(num_models=1)[0]
    return mdl_bst

# ================================
# MAIN EXECUTION
# ================================
if __name__ == '__main__':
    # Loading with the preprocessing of the Iot attack based datasets
    Xtrn, Xtst, ytrn, ytst = load_and_preprocess_datasets()

    # Step 1: Apply PCA for low-rank factorization
    Xtrn_pca, Xtst_pca = apply_pca(Xtrn, Xtst, rank=50)

    # Step 2: Hyperparameter tuning on machine learning models
    hyperparameter_tuning_ml_models(Xtrn_pca, Xtst_pca, ytrn, ytst)

    # Reshaping data as required for the input for the modeling techniques utilizing CNN and CNN-MLP
    Xtrn_reshaped = Xtrn.reshape(-1, 28, 28, 1)
    Xtst_reshaped = Xtst.reshape(-1, 28, 28, 1)

    # Step 3: Tune and train the CNN model
    best_cnn_model = tnnCNN_mdl(Xtrn_reshaped, ytrn)
    lss_cnn, accu_cnn = best_cnn_model.evaluate(Xtst_reshaped, ytst)
    print(f'Tuned CNN Accuracy: {accu_cnn * 100:.2f}%')

    # Step 4: Tune and train the CNN-MLP with low-rank factorization
    best_cnn_mlp_model = tunCNNMLP_mdl(Xtrn_reshaped, ytrn)
    cnn_mlp_loss, cnn_mlp_acc = best_cnn_mlp_model.evaluate(Xtst_reshaped, ytst)
    print(f'Tuned CNN-MLP with Low-Rank Factorization Accuracy: {cnn_mlp_acc * 100:.2f}%')
