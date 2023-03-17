import pandas as pd
import numpy as np
import gc

from IPython.display import clear_output

from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
from lightgbm import LGBMClassifier
from joblib import dump, load

def get_lvl2_model_preds(result, features, cat_feats):

    min_age = result['age'].min()
    max_age = result['age'].max()
    
    age_cats = [(min_age, 24), (25, 44), (45, 64), (65, max_age)]
    
    preds = np.zeros(0)
    customers_id = np.zeros(0)
    article_id = np.zeros(0)
    target = np.zeros(0)
    
    for cat_range in age_cats:
        cat_min_age = cat_range[0]
        cat_max_age = cat_range[1]
        
        print(f'Обучаем модель для возрастной группы {int(cat_min_age)}-{int(cat_max_age)}')
        
        X_train = result[features].loc[(result[features]['age'] >= cat_min_age) & (result[features]['age'] <= cat_max_age)]
        y_train = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['target']
        X_train[cat_feats] = X_train[cat_feats].astype('category')
        
        eval_data = Pool(X_train, y_train, cat_features=cat_feats)
        
        # CatBoostClassifier
        clf = CatBoostClassifier(iterations=150, eval_metric='Logloss', scale_pos_weight=50, learning_rate=0.7, use_best_model=True, random_seed=42)
        clf.fit(X_train, y_train, cat_features=cat_feats, eval_set=eval_data, early_stopping_rounds=5, verbose=True)
        
        # сохранаяем модель
        fname = f'catboost_model/{int(cat_min_age)}-{int(cat_max_age)}_cat_clf_catboost_model'
        clf.save_model(fname, format="cbm")
        
        # получаем предсказания модели на той же выборке
        preds_spam = clf.predict_proba(X_train)
        preds_spam = preds_spam[:,1]
        
        customers_id_spam = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['customer_id_short'].values.astype(np.int32)
        article_id_spam = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['article_id_short'].values.astype(np.int32)
        target_spam = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['target'].values
    
        customers_id = np.append(customers_id, customers_id_spam)
        article_id = np.append(article_id, article_id_spam)
        target = np.append(target, target_spam)
        preds = np.append(preds, preds_spam)
        
        del X_train;
        del y_train;
        del eval_data;
        del clf;
        del preds_spam;
        del customers_id_spam;
        del target_spam;
        gc.collect()
        
        clear_output(wait=True)
        
    return customers_id, article_id, target, preds
    
def get_lvl2_LGBM_model_preds(result, features, cat_feats):

    min_age = result['age'].min()
    max_age = result['age'].max()
    
    age_cats = [(min_age, 24), (25, 44), (45, 64), (65, max_age)]
    
    preds = np.zeros(0)
    customers_id = np.zeros(0)
    article_id = np.zeros(0)
    target = np.zeros(0)
    
    for cat_range in age_cats:
        cat_min_age = cat_range[0]
        cat_max_age = cat_range[1]
        
        print(f'Обучаем модель для возрастной группы {int(cat_min_age)}-{int(cat_max_age)}')
        
        X_train = result[features].loc[(result[features]['age'] >= cat_min_age) & (result[features]['age'] <= cat_max_age)]
        y_train = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['target']
        X_train[cat_feats] = X_train[cat_feats].astype('category')
        
        
        # LGBMClassifier
        clf = LGBMClassifier(objective="binary",
                            learning_rate=0.01,
                            colsample_bytree=0.9,
                            subsample=0.8,
                            random_state=42,
                            n_estimators=550,
                            num_leaves=91) ##############
        clf.fit(X_train, y_train, categorical_feature=cat_feats, verbose=False)
        
        # сохранаяем модель
        fname = f'LGBM_model/{int(cat_min_age)}-{int(cat_max_age)}_cat_clf_LGBM_model'
        dump(clf, fname)
        
        # получаем предсказания модели на той же выборке
        preds_spam = clf.predict_proba(X_train)
        preds_spam = preds_spam[:,1]
        
        customers_id_spam = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['customer_id_short'].values.astype(np.int32)
        article_id_spam = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['article_id_short'].values.astype(np.int32)
        target_spam = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['target'].values
    
        customers_id = np.append(customers_id, customers_id_spam)
        article_id = np.append(article_id, article_id_spam)
        target = np.append(target, target_spam)
        preds = np.append(preds, preds_spam)
        
        del X_train;
        del y_train;
        del clf;
        del preds_spam;
        del customers_id_spam;
        del target_spam;
        gc.collect()
        
    return customers_id, article_id, target, preds
    

def get_lvl2_model_validation_preds(result, features, cat_feats):

    min_age = 16
    max_age = 96
    
    age_cats = [(min_age, 24), (25, 44), (45, 64), (65, max_age)]
    
    preds = np.zeros(0)
    customers_id = np.zeros(0)
    article_id = np.zeros(0)
    target = np.zeros(0)
    
    for cat_range in age_cats:
        cat_min_age = cat_range[0]
        cat_max_age = cat_range[1]
        
        print(f'Получаем предсказания для возрастной группы {int(cat_min_age)}-{int(cat_max_age)}')
        
        X_train = result[features].loc[(result[features]['age'] >= cat_min_age) & (result[features]['age'] <= cat_max_age)]
        y_train = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['target']
        X_train[cat_feats] = X_train[cat_feats].astype('category')
        
        eval_data = Pool(X_train, y_train, cat_features=cat_feats)
        
        # загружаем CatBoostClassifier
        clf = CatBoostClassifier(iterations=150, eval_metric='Logloss', scale_pos_weight=50, learning_rate=0.7, use_best_model=True, random_seed=42)
        fname = f'catboost_model/{int(cat_min_age)}-{int(cat_max_age)}_cat_clf_catboost_model'
        clf.load_model(fname)
        
        # model preds
        preds_spam = clf.predict_proba(X_train)
        preds_spam = preds_spam[:,1]
        
        customers_id_spam = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['customer_id_short'].values.astype(np.int32)
        article_id_spam = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['article_id_short'].values.astype(np.int32)
        target_spam = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['target'].values
    
        customers_id = np.append(customers_id, customers_id_spam)
        article_id = np.append(article_id, article_id_spam)
        target = np.append(target, target_spam)
        preds = np.append(preds, preds_spam)
        
        del X_train;
        del y_train;
        del eval_data;
        del clf;
        del preds_spam;
        del customers_id_spam;
        del target_spam;
        gc.collect()
        
    return customers_id, article_id, target, preds
    
def get_lvl2_model_validation_LGBM_preds(result, features, cat_feats):

    min_age = 16
    max_age = 96
    
    age_cats = [(min_age, 24), (25, 44), (45, 64), (65, max_age)]
    
    preds = np.zeros(0)
    customers_id = np.zeros(0)
    article_id = np.zeros(0)
    target = np.zeros(0)
    
    for cat_range in age_cats:
        cat_min_age = cat_range[0]
        cat_max_age = cat_range[1]
        
        print(f'Получаем предсказания для возрастной группы {int(cat_min_age)}-{int(cat_max_age)}')
        
        X_train = result[features].loc[(result[features]['age'] >= cat_min_age) & (result[features]['age'] <= cat_max_age)]
        y_train = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['target']
        X_train[cat_feats] = X_train[cat_feats].astype('category')
        
        eval_data = Pool(X_train, y_train, cat_features=cat_feats)
        
        # загружаем CatBoostClassifier
        fname = f'LGBM_model/{int(cat_min_age)}-{int(cat_max_age)}_cat_clf_LGBM_model'
        clf = load(fname)
        
        # model preds
        preds_spam = clf.predict_proba(X_train)
        preds_spam = preds_spam[:,1]
        
        customers_id_spam = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['customer_id_short'].values.astype(np.int32)
        article_id_spam = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['article_id_short'].values.astype(np.int32)
        target_spam = result.loc[(result['age'] >= cat_min_age) & (result['age'] <= cat_max_age)]['target'].values
    
        customers_id = np.append(customers_id, customers_id_spam)
        article_id = np.append(article_id, article_id_spam)
        target = np.append(target, target_spam)
        preds = np.append(preds, preds_spam)
        
        del X_train;
        del y_train;
        del eval_data;
        del clf;
        del preds_spam;
        del customers_id_spam;
        del target_spam;
        gc.collect()
        
    return customers_id, article_id, target, preds
