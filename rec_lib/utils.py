import numpy as np
import pandas as pd
import gc

from numba import jit, typeof, typed, types, prange

# снижение объема занимаемой датафреймом памяти
def reduce_mem_usage(df):

    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type != 'datetime64[ns]':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if col_type != 'datetime64[ns]':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        else:
            if col_type != 'datetime64[ns]':
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
# создание словаря количества покупок пользователем одной категории товара за транзакцию (избавляемся от дублей)
def create_transactions_dict():
    transactions_dict = {}
    with open('archive/transactions_train.csv') as f:
        lines = f.readlines()
        lines = lines[1:]
        for (i, line) in enumerate(lines):
            data = line.strip()
            if data not in transactions_dict:
                transactions_dict[data] = 1
            else:
                transactions_dict[data] += 1
                
    return transactions_dict    

# установка значения таргета    
@jit(nopython=True, fastmath=True)
def set_target(arr_actual, val):
    if isin(arr_actual, val):
        return 1
    return 0

# преобразование датафрейма покупателей    
def customers_prep(customers):
    customers = customers[list(customers)[:-1]]
    customers['FN'] = customers['FN'].fillna(0.0)
    customers['FN'] = customers['FN'].apply(str)
    customers['Active'] = customers['Active'].fillna(0.0)
    customers['Active'] = customers['Active'].apply(str)
    customers['club_member_status'] = customers['club_member_status'].fillna('NONE')
    customers['club_member_status'] = customers['club_member_status'].apply(str)
    customers['fashion_news_frequency'] = customers['fashion_news_frequency'].fillna('NONE')
    customers.loc[customers['fashion_news_frequency'] == 'None'] = 'NONE'
    customers['fashion_news_frequency'] = customers['fashion_news_frequency'].apply(str)
    
    age_mode = float(customers['age'].mode())
    # age_median = customers[customers['age'] != 'NONE']['age'].median()
    customers['age'] = customers['age'].fillna(age_mode)
    customers[customers['age'] == 'NONE'] = age_mode
    
    return customers

# преобразование датафрейма категорий    
def articles_prep(articles):
    
    art_feats = ['article_id',
            'product_type_no',
#             'product_type_name',
            'product_group_name',
            'graphical_appearance_name',
            'colour_group_name',
            'perceived_colour_value_name',
            'perceived_colour_master_name',
            'department_no',
#             'department_name',
            'index_name',
            'index_group_name',
            'garment_group_name']

    articles = articles[art_feats]
    articles['product_type_no'] = articles['product_type_no'].apply(str)
    articles['department_no'] = articles['department_no'].apply(str)
    
    return articles

# добавляем фичи    
def get_features(result, transactions):
    
    # features 1-2. сумма продаж и средняя сумма заказа категорий товара
    article_sum_buys = transactions.groupby(['article_id_short'])['price'].agg(['sum', 'mean'])
    result = result.merge(article_sum_buys, on='article_id_short', how='left')
    del article_sum_buys;
    gc.collect()
    result = result.rename(columns= {'sum': 'articles_buy_sum'})
    result = result.rename(columns= {'mean': 'articles_buy_avg'})
    
    # features 3-4. количество покупок категорий и единиц категорий, купленных покупателями
    customers_num_articles_buys = transactions.groupby(['customer_id_short', 'article_id_short'])['values'].agg(['sum', 'count'])
    result = result.merge(customers_num_articles_buys, on=['customer_id_short', 'article_id_short'], how='left')
    del customers_num_articles_buys;
    gc.collect()
    result = result.rename(columns= {'sum': 'customers_articles_num_sum'})
    result = result.rename(columns= {'count': 'customers_articles_num_count'})
    result['customers_articles_num_sum'] = result['customers_articles_num_sum'].fillna(0.0)
    result['customers_articles_num_count'] = result['customers_articles_num_count'].fillna(0.0)
    
    # features 5-6. сумма и среднее покупок категорий покупателями
    customers_sum_articles_buys = transactions.groupby(['customer_id_short', 'article_id_short'])['price'].agg(['sum', 'mean'])
    result = result.merge(customers_sum_articles_buys, on=['customer_id_short', 'article_id_short'], how='left')
    del customers_sum_articles_buys;
    gc.collect()
    result = result.rename(columns= {'sum': 'customers_articles_sum'})
    result = result.rename(columns= {'mean': 'customers_articles_avg'})
    result['customers_articles_sum'] = result['customers_articles_sum'].fillna(0.0)
    result['customers_articles_avg'] = result['customers_articles_avg'].fillna(0.0)
    
    # features 7-8. общая сумма, потраченная покупателями на покупки и их средний чек
    transactions['order_sum'] = transactions['price'] * transactions['values']
    customers_checks = transactions.groupby(['t_dat', 'customer_id_short'])[['order_sum']].sum()
    customers_sum_spent = customers_checks.groupby(['customer_id_short'])[['order_sum']].sum()
    customers_avg_checks = customers_checks.groupby(['customer_id_short'])[['order_sum']].mean()
    result = result.merge(customers_sum_spent, on='customer_id_short', how='left')
    result = result.rename(columns= {'order_sum': 'customers_spent_sum'})
    result = result.merge(customers_avg_checks, on='customer_id_short', how='left')
    result = result.rename(columns= {'order_sum': 'customers_spent_avg'})
    del customers_checks;
    del customers_sum_spent;
    del customers_avg_checks;
    gc.collect()

    return result
    
# формирование датафрейма для подачи в ранжирующую модель 2го уровня
def df_lvl2_prep(result):
    
    s = result.apply(lambda x: pd.Series(x['own_rec']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'article_id_short'
    
    result = result.drop(['own_rec', 'sim_users'], axis=1).join(s)
    result['actual_article_id_short'] = result['actual_article_id_short'].apply(lambda x: np.array(x))
    
    actual_article_id_short_arr = np.array(result['actual_article_id_short'])
    article_id_short_arr = np.array(result['article_id_short'])
    
    target_arr = [set_target(actual_article_id_short_arr[i], article_id_short_arr[i]) for i in range(len(article_id_short_arr))]
    
    result['target'] = target_arr
    result = result.drop('actual_article_id_short', axis=1)
    
    del s;
    del actual_article_id_short_arr;
    del article_id_short_arr;
    del target_arr;
    gc.collect()
    
    # исходные id покупателей и категорий
    spam1 = pd.read_parquet('archive/transactions_train_for_power_bi.parquet', columns=['customer_id', 'customer_id_short'])
    spam1 = spam1.drop_duplicates(keep='last')
    spam2 = pd.read_parquet('archive/transactions_train_for_power_bi.parquet', columns=['article_id', 'article_id_short'])
    spam2 = spam2.drop_duplicates(keep='last')
    
    # добавляем исходные id покупателей и категорий
    result = result.merge(spam1, on='customer_id_short', how='left')
    result = result.merge(spam2, on='article_id_short', how='left')
    
    del spam1;
    del spam2;
    gc.collect()
       
    return result

# получение топ12 рекомендаций после ранжирования    
@jit(nopython=True, fastmath=True)
def top_12_recs(customers_id, article_id, target, preds):
    customers_id_arr = np.zeros(0)
    recs = np.zeros(0)
    k1 = 0
    k2 = 500
    for i in range(len(set(customers_id))):
        customers_id_spam = customers_id[k1:k2]
        article_id_spam = article_id[k1:k2]
        target_spam = target[k1:k2]
        preds_spam = preds[k1:k2]
        
        mask = preds_spam.argsort()[::-1]
        rec = np.zeros(12)
        
        customers_id_arr = np.append(customers_id_arr, customers_id_spam[0])
        
        for i in range(12):
            rec[i] = article_id_spam[mask[i]]
            
        recs = np.append(recs, rec)
            
        k1 += 500
        k2 += 500
        
    return customers_id_arr, recs
    
    
# построение датафрейма для получения оценок предсказаний модели train-test
def get_preds_result(customers_id, article_id, target, preds, filename):
    
    customers_id = customers_id.astype(np.int32)
    article_id = article_id.astype(np.int32)
        
    # получаем массив из топ12 предсказаний всех покупателей
    customers_id_arr, recs = top_12_recs(customers_id, article_id, target, preds)
    
    # трансформируем его
    recs = recs.reshape(len(set(customers_id)),12)
    customers_id_arr = customers_id_arr.astype(np.int32)
    
    # формируем датафрейм
    spam = pd.DataFrame([(customers_id_arr[i], recs[i]) for i in range(len(set(customers_id)))], columns=['customer_id_short', 'top_12_recs'])
    result_test = pd.read_parquet(f'archive/{filename}.parquet')
    result_test = result_test.merge(spam, on='customer_id_short', how='left')
    
    return result_test
    

# функция преобразования типа данных для сохранения в формате poarquet
def col_convert(val):
    if type(val) is not list:
        return list(val)
    return val    

# получение N рекомендаций с использованием ансамбля моделей ALS OwnRec + user-user + iten-item
# с проверкой вхождения категории в топ предыдущей недели
def get_recommendations(user, sim_users, item_model, model, used_userid_to_id, used_itemid_to_id, custom_sparse_user_item, id_to_used_itemid_nb, top_sim_weeks_articles_nb, N):
    
    if not used_userid_to_id.get(user):
        return top_sim_weeks_articles_nb[:N]
    
    res = model.recommend(userid=used_userid_to_id[user],
        user_items=custom_sparse_user_item[used_userid_to_id[user]],
        N=N,
        filter_already_liked_items=False,
        recalculate_user=True)
    
    
    res = rec_sort(res, id_to_used_itemid_nb)
    
    if len(res) < N:
        sim_users_recs = np.zeros(0)
        
        for i in range(len(sim_users)):
            spam_res = model.recommend(userid=used_userid_to_id[sim_users[i]],
                user_items=custom_sparse_user_item[used_userid_to_id[sim_users[i]]],
                N=N,
                filter_already_liked_items=False,
                recalculate_user=True)
            
            spam_res = rec_sort(spam_res, id_to_used_itemid_nb)
            
            sim_users_recs = np.concatenate((sim_users_recs, spam_res))
            
            
        # items add test
        for el in res:
            spam_res = item_model.similar_items(used_itemid_to_id[el])
            spam_res = rec_sort(spam_res, id_to_used_itemid_nb)

            sim_users_recs = np.concatenate((sim_users_recs, spam_res))
        
         

        sim_users_recs = sim_users_recs.astype(np.int64)
        
           
        res = final_rec(res, sim_users_recs, top_sim_weeks_articles_nb)
        res = res[:N]
    
    res = rec_len_check(res, top_sim_weeks_articles_nb, N)

    return res

# получение похожих покупателей    
def get_sim_users(user, used_userid_to_id, id_to_used_userid, model, N_USERS):
    
    if not used_userid_to_id.get(user):
        return []
    
    similar_users = model.similar_users(used_userid_to_id[user], N_USERS + 1)[0][1:]
    similar_users = np.array([id_to_used_userid[el] for el in similar_users])
    
    return similar_users

# проверка вхождения элемента в список numba type   
@jit(nopython=True, fastmath=True)
def isin(lst, el):
    for i in prange(len(lst)):
        if lst[i]==el:
            return True
    return False

'''    
@jit(nopython=True, fastmath=True)
def customer_articles_ratings(article_ratings, el_ratings, top_sim_weeks_articles_nb):
    for k,v in el_ratings.items():
        if k in article_ratings:
            if isin(top_sim_weeks_articles_nb, k):
                article_ratings[k] += el_ratings[k]*1.3
            else:
                article_ratings[k] += el_ratings[k]
        else:
            if isin(top_sim_weeks_articles_nb, k):
                article_ratings[k] = el_ratings[k]*1.3
            else:
                article_ratings[k] = el_ratings[k]
            
    return article_ratings
'''

# из полученного списа рекомендаций строим словарь по количеству вхождений категорий и берем топ N
@jit(nopython=True, fastmath=True)
def final_rec(res, sim_users_recs, top_sim_weeks_articles_nb):
    article_count = typed.Dict.empty(types.int64,types.int64)
    
    for i in range(len(sim_users_recs)):
        if sim_users_recs[i] in article_count:
            article_count[sim_users_recs[i]] += 1
        else:
            article_count[sim_users_recs[i]] = 1
            
    keys = typed.List(article_count.keys())
    values = typed.List(article_count.values())
    keys = np.asarray(keys)
    values = np.asarray(values)
    
    mask = values.argsort()[::-1]
    rec = np.zeros(len(mask)).astype(np.int64)
    for i in range(len(mask)):
        spam = mask[i]
        rec[i] = keys[spam]
         
    for i in range(len(rec)):
        if rec[i] not in res and rec[i] in top_sim_weeks_articles_nb:
            res = np.append(res, rec[i])

    return res.astype(np.int64)

# сортировка рекомендаций ALS моделей    
@jit(nopython=True, fastmath=True)
def rec_sort(res, id_to_used_itemid_nb):
    mask = res[1].argsort()[::-1]
    rec = np.zeros(len(mask))
    for i in range(len(mask)):
        spam = mask[i]
        rec[i] = id_to_used_itemid_nb[res[0][spam]]
        
    return rec.astype(np.int64)

# проверяем длину рекомендации, если меньше N - дополняем из топа категорий предыдущей недели
@jit(nopython=True, fastmath=True)
def rec_len_check(res, top_sim_weeks_articles_nb, N):
    i = 0
    while len(res) < N:
        if top_sim_weeks_articles_nb[i] not in res:
            res = np.append(res, top_sim_weeks_articles_nb[i])
        i += 1

    return res.astype(np.int64)[:N] 
