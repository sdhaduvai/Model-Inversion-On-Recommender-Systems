from itertools import combinations 
from numpy import random
import turicreate as tc
import pandas as pd
from sklearn.model_selection import KFold
from turicreate import SFrame
from sklearn.model_selection import train_test_split
import numpy as np
import csv

# Functions to calculate NDCG
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg
    
# Loading the dataset
df = pd.read_csv("Movies_and_TV.csv", sep=',', header=None)
df.columns = ['item', 'user', 'rating']
columns_titles = ["user","item", "rating"]                   
df=df.reindex(columns=columns_titles)

# Preprocessing the dataset to remove users with low item count
sub_df = df[df.groupby('user').user.transform('count')>20].copy() 

#Getting a list of all unique users
users = sub_df['user'].unique()

# Splitting the users into train and test
users_train, users_test = train_test_split(users, test_size=0.02, random_state=42)

# Creating seperate dataframes for training users and testing users
train_df = sub_df[sub_df.user.isin(users_train)]
test_df = sub_df[sub_df.user.isin(users_test)]

test_df.columns = ['user', 'item', 'rating']
sf1 = SFrame(data=train_df)
sf2 = SFrame(data=test_df)


# Retaining some portion of the test users' data into training data for the Original RecSys
train, test = tc.recommender.util.random_split_by_user(sf2, user_id='user', item_id='item', max_num_users=26788)

# Some portion of test user's data is added to the Original RecSys so that we can calculate their RMSE
final_sf = sf1 + train

test_users_train = train.to_dataframe()
test_users_eval = test.to_dataframe()

sample_of_users = test['user']
sample_of_users = list(set(sample_of_users))

# Training the original recsys
amazon = tc.recommender.item_similarity_recommender.create(final_sf, user_id='user', item_id='item', target='rating', similarity_type = 'cosine')

# Getting the RMSE for the target users
evals = amazon.evaluate_rmse(test, target='rating')

e_users = evals['rmse_by_user']['user']
e_rmse = evals['rmse_by_user']['rmse']
eval_dict = {}

my_users = test_users_eval.user.unique()
precision_dict, ndcg_dict = {}, {}
for user in my_users:
    # Getting the user's purchase history for calculating their RMSE, Precision and NDCG
    temp_test = test_users_eval[test_users_eval['user'] == user]['item'].tolist()
    temp_rating = test_users_eval[test_users_eval['user'] == user]['rating'].tolist()
    recs = amazon.recommend([user], len(temp_test))['item']
    result = [item for item in recs if item in temp_test]
    try:
        precision_dict[user] = len(result)/len(temp_test)
    except:
        # The random_split_by_user function did not leave any items in test set for this user, so we remove them
        sample_of_users.remove(user)
    relevance = []
    for item in recs:
        if item in temp_test:
            relevance.append(temp_rating[temp_test.index(item)])
        else:
            relevance.append(0)

    user_ndcg = ndcg_at_k(relevance, len(relevance))
    ndcg_dict[user] = user_ndcg

csvfile = open('icf_icf.csv', 'w')
resultwriter = csv.writer(csvfile)
resultwriter.writerow(['Target_Size', 'Training_data', 'User', 'Original Items', 'Num of Recs(K_O)', 'Rating', 'Num_of_preds(K_A)', 'Hits', 'Precision', 'NDCG', 'Recall', 'RMSE', 'Eval_RMSE', 'Original_Precision', 'Org_NDCG', 'Fold'])
csvfile.close()

for i in range(len(e_users)):
    eval_dict[e_users[i]] = e_rmse[i]

# We add target's recommendations with these 2 ratings
ratings = [5]

# We give the attacker different proportions of training data
fracs = [1]

# We target different number of users at once
targets = [1,2,3,4,5,6,7,8,9,10]
for target in targets:
    # Collecting 10 sets of 't' number of target users
    temp = sample_of_users
    count = 0
    selected_users = []
    while count < 5:
        short = []
        selected_indices = random.choice(range(len(temp)), target)
        for i in selected_indices:
            short.append(temp[i])
        temp = [i for j, i in enumerate(temp) if j not in selected_indices]

        selected_users.append(short)
        count += 1
    
    for rating in ratings:
        
        for set_of_users in selected_users:
            tf = train_df
            for user in set_of_users:
                # Making sure that attacker has no information about the targets
                tf = tf[tf.user != user]
            for user in set_of_users:
                nf = tf
                original_items = df.loc[df['user'] == user]['item'].tolist()
                original_ratings = df.loc[df['user'] == user]['rating'].tolist()
                current_user = df[df['user'] == user]
                current_user_SFrame = SFrame(current_user)
                n = len(original_items)

                recs = [5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 200, 300]
                recs = list(reversed(recs))

                for k_o in recs:
                    # 'k_o' refers to number of recommendations observed for the target
                    # a is a dataframe 
                    attack = amazon.recommend([user], k_o)
                    
                    for f in fracs:
                        if f != 1:
                            fold = 0
                        else:
                            fold = 4

                        # Running each proportion of training data 5 times to remove luck bias  
                        while fold < 5:
                            fold += 1
                            # Giving the attacker only a small proportion of training data
                            pf = nf.sample(frac=f)

                            # Adding recommendations to attacker's knowledgebase
                            for item in attack['item']:
                                pf = pf.append({'item' : item , 'user' : user , 'rating': rating} , ignore_index=True)
                            nframe = tc.SFrame(pf)
                            model2 = tc.recommender.item_similarity_recommender.create(nframe, user_id='user', item_id='item', target='rating', similarity_type = 'cosine')

                            # K_A refers to the number of attack recommendations
                            for i, k_a in enumerate(recs):
                                recomm_items = model2.recommend([user], k_a)['item']

                                result = [item for item in recomm_items if item in original_items]
                                relevance = []
                                for item in recomm_items:
                                    if item in original_items:
                                        relevance.append(original_ratings[original_items.index(item)])
                                    else:
                                        relevance.append(0)
                                        
                                user_ndcg = ndcg_at_k(relevance, k_a)
                                precision = len(result) / k_a
                                recall = len(results)/n

                                if i == 0:
                                    getting_rmse = model2.evaluate(current_user_SFrame)
                                    attack_rmse = getting_rmse['rmse_by_user']['user' == user]['rmse']
                                    
                                rmse = eval_dict[user]
                                if len(result) == 0:
                                    with open("icf_icf.csv", "a+") as csvfile:
                                        resultwriter = csv.writer(csvfile)
                                        resultwriter.writerow([target, f, user, len(original_items), k_o, rating, k_a, 0, 0, 0, 0, rmse, attack_rmse, precision_dict[user], ndcg_dict[user], fold])
                                else:
                                    precision = len(result) / n
                                    print("Result: ", [target, f, user, len(original_items), k_o, rating, k_a, len(result), precision, user_ndcg, recall, rmse, attack_rmse, precision_dict[user], ndcg_dict[user], fold])
                                    with open("icf_icf.csv", "a+") as csvfile:
                                        resultwriter = csv.writer(csvfile)
                                        resultwriter.writerow([target, f, user, len(original_items), k_o, rating, k_a, len(result), precision, user_ndcg, recall, rmse, attack_rmse, precision_dict[user], ndcg_dict[user], fold])