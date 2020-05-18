import turicreate as tc
import csv
import pandas as pd
import numpy as np

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

df = pd.read_csv("Movies_and_TV.csv", sep=',', header=None)
df.columns = ['item', 'user', 'rating']

columns_titles = ["user","item", "rating"]
                    
df=df.reindex(columns=columns_titles)

movies = tc.SFrame(data=df)

model = tc.recommender.item_similarity_recommender.create(movies, user_id='user', item_id='item', target='rating', similarity_type='pearson')

csvfile = open('rf_results.csv', 'w')
resultwriter = csv.writer(csvfile)
resultwriter.writerow(['User', 'Original Items', 'Retained History', 'Num of Recs', 'Rating', 'Hits', '% Recovered','Precision'])
csvfile.close()

try:
    users = ['A6ADO7B6FUVN', 'AYWSFRCIMOAYE', 'A2R6RA8FRBS608', 'A25ZVI6RH1KA5L', 'A13E0ARAXI6KJW', 'A2ZB8B7VQONZA6', 'A2FPDWTD9AENVK', 'A1TA5QYECZP1L1', 'A10H47FMW8NHII', 'ANAYSRE3LX8GZ']
    ratings  = [3.5, 5]
    for user in users:
        original_items = df.loc[df['user'] == user]['item'].tolist()
        original_rating = df.loc[df['user'] == user]['rating'].tolist()
        n = len(original_items)
        
        percentages = [75]
        #, 25, 50, 75, 100
        
        for percentage in percentages:
            c = (percentage/100) * n
            partial_items = original_items[:int(c)]
            partial_rating = original_rating[:int(c)]
            
            remaining_items = [i for i in original_items if i not in partial_items]
            remaining_ratings = []
            
            for item in remaining_items:
                remaining_ratings.append(original_rating[original_items.index(item)])
                
            y = len(partial_items)

            n = n-y
            recs = range(500, 0, -50)
            for t in recs:
                a = model.recommend([user], t)

                for rating in ratings:
                    nf = df[df.user != user]
                    for item in a['item']:
                        nf = nf.append({'item' : item , 'user' : user , 'rating': rating} , ignore_index=True)
                        
                    for i, item in enumerate(partial_items):
                        nf = nf.append({'item' : item , 'user' : user , 'rating': partial_rating[i]} , ignore_index=True)

                    nframe = tc.SFrame(nf)
                    model2 = tc.ranking_factorization_recommender.create(nframe, user_id='user', item_id='item', target='rating')
                    for k_a in range(1, 501, 50):
                        recomm_items = model2.recommend([user], k_a)['item']


                        result = [item for item in recomm_items if item in remaining_items]
                        relevance = []
                        for item in recomm_items:
                            if item in remaining_items:
                                relevance.append(remaining_ratings[remaining_items.index(item)])
                            else:
                                relevance.append(0)
    #                     ndcg = ndcg_at_k(relevance, len(relevance))
                        if len(result) == 0:
                            with open("rf_results.csv", "a+") as csvfile:
                                resultwriter = csv.writer(csvfile)
                                resultwriter.writerow([user, len(original_items), t, rating, len(recomm_items), 0, 0])
                        else:
                            precision = len(result) / n
    #                         recall = len(result) / len(remaining_items)
    #                         percentage = (len(result) / len(remaining_items)) * 100
                            print("Result: ", [user, len(original_items), t, rating, len(recomm_items), len(result), precision])
                            with open("rf_results.csv", "a+") as csvfile:
                                resultwriter = csv.writer(csvfile)
                                resultwriter.writerow([user, len(original_items), t, rating, len(recomm_items), len(result), precision])
except Exception as e:
    print(e)

print("It is done")