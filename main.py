import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse
import pickle



def condition_usersLessThree(df, unique_shopId, res):
    group_shop = df.groupby(['shopid'])
    filter_shopId = []
    freqCount = 0
    for i, shopId in enumerate(unique_shopId):
        # Filtered by buyers less than 3
        if group_shop.get_group(shopId).shape[0] < 3:
            res[shopId] = 0
            filter_shopId.append(shopId)
        # Filter each shop by count number of each userid which less than 2
        else:
            testdf = group_shop.get_group(shopId).sort_values(["event_time"])
            userfreq = testdf['userid'].value_counts()
            if userfreq.max() >= 3:
                freqCount += 1
            else:
                res[shopId] = 0
                filter_shopId.append(shopId)

    unique_shopId = [item for item in unique_shopId if item not in filter_shopId]
    df = df[df['shopid'].isin(unique_shopId)]
    df.to_csv('./condition_user.csv')
    out2 = open('./condition_user_res','wb')
    pickle.dump(res, out2)
    out2.close()

    return unique_shopId, df


def condition_candicatesEqualZero(df, unique_shopId, res):
    subTables = []
    filter_shopId = []

    # Get candicates which equal to 3 and difference between start and end time > 1 hour
    for i, shopId in enumerate(unique_shopId):
        testdf = group_shop.get_group(shopId)
        userfreq = testdf['userid'].value_counts()
        numberCandicate = userfreq[userfreq >=3].index.shape[0]
        candicates = np.array(userfreq[userfreq >=3].index)
        
        if(numberCandicate == 1):
            subtestdf = testdf[testdf['userid'] == candicates[0]].sort_values(["event_time"])
            if(subtestdf.shape[0] == 3):
                startT = parse(subtestdf.iloc[0]['event_time'])
                endT = parse(subtestdf.iloc[-1]['event_time'])
                if((endT-startT) > oneHour):
                    res[shopId] = 0
                    filter_shopId.append(shopId)
        else:
            user_filter = []
            for user in candicates:
                subtestdf = testdf[testdf['userid'] == user].sort_values(["event_time"])
                if(subtestdf.shape[0] == 3):
                    startT = parse(subtestdf.iloc[0]['event_time'])
                    endT = parse(subtestdf.iloc[-1]['event_time'])
                    if((endT-startT) > oneHour):
                        user_filter.append(user)
            candicates = [c for c in candicates if c not in user_filter]
            if len(candicates) == 0:
                res[shopId] = 0
                filter_shopId.append(shopId)
            else:
                subTables.append({'shopid': shopId,
                            "candicates": candicates})


    unique_shopId = [item for item in unique_shopId if item not in filter_shopId]
    df = df[df['shopid'].isin(unique_shopId)]
    df.to_csv('./condition_candicatesEqualZero.csv')
    out2 = open('./condition_candicatesEqualZero_res','wb')
    pickle.dump(res, out2)
    out2.close()

    out1 = open('./subTables','wb')
    pickle.dump(subTables, out1)
    out1.close()

    return unique_shopId, df



if __name__ == "__main__":
    ## Step 1 ##
    df = pd.read_csv('order_brush_order.csv')
    print(df.head())
    res = dict()

    # res shape should be 18770 * 2
    remain_answer_shop = 18770

    oneHour = datetime.timedelta(hours=1)

    print(df.shape)
    print("number of shop id: ", df['shopid'].nunique())
    print("number of user id: ", df['userid'].nunique())
    remain_answer_shop = df['shopid'].nunique()
    unique_shopId = df['shopid'].unique()
    unique_userId = df['userid'].unique()
    print("unique shop id: ", unique_shopId, len(unique_shopId))
    print("unique user id: ", unique_userId, len(unique_userId))

    # unique_shopId, df = condition_user(df, unique_shopId, res)
    print("Step0 Got number of answers: ", len(res), "remain answers: ", len(unique_shopId), len(res)+len(unique_shopId) == remain_answer_shop)


    ## Step 2 ##
    df = pd.read_csv('condition_user.csv')
    group_shop = df.groupby(['shopid'])
    res = pickle.load(open('./condition_user_res','rb'))
    unique_shopId = df['shopid'].unique()
    print("Step1 Got number of answers: ", len(res), "remain answers: ", len(unique_shopId), len(res)+len(unique_shopId) == remain_answer_shop)

    # unique_shopId, df = condition_candicatesEqualZero(df, unique_shopId, res)

    ## Step 3 ##
    df = pd.read_csv('condition_candicatesEqualZero.csv')
    group_shop = df.groupby(['shopid'])
    res = pickle.load(open('./condition_candicatesEqualZero_res','rb'))
    subTables = pickle.load(open('./subTables','rb'))
    unique_shopId = df['shopid'].unique()

    print("Step2 Got number of answers: ", len(res), "remain answers: ", len(unique_shopId), len(res)+len(unique_shopId) == remain_answer_shop)
