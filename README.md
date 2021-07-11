# Optiver-Realized-Volatility-Prediction

# Optiver Realized Volatility Prediction
### Kaggle Info
* [Optiver Realized Volatility Prediction | Kaggle](https://www.kaggle.com/c/optiver-realized-volatility-prediction)
* [Optiver Realized: EDA for starter(English version) | Kaggle](https://www.kaggle.com/chumajin/optiver-realized-eda-for-starter-english-version?select=sample_submission.csv)
* [Introduction to financial concepts and data | Kaggle](https://www.kaggle.com/jiashenliu/introduction-to-financial-concepts-and-data?scriptVersionId=67183666#Competition-data)

### Reference
* [均方誤差 - 維基百科，自由的百科全書](https://zh.wikipedia.org/wiki/%E5%9D%87%E6%96%B9%E8%AF%AF%E5%B7%AE)
* [lucko515/tesla-stocks-prediction: The implementation of LSTM in TensorFlow used for the stock prediction](https://github.com/lucko515/tesla-stocks-prediction)
* [What is EDA (Exploratory Data Analysis)? 淺談何謂探索式資料分析 - iT 邦幫忙::一起幫忙解決難題，拯救 IT 人的一天](https://ithelp.ithome.com.tw/articles/10213384)


#### A glimpse of our trading floor


![work_at_optiver](https://www.optiver.com/wp-content/uploads/2020/11/WorkingAtOptiver_Hero.jpg)

# Optiver Realized Volatility Prediction


# Introduction
In order to make Kagglers better prepared for this competition, Optiver's data scientists have created a tutorial notebook looping through some financial concepts covered in this particular trading challenge. Also, the data structure and the example code submission will also be presented in this notebook. 


# Order book
The term order book refers to an electronic list of buy and sell orders for a specific security or financial instrument organized by price level. An order book lists the number of shares being bid on or offered at each price point.

Below is a snapshot of an order book of a stock (let's call it stock A), as you can see, all intended buy orders are on the left side of the book displayed as "bid" while all intended sell orders are on the right side of the book displayed as "offer/ask"
# 


![order_book_1](https://www.optiver.com/wp-content/uploads/2021/05/OrderBook3.png)

An actively traded financial instrument always has a dense order book (A liquid book). As the order book data is a continous representation of market demand/supply it is always considered as the number one data source for market research. 


# Trade
An order book is a representation of trading intention on the market, however the market needs a buyer and seller at the **same** price to make the trade happen. Therefore, sometimes when someone wants to do a trade in a stock, they check the order book and find someone with counter-interest to trade with. 

For example, imagine you want to buy 20 shares of a stock A when you have the order book in the previous paragraph. Then you need to find some people who are willing to trade against you by selling 20 shares or more in total. You check the **offer** side of the book starting from the lowest price: there are 221 shares of selling interest on the level of 148. You can **lift** 20 shares for a price of 148 and **guarantee** your execution. This will be the resulting order book of stock A after your trade:


![order_book2](https://www.optiver.com/wp-content/uploads/2021/05/OrderBook4.png)

In this case, the seller(s) sold 20 shares and buyer bought 20 shares, the exchange will match the order between seller(s) and buyer and one trade message will be broadcast to public:

- 20 shares of stock A traded on the market at price of 148.


Similar to order book data, trade data is also extremely crucial to Optiver's data scientists, as it reflects how active the market is. Actually, some commonly seen technical signals of the financial market are derived from trade data directly, such as high-low or total traded volume.


# Market making and market efficiency
Imagine, on another day, stock A's order book becomes below shape, and you, again, want to buy 20 shares from all the intentional sellers. As you can see the book is not as dense as the previous one, and one can say, compared with the previous one, this book is **less liquid**.


![order_book_3](https://www.optiver.com/wp-content/uploads/2021/05/OrderBook5.png)


You could insert an order to buy at 148. However, there is nobody currently willing to sell to you at 148, so your order will be sitting in the book, waiting for someone to trade against it. If you get unlucky, the price goes up, and others start bidding at 149, and you never get to buy at all. Alternatively, you could insert an order to buy at 155. The exchange would match this order against the outstanding sell order of one share at 149, so you buy 1 lot at 149. Similarly, you'd buy 12 shares at a price of 150, and 7 shares at 151. Compared to trying to buy at 148, there is no risk of not getting the trade that you wanted, but you do end up buying at a higher price.


You can see that in such an inefficient market it is difficult to trade, as trading will be more expensive, and if you want quality execution of your orders, you need to deal with higher market risk. That is why investors love liquidity, and market makers like Optiver are there to provide it, no matter how extreme market conditions are.


A market maker is a firm or individual who actively quotes two-sided markets in a security, providing bids and offers (known as asks) along with the market size of each. As a market maker will show both bid and offer orders, an order book with the presence of market maker will be more liquid, therefore a more efficient market will be provided to end investors to trade freely without concern on executions.


# Order book statistics
There are a lot of statistics Optiver data scientist can derive from raw order book data to reflect market liquidity and stock valuation. These stats are proven to be fundamental inputs of any market prediction algorithms. Below we would like to list some common stats to inspire Kagglers mining more valuable signals from the order book data.

Let's come back to the original order book of stock A


![order_book_1](https://www.optiver.com/wp-content/uploads/2021/05/OrderBook3.png)


**bid/ask spread**
 
As different stocks trade on different level on the market we take the ratio of best offer price and best bid price to calculate the bid-ask spread. 

The formula of bid/ask spread can be written in below form:
$$BidAskSpread = BestOffer/BestBid -1$$


**Weighted averaged price**

The order book is also one of the primary source for stock valuation. A fair book-based valuation must take two factors into account: the level and the size of orders. In this competition we used weighted averaged price, or WAP, to calculate the instantaneous stock valuation and calculate realized volatility as our target. 

The formula of WAP can be written as below, which takes the top level price and volume information into account:

$$ WAP = \frac{BidPrice_{1}*AskSize_{1} + AskPrice_{1}*BidSize_{1}}{BidSize_{1} + AskSize_{1}} $$

As you can see, if two books have both bid and ask offers on the same price level respectively, the one with more offers in place will generate a lower stock valuation, as there are more intended seller in the book, and more seller implies a fact of more supply on the market resulting in a lower stock valuation.

Note that in most of cases, during the continuous trading hours, an order book should not have the scenario when bid order is higher than the offer, or ask, order. In another word, most likely, the bid and ask should never be **in cross.**

In this competition the target is constructed from the WAP. The WAP of the order book snapshot is 147.5317797.


# Log returns

**How can we compare the price of a stock between yesterday and today?**
 
The easiest method would be to just take the difference. This is definitely the most intuitive way, however **price differences** are not always comparable across stocks. For example, let's assume that we have invested $\$$1000 dollars in both stock A and stock B and that stock A moves from $\$$100 to $\$$102 and stock B moves from $\$$10 to $\$$11. We had a total of 10 shares of A ($\$1000 \ / \ \$100 = 10$) which led to a profit of $10 \cdot (\$102 - \$100) = \$20$ and a total of 100 shares of B that yielded \$100. So the price increase was larger for stock **A**, although the move was proportionally much larger for stock B.

We can solve the above problem by dividing the move by the starting price of the stock, effectively computing the percentage change in price, also known as the **stock return**. In our example, the return for stock A was $\frac{\$102 - \$100 }{\$100} = 2\%$, while for stock B it was $\frac{\$11 - \$10 }{\$10} = 10\%$. The stock return coincides with the percentage change in our invested capital.

Returns are widely used in finance, however **log returns** are preferred whenever some mathematical modelling is required. Calling $S_t$ the price of the stock $S$ at time $t$, we can define the log return between $t_1$ and $t_2$ as:
$$
r_{t_1, t_2} = \log \left( \frac{S_{t_2}}{S_{t_1}} \right)
$$
Usually, we look at log returns over fixed time intervals, so with 10-minute log return we mean $r_t = r_{t - 10 min, t}$.

Log returns present several advantages, for example:
- they are additive across time $r_{t_1, t_2} + r_{t_2, t_3} = r_{t_1, t_3}$
- regular returns cannot go below -100%, while log returns are not bounded


# Realized volatility
When we trade options, a valuable input to our models is the standard deviation of the stock log returns. The standard deviation will be different for log returns computed over longer or shorter intervals, for this reason it is usually normalized to a 1-year period and the annualized standard deviation is called **volatility**. 

In this competition, you will be given 10 minutes of book data and we ask you to predict what the volatility will be in the following 10 minutes. Volatility will be measured as follows:

We will compute the log returns over all consecutive book updates and we define the **realized volatility, $\sigma$,** as the squared root of the sum of squared log returns.
$$
\sigma = \sqrt{\sum_{t}r_{t-1, t}^2}
 $$
Where we use **WAP** as price of the stock to compute log returns.

We want to keep definitions as simple and clear as possible, so that Kagglers without financial knowledge will not be penalized. So we are not annualizing the volatility and we are assuming that log returns have 0 mean.


# Competition data
In this competition, Kagglers are challenged to generate a series of short-term signals from the book and trade data of a fixed 10-minute window to predict the realized volatility of the next 10-minute window. The target, which is given in train/test.csv, can be linked with the raw order book/trade data by the same **time_id** and **stock_id**. There is no overlap between the feature and target window.


Note that the competition data will come with partitioned parquet file. You can find a tutorial of parquet file handling in this [notebook](https://www.kaggle.com/sohier/working-with-parquet)


<iframe src="https://www.kaggle.com/embed/jiashenliu/introduction-to-financial-concepts-and-data?cellIds=23&kernelSessionId=67183666" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="Introduction to financial concepts and data"></iframe>


Taking the first row of data, it implies that the realized vol of the **target bucket** for time_id 5, stock_id 0 is 0.004136. How does the book and trade data in **feature bucket** look like for us to build signals?

<iframe src="https://www.kaggle.com/embed/jiashenliu/introduction-to-financial-concepts-and-data?cellId=27&cellIds=25&kernelSessionId=67183666" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="Introduction to financial concepts and data"></iframe>
<br><br>

**book data snapshot**

<iframe src="https://www.kaggle.com/embed/jiashenliu/introduction-to-financial-concepts-and-data?cellId=27&cellIds=27&kernelSessionId=67183666" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="Introduction to financial concepts and data"></iframe>
<br><br>


**trade date snapshot**

<iframe src="https://www.kaggle.com/embed/jiashenliu/introduction-to-financial-concepts-and-data?cellId=27&cellIds=29&kernelSessionId=67183666" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="Introduction to financial concepts and data"></iframe>

<br><br>

**Realized volatility calculation in python**


In this competition, our target is to predict short-term realized volatility. Although the order book and trade data for the target cannot be shared, we can still present the realized volatility calculation using the feature data we provided. 
 
As realized volatility is a statistical measure of price changes on a given stock, to calculate the price change we first need to have a stock valuation at the fixed interval (1 second). We will use weighted averaged price, or WAP, of the order book data we provided.

<iframe src="https://www.kaggle.com/embed/jiashenliu/introduction-to-financial-concepts-and-data?cellId=27&cellIds=32&kernelSessionId=67183666" height="150" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="Introduction to financial concepts and data"></iframe>

<br><br>

**The WAP of the stock is plotted below**

# %% [code] {"execution":{"iopub.status.busy":"2021-06-22T17:13:08.573937Z","iopub.execute_input":"2021-06-22T17:13:08.574295Z","iopub.status.idle":"2021-06-22T17:13:10.128115Z","shell.execute_reply.started":"2021-06-22T17:13:08.574265Z","shell.execute_reply":"2021-06-22T17:13:10.126867Z"}}
fig = px.line(book_example, x="seconds_in_bucket", y="wap", title='WAP of stock_id_0, time_id_5')
fig.show()

# %% [markdown]
# To compute the log return, we can simply take **the logarithm of the ratio** between two consecutive **WAP**. The first row will have an empty return as the previous book update is unknown, therefore the empty return data point will be dropped.

# %% [code] {"execution":{"iopub.status.busy":"2021-06-25T12:19:46.464955Z","iopub.execute_input":"2021-06-25T12:19:46.465341Z","iopub.status.idle":"2021-06-25T12:19:46.469499Z","shell.execute_reply.started":"2021-06-25T12:19:46.465304Z","shell.execute_reply":"2021-06-25T12:19:46.468504Z"}}
def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

# %% [code] {"execution":{"iopub.status.busy":"2021-06-25T12:19:47.146461Z","iopub.execute_input":"2021-06-25T12:19:47.147057Z","iopub.status.idle":"2021-06-25T12:19:47.159512Z","shell.execute_reply.started":"2021-06-25T12:19:47.146987Z","shell.execute_reply":"2021-06-25T12:19:47.158589Z"}}
book_example.loc[:,'log_return'] = log_return(book_example['wap'])
book_example = book_example[~book_example['log_return'].isnull()]

# %% [markdown] {"execution":{"iopub.status.busy":"2021-06-09T15:01:53.679074Z","iopub.execute_input":"2021-06-09T15:01:53.679605Z","iopub.status.idle":"2021-06-09T15:01:53.686279Z","shell.execute_reply.started":"2021-06-09T15:01:53.67957Z","shell.execute_reply":"2021-06-09T15:01:53.684738Z"}}
# **Let's plot the tick-to-tick return of this instrument over this time bucket**

# %% [code] {"execution":{"iopub.status.busy":"2021-06-22T17:49:48.840362Z","iopub.execute_input":"2021-06-22T17:49:48.840757Z","iopub.status.idle":"2021-06-22T17:49:48.917363Z","shell.execute_reply.started":"2021-06-22T17:49:48.840723Z","shell.execute_reply":"2021-06-22T17:49:48.9162Z"}}
fig = px.line(book_example, x="seconds_in_bucket", y="log_return", title='Log return of stock_id_0, time_id_5')
fig.show()

# %% [markdown]
# The realized vol of stock 0 in this feature bucket, will be:

# %% [code] {"execution":{"iopub.status.busy":"2021-06-25T12:19:49.310879Z","iopub.execute_input":"2021-06-25T12:19:49.311502Z","iopub.status.idle":"2021-06-25T12:19:49.320244Z","shell.execute_reply.started":"2021-06-25T12:19:49.311443Z","shell.execute_reply":"2021-06-25T12:19:49.319109Z"}}
def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))
realized_vol = realized_volatility(book_example['log_return'])
print(f'Realized volatility for stock_id 0 on time_id 5 is {realized_vol}')

# %% [markdown]
# # Naive prediction: using past realized volatility as target

# %% [markdown]
# A commonly known fact about volatility is that it tends to be autocorrelated. We can use this property to implement a naive model that just "predicts" realized volatility by using whatever the realized volatility was in the initial 10 minutes.
# 
# Let's calculate the past realized volatility across the training set to see how predictive a single naive signal can be.

# %% [code] {"execution":{"iopub.status.busy":"2021-06-22T17:58:30.665521Z","iopub.execute_input":"2021-06-22T17:58:30.665895Z","iopub.status.idle":"2021-06-22T17:58:30.673552Z","shell.execute_reply.started":"2021-06-22T17:58:30.665865Z","shell.execute_reply":"2021-06-22T17:58:30.672424Z"}}
import os
from sklearn.metrics import r2_score
import glob
list_order_book_file_train = glob.glob('/kaggle/input/optiver-realized-volatility-prediction/book_train.parquet/*')

# %% [markdown]
# As the data is partitioned by stock_id in this competition to allow Kagglers better manage the memory, we try to calculcate realized volatility stock by stock and combine them into one submission file. Note that the stock id as the partition column is not present if we load the single file so we will remedy that manually. We will reuse the log return and realized volatility functions defined in the previous session.

# %% [code] {"execution":{"iopub.status.busy":"2021-06-22T18:31:38.919415Z","iopub.execute_input":"2021-06-22T18:31:38.919961Z","iopub.status.idle":"2021-06-22T18:31:38.927476Z","shell.execute_reply.started":"2021-06-22T18:31:38.919927Z","shell.execute_reply":"2021-06-22T18:31:38.926646Z"}}
def realized_volatility_per_time_id(file_path, prediction_column_name):
    df_book_data = pd.read_parquet(file_path)
    df_book_data['wap'] =(df_book_data['bid_price1'] * df_book_data['ask_size1']+df_book_data['ask_price1'] * df_book_data['bid_size1'])  / (
                                      df_book_data['bid_size1']+ df_book_data[
                                  'ask_size1'])
    df_book_data['log_return'] = df_book_data.groupby(['time_id'])['wap'].apply(log_return)
    df_book_data = df_book_data[~df_book_data['log_return'].isnull()]
    df_realized_vol_per_stock =  pd.DataFrame(df_book_data.groupby(['time_id'])['log_return'].agg(realized_volatility)).reset_index()
    df_realized_vol_per_stock = df_realized_vol_per_stock.rename(columns = {'log_return':prediction_column_name})
    stock_id = file_path.split('=')[1]
    df_realized_vol_per_stock['row_id'] = df_realized_vol_per_stock['time_id'].apply(lambda x:f'{stock_id}-{x}')
    return df_realized_vol_per_stock[['row_id',prediction_column_name]]

# %% [markdown]
# Looping through each individual stocks, we can get the past realized volatility as prediction for each individual stocks.

# %% [code] {"execution":{"iopub.status.busy":"2021-06-22T18:01:24.769773Z","iopub.execute_input":"2021-06-22T18:01:24.770136Z","iopub.status.idle":"2021-06-22T18:12:57.346484Z","shell.execute_reply.started":"2021-06-22T18:01:24.770106Z","shell.execute_reply":"2021-06-22T18:12:57.345476Z"}}
def past_realized_volatility_per_stock(list_file,prediction_column_name):
    df_past_realized = pd.DataFrame()
    for file in list_file:
        df_past_realized = pd.concat([df_past_realized,
                                     realized_volatility_per_time_id(file,prediction_column_name)])
    return df_past_realized
df_past_realized_train = past_realized_volatility_per_stock(list_file=list_order_book_file_train,
                                                           prediction_column_name='pred')

# %% [markdown]
# Let's join the output dataframe with train.csv to see the performance of the naive prediction on training set.

# %% [code] {"execution":{"iopub.status.busy":"2021-06-22T18:13:50.128553Z","iopub.execute_input":"2021-06-22T18:13:50.129061Z","iopub.status.idle":"2021-06-22T18:13:51.891263Z","shell.execute_reply.started":"2021-06-22T18:13:50.129028Z","shell.execute_reply":"2021-06-22T18:13:51.890252Z"}}
train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
train = train[['row_id','target']]
df_joined = train.merge(df_past_realized_train[['row_id','pred']], on = ['row_id'], how = 'left')

# %% [markdown]
# We will evaluate the naive prediction result by two metrics: RMSPE and R squared. 

# %% [code] {"execution":{"iopub.status.busy":"2021-06-22T18:13:54.852792Z","iopub.execute_input":"2021-06-22T18:13:54.853278Z","iopub.status.idle":"2021-06-22T18:13:54.872291Z","shell.execute_reply.started":"2021-06-22T18:13:54.853239Z","shell.execute_reply":"2021-06-22T18:13:54.871482Z"}}
from sklearn.metrics import r2_score
def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
R2 = round(r2_score(y_true = df_joined['target'], y_pred = df_joined['pred']),3)
RMSPE = round(rmspe(y_true = df_joined['target'], y_pred = df_joined['pred']),3)
print(f'Performance of the naive prediction: R2 score: {R2}, RMSPE: {RMSPE}')

# %% [markdown]
# The performance of the naive model is not amazing but as a benchmark it is a reasonable start.

# %% [markdown]
# # Submission

# %% [markdown] {"execution":{"iopub.status.busy":"2021-06-09T15:25:51.891717Z","iopub.execute_input":"2021-06-09T15:25:51.89209Z","iopub.status.idle":"2021-06-09T15:25:51.898582Z","shell.execute_reply.started":"2021-06-09T15:25:51.892059Z","shell.execute_reply":"2021-06-09T15:25:51.89729Z"}}
# As a last step, we will make a submission via the tutorial notebook -- through a file written to output folder.  The naive submission scored a RMSPE 0.327 on public LB, the room of improvement is big for sure!

# %% [code] {"execution":{"iopub.status.busy":"2021-06-22T18:31:41.736646Z","iopub.execute_input":"2021-06-22T18:31:41.737029Z","iopub.status.idle":"2021-06-22T18:31:41.768867Z","shell.execute_reply.started":"2021-06-22T18:31:41.736997Z","shell.execute_reply":"2021-06-22T18:31:41.767699Z"}}
list_order_book_file_test = glob.glob('/kaggle/input/optiver-realized-volatility-prediction/book_test.parquet/*')
df_naive_pred_test = past_realized_volatility_per_stock(list_file=list_order_book_file_test,
                                                           prediction_column_name='target')
df_naive_pred_test.to_csv('submission.csv',index = False)


Note that in this competition, there will be only few rows of test data that can be downloaded. The actual evaluation program will run in background after you commit the notebook and manually submit the output. Please check to [code requirement](https://www.kaggle.com/c/optiver-realized-volatility-prediction/overview/code-requirements) for more explanation.


The private leaderboard will be built against the real market data collected after the training period, therefore the public and private leaderboard data will have zero overlap. It will be exciting to get your model tested against the live market! As this competition will provide a very rich dataset representing market microstructure, there is unlimited amount of signals one can come up with. It is all on you, good luck! We at Optiver are really looking forward to learn from the talented Kaggle community!

