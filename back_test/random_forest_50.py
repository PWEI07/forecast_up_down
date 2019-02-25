import talib
from tree_clf.random_forest import ForestBuilder
from datetime import timedelta
from pandas import Series, DataFrame
import numpy as np


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    context.pool = index_components('000016.XSHG', date='2012-01-01')
    context.model_generators = {}  # 用于存储投资池中各个股票的模型生成器
    for i in context.pool:
        context.model_generators[i] = ForestBuilder(i)

    context.eligible_models = {}  # 用于存储通过标准的模型
    context.eligible_models_weights = Series()  # 用于存储达标模型的资金权重
    context.eligible_models_scalers = {}  # 用于存储达标模型的scaler

    scheduler.run_monthly(get_models)  # 每个月训练一次模型，选出符合标准的放入context.eligible_models
    context.precision_threshold = 0.65  # 设置精确度阈值，通过此阈值的模型才可以入选
    context.F_threshold = 0.7  # 设置F1 score阈值，通过此阈值的模型才可以入选
    context.transaction = DataFrame()  # 用于记录买入的品种，日期，价格
    scheduler.run_daily(sell_old, time_rule=market_open(minute=0))  # 每天09:31卖掉持有日大于等于5天的股票
    scheduler.run_daily(order_buy_list, time_rule=market_open(minute=0))  # 每天09:31买入context.buy_list中的股票

    for i in np.arange(0, 240, 5):
        scheduler.run_daily(check_and_sell, time_rule=market_open(minute=i))  # 为加快回测运行速度，每隔5分钟（而不是每分钟）检查是否有市价超过成本价1%的股票并卖出
    context.buy_list = []  # 预测要涨 需要买的股票


def get_models(context):
    """
    获得该期达标模型，以及其scaler，并根据其模型得分计算对其分配的资金权重
    :param context:
    :return:
    """
    for i in context.pool:
        end_date = context.now.strftime('%Y-%m-%d')
        start_date = (context.now - timedelta(days=365*4)).strftime('%Y-%m-%d')
        context.eligible_models = {}  # 先将上一期达标模型清空
        context.eligible_models_weights = Series()  # # 先将上一期达标模型的资金权重清空
        model_i = context.model_generators[i].get_eligible_model(start_date, end_date,
                                                                 precision_threshold=context.precision_threshold,
                                                                 F_threshold=context.F_threshold)
        if model_i is not None:
            context.eligible_models[i] = model_i['best_rf']
            context.eligible_models_scalers[i] = model_i['scaler']
            context.eligible_models_weights[i] = model_i['test_precision'] + model_i['test_f1']
    context.eligible_models_weights /= context.eligible_models_weights.sum()


def sell_old(context, bar_dict):
    """
    卖掉持有日超过5天的股票
    :param context:
    :param bar_dict:
    :return:
    """
    temp_sell_df = context.transaction.date[(context.now - context.transaction.date) >= timedelta(days=5)]
    for i in temp_sell_df.index:
        order_shares(i, amount=-1 * context.portfolio.positions[i].quantity)
        context.transaction.drop(index=i, inplace=True)


def record_transaction(context, bar_dict, order_book_id):
    """
    记录做多的股票代码 开仓时间 开仓价格
    :param context:
    :return:
    """
    context.transaction = context.transaction.append(Series(index=['date', 'cost'],
                                                            data=[context.now, bar_dict[order_book_id].last],
                                                            name=order_book_id))


def make_prediction(context):
    """
    对每个达标模型，获取对应股票最近60个交易日的输入数据，利用模型进行预测
    :param context:
    :return:
    """
    context.buy_series = Series()
    for stock in context.eligible_models:
        if context.portfolio.positions[i].quantity != 0:
            continue  # 如果该模型已经开仓 则跳过
        else:
            pass
        model_generator = context.model_generators[stock]  # 调取模型生成器
        start_date = get_previous_trading_date(context.now, 101)
        end_date = get_previous_trading_date(context.now, 1)
        model_X = model_generator.xdata(start_date, end_date)  # 获取对应股票最近100个交易日的输入数据
        predictions = context.eligible_models[stock].predict(context.eligible_models_scalers[stock].transform(model_X))
        if predictions[-1]:
            context.buy_list.append(stock)
        else:
            pass


def order_buy_list(context, bar_dict):
    for stock in context.buy_list:
        order_percent(stock, context.eligible_models_weights[stock])
        record_transaction(context, bar_dict, stock)


def check_and_sell(context, bar_dict):
    """
    对于持有的股票，如果现在市场价高于其成本价1%以上，则全部卖出
    :param context:
    :param bar_dict:
    :return:
    """
    for i in context.transaction.index:
        if bar_dict[i].last >= context.transaction.loc[i].cost * 1.01:
            order_shares(i, amount=-1 * context.portfolio.positions[i].quantity)
        else:
            pass


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    make_prediction(context)  # 每天开始交易前，计算有哪些股票上涨的概率较大


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    pass


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass
