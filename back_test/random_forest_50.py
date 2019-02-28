from tree_clf.random_forest import ForestBuilder
from datetime import timedelta
from pandas import Series, DataFrame
import numpy as np
import rqdatac as rd
rd.init()


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    context.eligible_models = {}  # 用于存储通过标准的模型
    context.eligible_models_weights = Series()  # 用于存储达标模型的资金权重
    context.eligible_models_scalers = {}  # 用于存储达标模型的scaler

    scheduler.run_monthly(get_models, tradingday=1)  # 每个月训练一次模型，选出符合标准的放入context.eligible_models
    context.precision_threshold = 0.7  # 设置精确度阈值，通过此阈值的模型才可以入选
    context.F_threshold = 0.8  # 设置F1 score阈值，通过此阈值的模型才可以入选
    context.transaction = DataFrame(columns=['date', 'cost'])  # 用于记录买入的品种，日期，价格
    scheduler.run_daily(sell_old, time_rule=market_open(minute=0))  # 每天09:31卖掉持有日大于等于5天的股票
    scheduler.run_daily(make_prediction, time_rule=market_open(minute=1))  # 每天09:32进行预测
    scheduler.run_daily(order_buy_list, time_rule=market_open(minute=2))  # 每天09:33买入context.buy_list中的股票

    context.buy_list = []  # 预测要涨 需要买的股票


def get_models(context, bar_dict):
    """
    获得该期达标模型，以及其scaler，并根据其模型得分计算对其分配的资金权重
    :param context:
    :return:
    """
    # 每个月 首先更新股票池
    context.pool = index_components('000016.XSHG')
    # context.pool = ['600489.XSHG', '600585.XSHG', '601899.XSHG', '600188.XSHG', '600348.XSHG']
    context.model_generators = {}  # 用于存储投资池中各个股票的模型生成器
    for i in context.pool:
        context.model_generators[i] = ForestBuilder(i)

    context.eligible_models = {}  # 先将上一期达标模型清空
    context.eligible_models_scalers = {}
    context.eligible_models_weights = Series()
    end_date = get_previous_trading_date(context.now, n=1).strftime('%Y-%m-%d')
    start_date = (context.now - timedelta(days=365 * 4)).strftime('%Y-%m-%d')

    for i in context.pool:
        model_i = context.model_generators[i].get_eligible_model(start_date, end_date,
                                                                 precision_threshold=context.precision_threshold,
                                                                 F_threshold=context.F_threshold)
        if model_i is not None:
            print(i, ':')
            print(model_i, '\n\n')
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
    temp_sell_df = context.transaction[(context.now - context.transaction['date']) >= timedelta(days=5)]
    for i in temp_sell_df.index:
        print('start to sell old ', i)
        order_shares(i, amount=-1 * context.portfolio.positions[i].sellable)
        if context.portfolio.positions[i].quantity == 0:
            context.transaction.drop(index=i, inplace=True)
        else:
            pass


def record_transaction(context, bar_dict, order_book_id):
    """
    记录做多的股票代码 开仓时间 开仓价格
    :param context:
    :return:
    """
    context.transaction = context.transaction.append(Series(index=['date', 'cost'],
                                                            data=[context.now, bar_dict[order_book_id].last],
                                                            name=order_book_id))


def make_prediction(context, bar_dict):
    """
    对每个达标模型，获取对应股票最近60个交易日的输入数据，利用模型进行预测
    :param context:
    :return:
    """
    context.buy_list = []
    start_date = get_previous_trading_date(context.now, 101)
    end_date = get_previous_trading_date(context.now, 1)

    for stock in context.eligible_models:
        if stock in context.portfolio.positions:
            continue  # 如果该模型已经开仓 则跳过
        else:
            pass
        model_generator = context.model_generators[stock]  # 调取模型生成器
        model_X = model_generator.xdata(start_date, end_date)  # 获取对应股票最近100个交易日的输入数据
        predictions = context.eligible_models[stock].predict(context.eligible_models_scalers[stock].transform(model_X))
        if predictions[-1]:
            context.buy_list.append(stock)
        else:
            pass


def order_buy_list(context, bar_dict):
    for stock in context.buy_list:
        print('buy stock ', stock)
        order = order_percent(stock, 0.8 * context.eligible_models_weights[stock])
        if order is not None:
            if order.filled_quantity != 0:
                record_transaction(context, bar_dict, stock)
            else:
                pass
        else:
            pass


def check_and_sell(context, bar_dict):
    """
    对于持有的股票，如果现在市场价高于其成本价1%以上，则全部卖出
    :param context:
    :param bar_dict:
    :return:
    """
    for i in context.transaction.index:

        if bar_dict[i].last >= np.mean(context.transaction.loc[i].cost) * 1.015:
            print('sell and make profit on ', i, '\n\n')
            order_shares(i, amount=-1 * context.portfolio.positions[i].sellable)
            if context.portfolio.positions[i].quantity == 0:
                context.transaction.drop(index=i, inplace=True)
            else:
                pass
        else:
            pass


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    for i in context.transaction.index:
        if i not in context.portfolio.positions.keys():
            print(i, 'is in context.transaction.index, but not in context.portfolio.positions.keys() !\n\n')
            context.transaction.drop(index=i, inplace=True)
        else:
            pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    check_and_sell(context, bar_dict)
    # minute = context.now.minute
    # if minute == 1 or minute == 29:
    #     check_and_sell(context, bar_dict)  # 加快运行速度 只在每小时的第1 29分钟检查是否需要获利了结
    # else:
    #     pass


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass
