import talib
from tree_clf.random_forest import ForestBuilder
from datetime import timedelta
from pandas import Series, DataFrame
model = ForestBuilder  # 分类所用的模型

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


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass

# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):

    # 因为策略需要用到均线，所以需要读取历史数据
    prices = history_bars(context.s1, context.LONGPERIOD+1, '1d', 'close')

    # 使用talib计算长短两根均线，均线以array的格式表达
    short_avg = talib.SMA(prices, context.SHORTPERIOD)
    long_avg = talib.SMA(prices, context.LONGPERIOD)

    plot("short avg", short_avg[-1])
    plot("long avg", long_avg[-1])

    # 计算现在portfolio中股票的仓位
    cur_position = context.portfolio.positions[context.s1].quantity
    # 计算现在portfolio中的现金可以购买多少股票
    shares = context.portfolio.cash/bar_dict[context.s1].close

    # 如果短均线从上往下跌破长均线，也就是在目前的bar短线平均值低于长线平均值，而上一个bar的短线平均值高于长线平均值
    if short_avg[-1] - long_avg[-1] < 0 and short_avg[-2] - long_avg[-2] > 0 and cur_position > 0:
        # 进行清仓
        order_target_value(context.s1, 0)

    # 如果短均线从下往上突破长均线，为入场信号
    if short_avg[-1] - long_avg[-1] > 0 and short_avg[-2] - long_avg[-2] < 0:
        # 满仓入股
        order_shares(context.s1, shares)

# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass