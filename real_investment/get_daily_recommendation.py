from tree_clf.random_forest import ForestBuilder
from datetime import timedelta
from pandas import Series
import rqdatac as rd
rd.init()


def predict0226(current_date):
    """
    这个函数不是用于回测的 是方便临时预测 辅助投资决策
    :param current_date:
    :return:
    """
    end_date = rd.get_previous_trading_date(current_date, n=1).strftime('%Y-%m-%d')
    start_date = (current_date - timedelta(days=365 * 4)).strftime('%Y-%m-%d')

    pool = rd.index_components('000300.XSHG', date=end_date)
    model_generators = {}  # 用于存储投资池中各个股票的模型生成器
    eligible_models = {}  # 用于存储通过标准的模型
    eligible_models_weights = Series()  # 用于存储达标模型的资金权重
    eligible_models_scalers = {}  # 用于存储达标模型的scaler
    for i in pool:
        model_generators[i] = ForestBuilder(i)
    for i in pool:
        model_i = model_generators[i].get_eligible_model(start_date, end_date,
                                                                 precision_threshold=0.75,
                                                                 F_threshold=0.8)
        if model_i is not None:
            print(i, ':')
            print(model_i, '\n\n')
            eligible_models[i] = model_i['best_rf']
            eligible_models_scalers[i] = model_i['scaler']
            eligible_models_weights[i] = model_i['test_precision'] + model_i['test_f1']
    eligible_models_weights /= eligible_models_weights.sum()
    buy_list = Series()
    start_date2 = rd.get_previous_trading_date(current_date, 101)

    for stock in eligible_models:
        model_generator = model_generators[stock]  # 调取模型生成器
        model_X = model_generator.xdata(start_date2, end_date)  # 获取对应股票最近100个交易日的输入数据
        predictions = eligible_models[stock].predict(eligible_models_scalers[stock].transform(model_X))
        if predictions[-1]:
            buy_list[stock] = eligible_models_weights[stock]
        else:
            pass
    return buy_list
