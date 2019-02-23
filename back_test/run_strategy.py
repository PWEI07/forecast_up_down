from rqalpha import run_file
import matplotlib.pyplot as plt
import pandas as pd

config = {
    "base": {
        "start_date": "2017-10-09",
        "end_date": "2018-11-21",
        "run_type": "b",
        "benchmark": "000985.XSHG",
        "frequency": "1d"
        , "data_bundle_path": "/Users/peter_zirui_wei/Documents/rqpro/bundle"
        , "accounts": {
            "future": 1000000,
            "stock": 1000000000
        }
    },
    "extra": {
        "log_level": "debug",
    },
    "mod": {
        "sys_simulation": {
            "priority": 100,
            "slippage": 0.001,
            "matching_type": 'current_bar',
            "plot": True,
            "volume_limit": False
        }
        # ,
        # "ricequant_data": {
        #     "priority": 101,
        #     "enabled": True,
        #     "rqdata_client_addr": "rqdatad.ricequant.com",
        #     "rqdata_client_port": 16003,
        #     "rqdata_client_username": "ricequant",
        #     "rqdata_client_password": "Ricequant123",
        #     # "redis_url": "redis://tinker:6379/2"
        #     "redis_url": "redis://paladin:6380/2"
        # }
    }
}

# result1 = run_file("F://ricequant_internship//ENG-7598//without_daily_low_sentiment_strategy.py", config)['sys_analyser']
# series1 = result1['portfolio']['unit_net_value']  # 去除掉最低一组后的走势
# series1.name = 'without_daily_low_sentiment'

# result2 = run_file('without_hourly_low_sentiment_strategy.py', config)['sys_analyser']
# series2 = result2['portfolio']['unit_net_value']
# series2.name = 'without_hourly_low_sentiment'

# result3 = run_file('benchmark_strategy.py', config)['sys_analyser']
# series3 = result3['portfolio']['unit_net_value']  # 所有股票等权的走势
# series3.name = 'same_weighted_benchmark'

# result4 = run_file("daily_high_sentiment_strategy.py", config)['sys_analyser']
# series4 = result1['portfolio']['unit_net_value']  # 去除掉最低一组后的走势
# series4.name = 'without_daily_low_sentiment'
result5 = run_file("/Users/peter_zirui_wei/PycharmProjects/forecast_up_down/back_test/random_forest_50.py", config)['sys_analyser']

# result5 = run_file("/Users/peter_zirui_wei/PycharmProjects/forecast_up_down/back_test/random_forest_50.py", config)['sys_analyser']
series5 = result5['portfolio']['unit_net_value']  # 去除掉最低一组后的走势
series5.name = 'daily_high_sentiment'

# compare_df = pd.concat([series1, series5, series3], axis=1)
# compare_df.columns = ['without_daily_low_sentiment', 'daily_high_sentiment', 'same_weighted_benchmark']
# compare_df.plot()
# plt.show()



