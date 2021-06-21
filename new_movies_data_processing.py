import pandas as pd

df = pd.read_csv("data.csv")


new_df = df.drop(columns=["content_genre","timestamp","bitrate","player_version","rebuffer_count","rebuffer_duration","playback_duration"])



def calcRebufferRates(column_name):

    df[column_name] = df[column_name].str.lower()
    rebuffer_duration = df.groupby([column_name]).rebuffer_duration.sum().reset_index()


    playback_duration = df.groupby([column_name]).playback_duration.sum().reset_index()

    rebuffer_duration_sum = rebuffer_duration["rebuffer_duration"].values.tolist()
    categories = rebuffer_duration[column_name].values.tolist()
    playback_duration_sum = playback_duration["playback_duration"].values.tolist()

    Rebuffer = []

    for i, j in zip(rebuffer_duration_sum, playback_duration_sum):

        try:

            Rebuffer.append(i/j)

        except:

            Rebuffer.append("NaN")

    fatal_true_count_values  =  df.groupby(column_name)['has_fatal_error'].sum().values.tolist()

    fatal_total_count_values  =  df[[column_name]].value_counts().values.tolist()

    error_rate = []

    for i, j in zip(fatal_true_count_values, fatal_total_count_values):

        try:

            error_rate.append(i/j)

        except:

            error_rate.append("NaN")

    final_df = pd.DataFrame()

    final_df["Categories"] = categories

    # final_df["rebuffer_duration_sum"] = rebuffer_duration_sum
    #
    # final_df["playback_duration_sum"] = playback_duration_sum

    final_df["RebufferRate"] = Rebuffer

    # final_df["fatal_true_count_values"] = fatal_true_count_values
    # final_df["fatal_total_count_values"] = fatal_total_count_values
    final_df["error_rate"] = error_rate

    final_df.sort_values('error_rate',inplace=True)

    final_df.to_csv(column_name+".csv")




calcRebufferRates("user_geoip_isp")



print()
