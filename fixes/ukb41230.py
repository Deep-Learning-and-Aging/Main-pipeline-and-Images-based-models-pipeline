import pandas as pd


data = pd.read_csv("data/fake_short_ukb41230.csv").set_index("eid")

data.loc[data["34-0.0"].isna(), "34-0.0"] = 1947.0
data.loc[data["52-0.0"].isna(), "52-0.0"] = 11.0
data.to_csv("data/fake_short_ukb41230.csv")