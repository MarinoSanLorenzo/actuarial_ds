import pandas as pd
from src.constants import Constants, params



df = pd.read_csv(params.get(Constants.URL_LINK_TO_DATA), delimiter = ",")