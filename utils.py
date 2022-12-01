import country_converter as coco
import pandas as pd
import pycountry
import functools


@functools.lru_cache(None)
def standard_country_names(name):
    return coco.convert(name, to='ISO3', not_found=None)

@functools.lru_cache(None)
def do_fuzzy_search(country):
    try:
        result = pycountry.countries.search_fuzzy(country)
    except Exception:
        return np.nan
    else:
        return result[0].alpha_3

empty_bar =  {
    "layout": {
        "xaxis": {
            "visible": False
        },
        "yaxis": {
            "visible": False
        },
    "paper_bgcolor": '#a4b6fa',
        "margin": {'l':20, 'r':20, 't':20, 'b':20},
        "annotations": [
            {
                "text": "Empty Plot: Please Select States on Map",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 20
                }
            }
        ],

    }
}

empty_heatmap =  {
    "layout": {
        "xaxis": {
            "visible": False
        },
        "yaxis": {
            "visible": False
        },
    "paper_bgcolor": '#a4b6fa',
        "margin": {'l':20, 'r':20, 't':20, 'b':20},
        "annotations": [
            {
                "text": "Empty Heatmap: Please set params and hit the button!",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 20
                }
            }
        ],

    }
}

empty_pred_line =  {
    "layout": {
        "xaxis": {
            "visible": False
        },
        "yaxis": {
            "visible": False
        },
    "paper_bgcolor": '#a4b6fa',
        "margin": {'l':20, 'r':20, 't':20, 'b':20},
        "annotations": [
            {
                "text": "Empty Prediction: Please set params and hit the button!",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 20
                }
            }
        ],

    }
}
#
# diet_data_path = './data/Food_Supply_kcal_Data.csv'
# activity_data_path = './data/Food_Supply_kcal_Data.csv'
# covid_data_path = './data/owid-covid-data.csv'
# diet_df = pd.read_csv(diet_data_path)
# diet_df['standard'] = ['ISO3']*diet_df.shape[0]
# activity_df = pd.read_csv(activity_data_path)
# covid_df = pd.read_csv(covid_data_path)
#
#
# covid_df = covid_df.loc[~covid_df['location'].isin(['Africa', 'Asia', 'Europe', 'European Union', 'International', 'North America', 'Oceania', 'South America', 'World', 'Timor'])]
# diet_df['Country_names'] = diet_df['Country'].apply(standard_country_names)
# covid_df['location'] = covid_df['location'].apply(standard_country_names)
#
# diet_df.to_csv('./data/Food_Supply_kcal_Data_new.csv')
# covid_df.to_csv('./data/owid-covid-data_new.csv')
