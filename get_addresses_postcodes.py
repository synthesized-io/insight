import os
import sys
import requests
import gzip

import numpy as np
import pandas as pd
import simplejson

API_KEY = 'tQwTXklRgku2psx6TJbATQ22209'

ALL_POSTCODES_LEVEL_0 = [
    'AB', 'AL', 'B', 'BA', 'BB', 'BD', 'BF', 'BH', 'BL', 'BN', 'BR', 'BS', 'BT', 'BX', 'CA', 'CB', 'CF', 'CH', 'CM',
    'CO', 'CR', 'CT', 'CV', 'CW', 'DA', 'DD', 'DE', 'DG', 'DH', 'DL', 'DN', 'DT', 'DY', 'E', 'EC', 'EH', 'EN', 'EX',
    'FK', 'FY', 'G', 'GL', 'GU', 'GY', 'HA', 'HD', 'HG', 'HP', 'HR', 'HS', 'HU', 'HX', 'IG', 'IM', 'IP', 'IV', 'JE',
    'KA', 'KT', 'KW', 'KY', 'L', 'LA', 'LD', 'LE', 'LL', 'LN', 'LS', 'LU', 'M', 'ME', 'MK', 'ML', 'N', 'NE', 'NG',
    'NN', 'NP', 'NR', 'NW', 'OL', 'OX', 'PA', 'PE', 'PH', 'PL', 'PO', 'PR', 'RG', 'RH', 'RM', 'S', 'SA', 'SE', 'SG',
    'SK', 'SL', 'SM', 'SN', 'SO', 'SP', 'SR', 'SS', 'ST', 'SW', 'SY', 'TA', 'TD', 'TF', 'TN', 'TQ', 'TR', 'TS', 'TW',
    'UB', 'W', 'WA', 'WC', 'WD', 'WF', 'WN', 'WR', 'WS', 'WV', 'YO', 'ZE'
]


def _get_postcode_key(postcode: str, postcode_level: int = 0):
    if postcode == 'NaN':
        return 'NaN'

    if postcode_level == 0:  # 1-2 letters
        index = 2 - postcode[1].isdigit()
    elif postcode_level == 1:
        index = postcode.index(' ')
    elif postcode_level == 2:
        index = postcode.index(' ') + 2
    else:
        raise ValueError(postcode_level)
    return postcode[:index]


if __name__ == '__main__':

    synth_path = sys.argv[1]
    assert os.path.exists(synth_path + 'data/addresses.jsonl.gz')

    stored_postcodes = []
    f_out = gzip.open(synth_path + 'data/addresses_out.jsonl.gz', 'w')
    with gzip.open(synth_path + 'data/addresses.jsonl.gz', 'r') as f:
        for line in f:
            f_out.write(line)
            js = simplejson.loads(line)
            stored_postcodes.append(js['postcode'])

    if os.path.exists(synth_path + 'Postcode_Estimates_Table_1.csv'):
        df = pd.read_csv(synth_path + 'Postcode_Estimates_Table_1.csv')
        print('Total num. postcodes', len(df))
        df = df[df['Postcode'].apply(lambda pc: False if pc in stored_postcodes else True)]
        print('Num. postcodes cleaned', len(df))

        df['PostcodeKey'] = df['Postcode'].apply(_get_postcode_key)
    else:
        print("To load first postcodes with higher population density, download estimates from "
              "'https://www.nomisweb.co.uk/output/census/2011/Postcode_Estimates_Table_1.csv' and store them in this "
              "folder")
        df = None

    print('** Start looping **')
    keep_looping = True
    while keep_looping:
        for postcode_key in ALL_POSTCODES_LEVEL_0:
            if df is not None and postcode_key in df['PostcodeKey'].unique():
                max_pc = df.loc[df['PostcodeKey'] == postcode_key, 'Total'].max()
                postcode = df.loc[(df['PostcodeKey'] == postcode_key) & (df['Total'] == max_pc), 'Postcode'].values[0]
            else:
                url_pc_key = 'https://api.postcodes.io/postcodes/{}/autocomplete'.format(postcode_key)
                r_pc_key = requests.get(url_pc_key)
                postcodes = r_pc_key.json()['result']
                if postcodes and len(postcodes) > 0:
                    postcode = np.random.choice(postcodes)
                else:
                    continue

            if postcode in stored_postcodes:
                continue

            url = 'https://api.getAddress.io/find/{postcode}?api-key={api_key}&expand=true'.format(postcode=postcode,
                                                                                                   api_key=API_KEY)
            r = requests.get(url)

            # Write to file
            if r.ok:
                js = r.json()
                f_out.write(simplejson.dumps(r.json()))

                # Delete from DF
                if df is not None:
                    df = df[df['Postcode'] != postcode]

            else:
                if r.status_code in (400, 404):
                    print('Postcode {} NOT FOUND'.format(postcode))
                else:
                    keep_looping = False
                    break

    f_out.close()
    os.rename(synth_path + 'data/addresses_out.jsonl.gz',
              synth_path + 'data/addresses.jsonl.gz')

