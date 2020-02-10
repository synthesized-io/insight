import json
import urllib.parse
import urllib.request
from operator import itemgetter
from urllib.error import  HTTPError


def lookup_addresses(postcode):
    base_url = "https://api.getAddress.io/find/{postcode}".format(postcode=urllib.request.quote(postcode))
    params = urllib.parse.urlencode({'api-key': 'tQwTXklRgku2psx6TJbATQ22209', 'expand': True})
    url = base_url + '?' + params
    with urllib.request.urlopen(url) as response:
        data = response.read()
        js = json.loads(data.decode('utf-8'))
        js['postcode_orig'] = postcode
        return js


def lookup_postcode(postcode):
    url = "https://api.postcodes.io/postcodes/{postcode}".format(postcode=urllib.request.quote(postcode))
    with urllib.request.urlopen(url) as response:
        data = response.read()
        js = json.loads(data.decode('utf-8'))
        return js['result']['postcode']


def lookup_nearest_postcodes(postcode):
    url = "https://api.postcodes.io/postcodes/{postcode}/nearest".format(postcode=urllib.request.quote(postcode))
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
            js = json.loads(data.decode('utf-8'))
            return list(map(itemgetter('postcode'), js['result']))
    except HTTPError:
        return [postcode]


postcodes = ['IV316AP', 'PO139NS', 'IV170YB', 'CA263XG', 'CO139HJ', 'IM9 3EB',
             'GL167RU', 'SE181DB', 'NP132JL', 'TW7 6LG', 'HS2 0NG', 'PO318HA',
             'L23 3AG', 'BH232JG', 'BN3 1TN', 'SE128BH', 'LL545PS', 'PE218NU',
             'SY226UN', 'W6  0RQ', 'WR9 8NY', 'CF634QQ', 'GL7 1LL', 'PA607XZ',
             'CM4 0NQ', 'GU7 3AW', 'EH320JS', 'KA256BN', 'PH152AH', 'EH146AE',
             'LL452EQ', 'SE3 0QX', 'HR7 4JG', 'SW130EH', 'PA417AD', 'KY7 6BS',
             'NW4 4QD', 'RH3 7BW', 'PH243BL', 'CT201NR', 'FK147DZ', 'BB87JJ',
             'LL670NN']

postcodes_pa = ['IV327LP', 'CT202EJ', 'GL179QS']
postcodes_pa_ext = ['IV32 7LP', 'CT20 2EJ', 'CT20 2EH', 'CT20 2EF', 'CT20 2EW', 'CT20 2HJ', 'CT20 2HH', 'CT20 2EL', 'CT20 2ET', 'GL17 9QS']

# with open('postcodes.jsonl', 'a') as f:
#     for postcode in postcodes:
#         js = lookup_addresses(postcode)
#         json.dump(js, f)
#         f.write('\n')
#
# with open('postcodes_pa.jsonl', 'a') as f:
#     for postcode in postcodes_pa:
#         js = lookup_addresses(postcode)
#         json.dump(js, f)
#         f.write('\n')

# with open('postcodes_pa_ext.jsonl', 'a') as f:
#     for postcode in postcodes_pa_ext:
#         js = lookup_addresses(postcode)
#         json.dump(js, f)
#         f.write('\n')

# print(lookup_postcode('IM93EB'))
# print(lookup_nearest_postcodes(lookup_postcode('IM93EB')))

# postocdes_ext = [n for p in postcodes for n in lookup_nearest_postcodes(p)]
# print(len(postcodes), len(postocdes_ext))

# postocdes_pa_ext = [n for p in postcodes_pa for n in lookup_nearest_postcodes(p)]
# print(len(postcodes_pa), len(postocdes_pa_ext))
# print(postocdes_pa_ext)
