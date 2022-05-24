def is_integer_num(n):
     if isinstance(n, int):
         return True
     if isinstance(n, float):
         return n.is_integer()
     return False

def convert_dict_str_vals_to_float(d):
    for k, v in d.items():
        d[k] = float(v) if is_float(v) else v
    return d

def is_float(v) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False

def show(label, value):
    print(label + ': ' + str(value))

def split_list(list, key, key_list_1, key_list_2):
    l1, l2 = [], []
    for item in list:
        if item['value']:
            if item[key] in key_list_1:
                l1.append(item)
            if item[key] in key_list_2:
                l2.append(item)
    return l1, l2

def list_to_dict(list, key, value, check = False):
    dict = {}
    for item in list:
        if (check and item[value]) or not check:
            dict[item[key]] = item[value]
    return dict

def filter_list(data, key, filter_value):
    return list(filter(lambda x: x[key] == filter_value, data))

def shared_items(dict_1, dict_2):
    return {k: dict_1[k] for k in dict_1 if k in dict_2 and dict_1[k] == dict_2[k]}
