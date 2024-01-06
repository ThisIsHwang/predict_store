import pandas as pd
import numpy as np
import dask.dataframe as dd

class Utils:
    def __init__(self):
        self.commercial_list = ["식육(숯불구이)", "김밥(도시락)", "패밀리레스트랑", "골프연습장업", "노래연습장업", "일반조리판매", "체력단련장업",
                                "통닭(치킨)", "문화예술법인", "관광공연장업", "사회복지시설", "일반야영장업", "관광숙박업", "동물미용업", "무도학원업",
                                "복합쇼핑몰", "석유판매업", "아이스크림", "영화상영관", "제과점영업", "체육도장업", "축산판매업", "패스트푸드",
                                "한옥체험업", "호프/통닭", "동물전시업", "라이브카페", "축산가공업", "동물판매업", "산후조리업", "산후조리원",
                                "간이주점", "감성주점", "공공기관", "노래클럽", "단란주점", "당구장업", "대형마트", "동물병원", "동물약국",
                                "목욕장업", "수영장업", "스텐드바", "어린이집", "전통사찰", "전통찻집", "푸드트럭", "관광호텔", "쇼핑센터",
                                "극장식당", "무도장업", "키즈카페", "경양식", "공연장", "과자점", "냉면집", "떡카페", "룸살롱", "미용업",
                                "백화점", "뷔페식", "상조업", "세탁업", "숙박업", "안경업", "이용업", "인쇄사", "중국식", "출판사", "카바레",
                                "커피숍", "편의점", "기숙사", "제재업", "까페", "다방", "병원", "분식", "약국", "요정", "의원", "일식", "학교",
                                "한식", "횟집", "시장", "극장"]
        
        self.distribution_data_path = ['data/converted_filtered_data.csv']
    
    # private function
    def __load_data(self, path):
        dtype = {'업태구분명': 'object'}
        usecols = ['좌표정보(x)', '좌표정보(y)', '업태구분명', '영업상태명', '사업장명']
        data = dd.read_csv(path, assume_missing=True, usecols=usecols, dtype=dtype)
        data = data[data['영업상태명'] == '영업/정상']
        data = data.compute()
        return data
    
    # utils = Utils()
    # utils.load_data()
    # Data can be preloaded.
    # Even if there is no data loaded because load_data() is not running, cnt_distribution() can check and load it.
    # Therefore, it can be used when the user wants to load data in the background before running cnt_distribution().
    def load_data(self, path=None):
        if not hasattr(self, 'data'):
            if path is None:
                path = self.distribution_data_path
            self.data = pd.concat([self.__load_data(path) for path in self.distribution_data_path], ignore_index=True)
            self.data = self.data.drop_duplicates()
            self.distribution_data_path = None

    # private function
    def __vectorized_haversine(self, lat1, lon1, lat2_array, lon2_array):
        lon1, lat1, lon2_array, lat2_array = np.radians(lon1), np.radians(lat1), np.radians(lon2_array), np.radians(lat2_array)
        dlon = lon2_array - lon1
        dlat = lat2_array - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2_array) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = 6371
        return R * c
    
    # utils = Utils()
    # utils.cnt_distribution(lat, lon, rad)
    # input: latitude, longitude, radius(km)
    # ex)  utils.cnt_distribution(37.519939, 126.904052, 1)
    # output: dictionary
    # ex) ({"병원" : 1, "주차장" : 2, "교회" : 1, …})
    def cnt_distribution(self, lat, lon, rad):
        self.load_data()

        distances = self.__vectorized_haversine(lat, lon, self.data['좌표정보(y)'].values, self.data['좌표정보(x)'].values)
        mask = distances < rad
        data = self.data[mask]

        result = {}
        for value in self.commercial_list:
            result[value] = data[data['업태구분명'] == value].shape[0]
        return result

# Example usage

# utils = Utils()
# utils.load_data()
# print(utils.cnt_distribution(37.519939, 126.904052, 1))
# print(utils.cnt_distribution(37.519939, 126.904052, 1))
import pandas as pd


def process_hashes(data):
    data['hashes'] = data['hashes'].str.strip('[]').str.split(',')
    all_hashes = set()
    for hashes in data['hashes']:
        all_hashes.update(hashes)
    all_hashes.discard('')
    for hash in all_hashes:
        data["whether_" + hash] = data['hashes'].apply(lambda x: 1 if hash in x else 0)
    data.drop('hashes', axis=1, inplace=True)
    return data


def load_data():
    data_folder = "data/"
    files = ['inter_diningcode_youngdeungpo_dropped_20241.csv', 'inter_diningcode_jongro_dropped_20241.csv', 'inter_diningcode_gangnam_dropped_20241.csv',
             'inter_hashes_youngdeungpo_dropped_20241.csv', 'inter_hashes_jongro_dropped_20241.csv', 'inter_hashes_gangnam_dropped_20241.csv']

    datasets = [pd.read_csv(data_folder + file) for file in files]

    for dataset in datasets:
        columns_to_drop = ['categories', 'Info Title', 'Title Place', 'score', 'userScore', 'heart',
                           'title', 'address', 'roadAddress', 'mapx', 'mapy']
        dataset.drop(columns=columns_to_drop, inplace=True)

    youngdeungpo_data, jongro_data, gangnam_data, youngdeungpo_hashes, jongro_hashes, gangnam_hashes = datasets

    # youngdeungpo_data = apply_hashes_processing(youngdeungpo_data)
    # jongro_data = apply_hashes_processing(jongro_data)
    # gangnam_data = apply_hashes_processing(gangnam_data)

    youngdeungpo_data = pd.merge(youngdeungpo_data, youngdeungpo_hashes,
                                 on=['Title', 'Latitude', 'Longitude', 'category', 'hashes'], how='inner')
    jongro_data = pd.merge(jongro_data, jongro_hashes, on=['Title', 'Latitude', 'Longitude', 'category', 'hashes'], how='inner')
    gangnam_data = pd.merge(gangnam_data, gangnam_hashes, on=['Title', 'Latitude', 'Longitude', 'category','hashes'], how='inner')

    youngdeungpo_data.drop_duplicates(subset=['Title', 'Latitude', 'Longitude', 'category'], inplace=True)
    jongro_data.drop_duplicates(subset=['Title', 'Latitude', 'Longitude', 'category'], inplace=True)
    gangnam_data.drop_duplicates(subset=['Title', 'Latitude', 'Longitude', 'category'], inplace=True)

    data = pd.concat([youngdeungpo_data, jongro_data, gangnam_data], ignore_index=True)

    data['category'] = data['category'].str.split('>').str[-1]

    data_for_hash = process_hashes(data.copy(deep=True))

    data_for_type = data.drop(columns=['hashes'])
    data_for_type.fillna(0, inplace=True)
    data_for_hash.fillna(0, inplace=True)

    return data_for_type, data_for_hash

def process_hashes_string(hashes_str):
    """
    Convert a comma-separated string into a list of non-empty items,
    unless it's already in a list-like format.

    :param hashes_str: A string containing items separated by commas or a string representation of a list.
    :return: A list of non-empty items, or the original string if it's already list-like.
    """
    # Check if the string is already in a list-like format
    if hashes_str.startswith('[') and hashes_str.endswith(']'):
        return hashes_str

    # Split the string by commas and strip whitespace
    items = [item.strip() for item in hashes_str.split(',') if item.strip()]
    return str(items)

def apply_hashes_processing(df, column_name='hashes'):
    """
    Apply the hashes processing to a specific column in a DataFrame.

    :param df: The DataFrame to process.
    :param column_name: The name of the column containing the hashes.
    :return: A DataFrame with the processed column.
    """
    df[column_name] = df[column_name].apply(process_hashes_string)
    return df
