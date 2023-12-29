import pandas as pd
import numpy as np
import math

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
        
        self.distribution_data_path = ['data/final_merged_filtered_youngdeungpo_data.csv',
                                       'data/final_merged_filtered_jongro_data.csv']
    
    # private function
    def __load_data(self, path):
        data = pd.read_csv(path, low_memory=False)
        data = data[['좌표정보(x)', '좌표정보(y)', '업태구분명', '영업상태명', '사업장명']]
        data = data[data['영업상태명'] == '영업/정상']
        return data
    
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
        if not hasattr(self, 'data'):
            self.data = pd.concat([self.__load_data(path) for path in self.distribution_data_path], ignore_index=True)
            self.data = self.data.drop_duplicates()
            self.distribution_data_path = None

        distances = self.__vectorized_haversine(lat, lon, self.data['좌표정보(y)'].values, self.data['좌표정보(x)'].values)
        mask = distances < rad
        data = self.data[mask]

        result = {}
        for value in self.commercial_list:
            result[value] = data[data['업태구분명'] == value].shape[0]
        return result