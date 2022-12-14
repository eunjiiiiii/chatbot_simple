from searcher.base_searcher import BaseSearcher
import re


class WeatherSearcher(BaseSearcher):

    def __init__(self):
        self.CSS = {
            # 검색에 사용할 CSS 셀렉터들을 정의합니다.
            'naver_weather': 'div.weather_info > div > div._today > div.weather_graphic > div.weather_main > i > span.blind',
            'naver_temperature': 'div.weather_info > div > div._today > div.weather_graphic > div.temperature_text > strong',
            'naver_temperature_feel': 'div.weather_info > div > div._today > div.temperature_info > dl > dd.desc',
            'google_weather': '#wob_dcp > #wob_dc',
            'google_temperature': '#wob_tm'
        }

        self.data_dict = {
            # 데이터를 담을 딕셔너리 구조를 정의합니다.
            'today_weather': None,  # '맑음,어제보다 1'낮아요'
            'today_temperature': None,  # '21',
            'today_temperature_feel': None, # '22.5'',
            'today_humidity': None, #'75%'
            'today_wind': None # 1.7m/s
        }

    def _make_query(self, location: str, date: str) -> str:
        """
        검색할 쿼리를 만듭니다.
        
        :param location: 지역
        :param date: 날짜
        :return: "지역 날짜 날씨"로 만들어진 쿼리
        """

        return ' '.join([location, date, '날씨'])

    def naver_search(self, location: str) -> dict:
        """
        네이버를 이용해 날씨를 검색합니다.

        :param location: 지역
        :return: 크롤링된 내용
        """

        query = self._make_query('오늘', location)  # 한번 서치에 전부 가져옴
        result = self._bs4_contents(self.url['naver'],
                                    selectors=[self.CSS['naver_weather'],
                                               #self.CSS['naver_temperature_compare'],
                                               self.CSS['naver_temperature'],
                                               self.CSS['naver_temperature_feel']],
                                    query=query)

        i = 0
        for k in self.data_dict.keys():
            if 'specific' not in k:
                # specific 빼고 전부 담음
                self.data_dict[k] = re.sub(' ', '', result[i][0])
                i += 1

        return self.data_dict

    def google_search(self, location: str, date: str) -> dict:
        """
        구글을 이용해 날씨를 검색합니다.

        :param location: 지역
        :param date: 날짜
        :return: 크롤링된 내용
        """

        query = self._make_query(location, date)  # 날짜마다 따로 가져와야함
        result = self._bs4_contents(self.url['google'],
                                    selectors=[self.CSS['google_weather'],
                                               self.CSS['google_temperature']],
                                    query=query)

        self.data_dict['specific_weather'] = re.sub(' ', '', result[0][0])
        self.data_dict['specific_temperature'] = re.sub(' ', '', result[1][0])
        # specific만 담음

        return self.data_dict
