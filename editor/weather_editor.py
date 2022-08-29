from editor.base_editor import BaseEditor
import re


class WeatherEditor(BaseEditor):

    def edit_today(self, result: dict) -> dict:
        """
        오늘 날씨 딕셔너리를 수정합니다.

        :param result: 입력 딕셔너리
        :return: 수정된 딕셔너리
        """

        weather = result['today_weather']
        #comparison = result['today_weather'].split(',')[1].strip()
        temperature = result['today_temperature']
        temperature_feel = result['today_temperature_feel'].replace('°','')
        humidity = result['today_humidity']
        wind = result['today_wind']

        result = {'weather': weather,
                  #'comparison': comparison,
                  'temperature': temperature,
                  'temperature_feel': temperature_feel,
                  'humidity': humidity,
                  'wind': wind}

        return result

    def edit_tomorrow(self, result: dict) -> tuple:
        """
        내일 날씨 딕셔너리를 수정합니다.

        :param result: 입력 딕셔너리
        :return: 수정된 딕셔너리
        """

        morning_weather = self.weather[result['tomorrow_morning_weather']]
        afternoon_weather = self.weather[result['tomorrow_afternoon_weather']]
        morning_temperature = result['tomorrow_morning_temperature']
        afternoon_temperature = result['tomorrow_afternoon_temperature']
        josa = self.enumerate_josa('는', '도', [morning_weather, afternoon_weather])

        result = {'morning_weather': morning_weather,
                  'afternoon_weather': afternoon_weather,
                  'morning_temperature': morning_temperature,
                  'afternoon_temperature': afternoon_temperature}

        return result, josa

    def edit_after(self, result: dict) -> tuple:
        """
        모레 날씨 딕셔너리를 수정합니다.

        :param result: 입력 딕셔너리
        :return: 수정된 딕셔너리
        """

        morning_weather = self.weather[result['after_morning_weather']]
        afternoon_weather = self.weather[result['after_afternoon_weather']]
        morning_temperature = result['after_morning_temperature']
        afternoon_temperature = result['after_afternoon_temperature']
        josa = self.enumerate_josa('는', '도', [morning_weather, afternoon_weather])

        result = {'morning_weather': morning_weather,
                  'afternoon_weather': afternoon_weather,
                  'morning_temperature': morning_temperature,
                  'afternoon_temperature': afternoon_temperature}

        return result, josa

    def edit_specific(self, result: dict) -> dict:
        """
        특정 날짜 딕셔너리를 수정합니다.

        :param result: 입력 딕셔너리
        :return: 수정된 딕셔너리
        """

        weather = self.weather[re.sub(' ', '', result['specific_weather'])]
        temperature = result['specific_temperature']

        result = {'weather': weather,
                  'temperature': temperature}

        return result
