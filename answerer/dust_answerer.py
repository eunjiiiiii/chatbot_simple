from answerer.base_answerer import BaseAnswerer


class DustAnswerer(BaseAnswerer):

    def morning_afternoon_form(self, location: str, date: str, result: dict, josa: list) -> str:
        """
        오전-오후 미세먼지 출력 포맷

        :param location: 지역
        :param date: 날짜
        :param josa: 조사 리스트
        :param result: 데이터 딕셔너리
        :return: 출력 메시지
        """

        msg = ['','','','']
        msg[0] = self.dust_init.format(location=location)
        msg[1] =  '{date} 오전 미세먼지 상태{j0} {morning_fine_dust} 오후 상태{j1} {afternoon_fine_dust} '.format(date=date, j0=josa[0], j1=josa[1],morning_fine_dust=result['morning_fine_dust'],afternoon_fine_dust=result['afternoon_fine_dust'])
        msg[2] = '오전 초미세먼지 상태{j2} {morning_ultra_dust} 오후 상태{j3} {afternoon_ultra_dust} '.format(j2=josa[2],j3=josa[3],morning_ultra_dust=result['morning_ultra_dust'], afternoon_ultra_dust=result['afternoon_ultra_dust'])
        msg[3] = '오전 대기중 오존 상태{j4} {morning_ozon} 오후 상태{j5} {afternoon_ozon} '.format(j4=josa[4],j5=josa[5], morning_ozon=result['morning_ozon'],afternoon_ozon=result['afternoon_ozon'])

        return msg

    def single_form(self, location: str, date: str, result: dict, josa: list) -> str:
        """
        싱글 (오전-오후 없는) 미세먼지 출력 포맷

        :param location: 지역
        :param date: 날짜
        :param josa: 조사 리스트
        :param result: 데이터 딕셔너리
        :return: 출력 메시지
        """

        msg = ['', '', '', '']
        msg[0] = self.dust_init.format(location=location)
        msg[1] = '{date} 미세먼지 상태{j0} {today_fine_dust} '.format(date=date,j0=josa[0], today_fine_dust=result['today_fine_dust'])
        msg[2] = '초미세먼지 상태{j1} {today_ultra_dust} '.format(j1=josa[1], today_ultra_dust=result['today_ultra_dust'])
        msg[3] = '대기중 오존 상태{j2} {today_ozon} '.format( j2=josa[2], today_ozon=result['today_ozon'])

        return msg
