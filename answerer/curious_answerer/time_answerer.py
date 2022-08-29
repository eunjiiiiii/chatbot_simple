from datetime import datetime


class TimeAnswerer:

    def now_time_form(self, date):
        """
        현재 시간 출력 포맷

        :return : 현재 시간(오전/오후, 시, 분)
        """
        now = datetime.now()

        msg = ['']
        msg[0] = '현재 시간은 {AMorPM} {hour}시 {minute}분입니다.'.format(AMorPM=self.__isAMPM(now.hour)[0],
                                                                hour=self.__isAMPM(now.hour)[1], minute=now.minute)

        return msg

    def __isAMPM(self, hour_):
        """
        24시간제를 12시간제로 변경해주는 함수
        """

        AMPM = ''
        if hour_ == 12:
            AMPM = '오후'
            hour = hour_
        elif 12 < hour_ < 24:
            AMPM = '오후'
            hour = hour_ - 12
        else:
            AMPM = '오전'
            hour = hour_

        return [AMPM, hour]



    def now_date_form(self, date):
        """
        현재 날짜 출력 포맷

        :return : 현재 날짜(년, 월, 일)
        """
        now = datetime.now()

        date = date.replace('은는이가을를','')

        msg = ['']
        if any(x in date for x in ['오늘', '현재', '지금']):
            msg[0] = '오늘은 {year}년 {month}월 {day}일 입니다.'.format(year=now.year, month=now.month, day=now.day)
        elif any(x in date for x in ['어제', '어저께', '하루전','전날']):
            msg[0] = '어제는 {year}년 {month}월 {day}일 이었어요.'.format(year=now.year, month=now.month, day=int(now.day-1))
        elif any(x in date for x in ['그제', '그저께', '이틀전']):
            msg[0] = '그저께는 {year}년 {month}월 {day}일 이었어요.'.format(year=now.year, month=now.month, day=int(now.day - 2))
        else:
            msg[0] = '그 날짜는 알수가 없어요.'
        return msg

    def now_weekday_form(self, date):
        """
        현재 요일 출력 포맷

        :return : 현재 요일(요일)
        """

        now = datetime.now()

        msg = ['']
        msg[0] = '오늘은 {weekday} 입니다.'.format(weekday=self.__integer2weekday(now.weekday()))

        if date in ['오늘', '오늘은', '현재']:
            msg[0] = '오늘은 {weekday} 입니다.'.format(weekday=self.__integer2weekday(now.weekday()))
        elif date in ['어제','어젠','어제는']:
            msg[0] = '어제는 {weekday} 이었어요.'.format(weekday=self.__integer2weekday(now.weekday()-1))
        elif date in ['그저께', '그제']:
            msg[0] = '그저께는 {weekday} 이었어요.'.format(weekday=self.__integer2weekday(now.weekday()-2))
        else:
            msg[0] = '그 요일은 알수가 없어요.'

        return msg

    def __integer2weekday(self, weekday):
        """
        datetime.now().weekday()의 값(int)을 문자열(ex. 월요일)로 바꿔주는 함수

        :param weekday: 요일(int)
        :return : 요일(str, ex. 월요일)
        """

        weekdays = ['월', '화', '수', '목', '금', '토', '일']
        res = weekdays[weekday] + '요일'

        return res
