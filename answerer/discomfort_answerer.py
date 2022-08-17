import emotionchat_config as config
import re

class DiscomfortAnswerer:

    def fill_slot(self, entity: list) -> str:
        """
        불,궁 대화일 때 slot filling 함수
        :param entity: 필요한 엔티티 리스트
        :return: 필요한 엔티티를 묻는 챗봇의 답변(질문)
        """

        msg = []
        for e in entity:
            msg += self.entity_question(e)

        return msg

    def entity_question(self, unique_entity: str) -> str:
        """
        엔티티 종류별로 slot filling 질문하는 함수
        :param unique_entity: 필요한 단일 엔티티
        :return: 엔티티 종류별로 slot filling 질문 텍스트
        """
        e = unique_entity
        if e == 'BODY':
            msg = ['어디가 아프신가요? ']
        elif e == 'SYMPTOM':
            msg = ['어떻게 아프신가요? ']
        elif e == 'FOOD':
            msg = ['어떤 음식이 별로세요? ']
        elif e == 'PLACE':
            msg = ['어디로 가고싶으세요? ']
        elif e == 'LOCATION':
            msg = ['어느 지역을 알려드릴까요? ']
        else:
            msg = config.ANSWER['default_error_uncomfort']

        return msg

    def requestAct_check_form(self) -> str:
        """
        활동요구 재질의 출력 포맷
        :return: 출력 메시지
        """
        msg = ['도움이 필요하신거죠? \n']
        msg += config.ANSWER['call_caregiver']

        return msg

    def environmentalDiscomfort_check_form(self) -> str:
        """
        환경 불편 호소 재질의 출력 포맷
        :param place: 장소
        :return: 출력 메시지
        """

        msg = ['많이 불편하신가요? \n']
        msg += config.ANSWER['call_caregiver']

        return msg


    def physicalDiscomfort_check_form(self, body: str, symptom: str) -> str:
        """
        신체 불편 호소 재질의 출력 포맷
        :param body: 신체 부위
        :param symptom: 증상
        :return: 출력 메시지
        """

        msg = []
        if symptom != '':
            msg += [str(symptom + self.yi(symptom) + ' 있으시군요.\n')]
        if body != '':
            msg += [str(body + self.yi(symptom) + ' 많이 아프신가요?\n')]
        msg += config.ANSWER['call_caregiver']

        return msg

    def discomfort_sol_form(self) -> str:
        """
        불편함 호소 해결 출력(간병인 호출) 포맷
        :return: 출력 메시지
        """

        msg = ['많이 힘드셨겠어요. ']
        msg += config.ANSWER['call_caregiver'] # 간병인 불러드릴까요?

        return msg


    def ends_with_jong(self, entity):
        """
        엔티티에 받침이 있는지 알아내는 함수
        :param entity: 엔티티명
        :return: 받침 유무
        """
        m = re.search("[가-힣]+", entity)
        if m:
            k = m.group()[-1]
            return (ord(k) - ord("가")) % 28 > 0
        else:
            return

    def ul(self, entity):
        """
        종성에 따라 조사를 을 혹은 를로 출력해주는 함수
        :param entity: 엔티티명
        :return: 을 or 를
        """
        josa = "을" if self.ends_with_jong(entity) else "를"
        print(f"{entity}{josa} ", end='')
        return josa

    def yi(self, entity):
        """
        종성에 따라 조사를 이 혹은 가로 출력해주는 함수
        :param entity: 엔티티명
        :return: 이 or 가
        """
        josa = "이" if self.ends_with_jong(entity) else "가"
        print(f"{entity}{josa} ", end='')
        return josa

    def wa(self, entity):
        """
        종성에 따라 조사를 와 혹은 과로 출력해주는 함수
        :param entity: 엔티티명
        :return: 와 or 과
        """
        josa = "과" if self.ends_with_jong(entity) else "와"
        print(f"{entity}{josa} ", end='')
        return josa

    def en(self, entity):
        """
        종성에 따라 조사를 은 혹은 는로 출력해주는 함수
        :param entity: 엔티티명
        :return: 은 or 는
        """
        josa = "은" if self.ends_with_jong(entity) else "는"
        print(f"{entity}{josa} ", end='')
        return josa