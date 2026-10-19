"""Reproducible local experiments; none are measured factory fault recipes."""
SCENARIOS = [
    dict(id='reference', label='기준 조건', description='문서의 호기별 운영값 + 기존 기계 가정. 정상 품질을 보장하는 값은 아닙니다.',
         plant={}, motion={}),
    dict(id='delay', label='이동 지연', description='신호 지연을 3초로 늘려 끝단에 더 머무는지 봅니다.',
         plant=dict(delay=3), motion={}),
    dict(id='coast', label='정지 후 관성', description='가감속 응답 2.5초, 이동속도 25mm/s. OFF 후 잔류 이동을 확인합니다.',
         plant=dict(inertia=2.5, speed=25), motion={}),
    dict(id='slow', label='이동속도 부족', description='대차를 3mm/s로 낮춥니다. 6rpm·60mm 케이블의 이론 정렬 속도는 6mm/s입니다.',
         plant=dict(speed=3), motion={}),
    dict(id='fast', label='회전속도 과다', description='드럼을 18rpm으로 높이고 이송은 유지합니다. 감김과 보정의 시간 차이를 봅니다.',
         plant=dict(rpm=18), motion={}),
    dict(id='late_return', label='복귀 대기각 확대', description='실제 코드의 복귀 대기각을 좌우 180°로 변경합니다. 조기 복귀 조건이 먼저 충족되면 대기각을 늘려도 결과가 같을 수 있습니다.',
         plant={}, motion=dict(hold_left=180, hold_right=180)),
    dict(id='short_pulse', label='짧은 신호 미응답', description='가정: 요청 시점부터 ON 시간을 세고 5초 후 구동합니다. 3·4초 일반 신호와 1.1초 스윙 신호가 모두 지연보다 짧아 미응답합니다.',
         plant=dict(delay=5, relay_mode='on_delay'), motion={}),
    dict(id='loadcell', label='로드셀·언더롤러 간섭', description='접촉 보정이 AI 명령보다 우선한다고 가정합니다. 반대 방향 명령이 겹치는 시간을 봅니다.',
         plant=dict(loadcell='underroller', delay=1.8, inertia=1, speed=20), motion={}),
    dict(id='robot', label='로봇 보정 대안', description='간섭 시험과 같은 조건에서 접촉 보정 대상만 로봇 손으로 바꿉니다.',
         plant=dict(loadcell='robot', delay=1.8, inertia=1, speed=20), motion={}),
]


def catalog(defaults):
    # Short endpoint experiments preserve the user's dimensions and selected site.
    response = {key: defaults[key] for key in
                ('rpm', 'speed', 'delay', 'inertia', 'response', 'ppm', 'calls', 'loadcell', 'relay_mode')}
    return dict(scenarios=SCENARIOS, common={**response, 'start': 'right', 'seconds': 240,
                                          'operating': 'handover'},
                provenance='운영시간·판단 조건: 인계 문서/원본 코드. 치수·기계 응답·로드셀: 미교정 가정.')
