# 권취 가상 실험실

3D 권취 재생, 원본/작업본 제어 코드 비교, 스윙·복귀 설정 실험 도구입니다.

## 새 PC에서 실행

Python 3.10 이상을 설치한 뒤:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe launch.py
```

주소: http://127.0.0.1:18765/

이 저장소는 실행 가능한 Python 제어 소스만 포함한 배포용 복사본입니다.
`original/manifest.json`은 이 저장소에 포함된 기준 소스의 해시이며 원본 ZIP 전체의 검증 목록이 아닙니다.
원래 전체 원본·작업본·실행 기록은 기존 PC에 보존했습니다. 체크포인트 암호는 제거하고
`WINDING_CHECKPOINT_PASSWORD` 환경변수로 대체했습니다.
영상 AI 기능에는 별도의 승인된 Saige 실행 환경·체크포인트가 필요합니다.
3D 및 상태 전이 실험은 영상 AI 없이 사용합니다. 실제 설비 보정이 된 디지털 트윈은 아닙니다.
모델 가중치·사내 영상·vendor 바이너리·로그·결과·백업은 포함하지 않습니다.

## 검증

```powershell
.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py" -v
```

자세한 사용법은 [SIMULATION_GUIDE.md](SIMULATION_GUIDE.md)를 참고하세요.
