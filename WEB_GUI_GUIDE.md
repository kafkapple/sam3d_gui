# SAM 3D GUI - 통합 웹 인터페이스 가이드

## 🚀 빠른 시작

### 1. 웹 서버 실행

```bash
cd /home/joon/dev/sam3d_gui
./run.sh
```

### 2. 브라우저로 접근

```
http://localhost:7860
```

원격 접속:
```
http://<서버IP>:7860
```

---

## 🎯 두 가지 작업 모드

### 모드 1: 🚀 Quick Mode (빠른 자동 처리)

**사용 시나리오**: 빠르게 여러 비디오의 모션 감지 및 객체 추출

**워크플로우**:
1. **📂 비디오 스캔** 클릭
2. 드롭다운에서 비디오 선택
3. 파라미터 설정:
   - 시작 시간: 0.0초
   - 길이: 3.0초
   - 모션 임계값: 50 픽셀
   - 세그멘테이션: contour (권장)
4. **🚀 자동 처리** 클릭
5. 결과 확인: 모션 감지 여부, 변위 통계, 객체 정보

**장점**:
- ⚡ 빠름 (30초 이내)
- 🔄 여러 비디오 연속 처리 가능
- 📊 자동 모션 감지 및 통계

**단점**:
- 정확도가 Interactive Mode보다 낮을 수 있음
- 복잡한 배경에서는 세그멘테이션 품질이 떨어질 수 있음

---

### 모드 2: 🎨 Interactive Mode (대화형 Annotation)

**사용 시나리오**: 정확한 세그멘테이션이 필요하거나, 3D mesh 생성할 때

**워크플로우**:
1. **📹 비디오 로드**:
   - 데이터 디렉토리 입력
   - 비디오 파일 (상대 경로) 입력
   - 시작/길이 설정
   - 로드 버튼 클릭

2. **🎯 Point Annotation**:
   - Point 타입 선택: **Foreground** (객체)
   - 이미지에서 객체 위치 클릭 (여러 점 가능)
   - (선택) **Background** 선택하여 배경 클릭

3. **✂️ Segment Current Frame**:
   - 현재 프레임 세그멘테이션
   - 결과 확인 (녹색 마스크)

4. **🔄 Propagate to All Frames**:
   - 전체 비디오 tracking
   - 각 프레임에 마스크 자동 생성

5. **💾 저장**:
   - **🎲 Generate 3D Mesh**: PLY 파일 생성 및 다운로드
   - **💾 Save Masks**: 모든 프레임 마스크 PNG로 저장

**장점**:
- ✅ 정확한 세그멘테이션
- ✅ 복잡한 배경 처리 가능
- ✅ 3D mesh 생성
- ✅ 프레임별 마스크 저장

**단점**:
- 🕐 시간이 더 걸림 (1-2분)
- 👆 수동 점 클릭 필요

---

## 📊 모드 비교

| 특징 | Quick Mode | Interactive Mode |
|------|-----------|------------------|
| **속도** | 🚀 빠름 (30초) | 🕐 보통 (1-2분) |
| **정확도** | ⭐⭐⭐ 좋음 | ⭐⭐⭐⭐⭐ 매우 좋음 |
| **사용 편의성** | ✅ 자동 | 👆 클릭 필요 |
| **모션 감지** | ✅ 지원 | ❌ 미지원 |
| **3D mesh** | ❌ 미지원 | ✅ 지원 |
| **마스크 저장** | ❌ 미지원 | ✅ 지원 |
| **복잡한 배경** | ⚠️ 어려울 수 있음 | ✅ 처리 가능 |

**권장 사용법**:
- **대량 스크리닝**: Quick Mode로 모션 있는 구간 찾기
- **정밀 분석**: Interactive Mode로 정확한 세그멘테이션 & 3D 생성

---

## 🎬 실전 사용 예시

### 예시 1: 마우스 행동 분석

**목표**: 30초 비디오에서 마우스가 움직이는 모든 구간 찾기

**방법**:
1. Quick Mode 사용
2. 3초씩 나눠서 10번 처리 (0-3s, 3-6s, ..., 27-30s)
3. 모션 감지된 구간만 기록
4. 결과: 모션이 감지된 구간 리스트

### 예시 2: 특정 프레임 3D 모델링

**목표**: 마우스가 정지해 있는 프레임의 3D mesh 생성

**방법**:
1. Quick Mode로 정지 구간 찾기 (모션 감지 안 됨)
2. Interactive Mode로 해당 구간 로드
3. 마우스 위치 클릭 (foreground)
4. Segment Current Frame
5. Generate 3D Mesh
6. PLY 파일 다운로드 → MeshLab으로 열기

### 예시 3: 전체 비디오 마스크 추출

**목표**: 비디오 전체 프레임의 객체 마스크 PNG 저장

**방법**:
1. Interactive Mode로 비디오 로드
2. 첫 프레임에 점 annotation
3. Segment Current Frame
4. Propagate to All Frames
5. Save Masks 클릭
6. `outputs/masks/` 폴더에 마스크 저장됨

---

## 🔧 고급 기능

### Gradio Share (공개 링크)

`src/web_app.py` 수정:
```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,  # ← 이것을 True로
    debug=True
)
```

실행하면 `https://xxxxx.gradio.live` 공개 URL 생성

### 백그라운드 실행

```bash
nohup ./run.sh > web_gui.log 2>&1 &

# 로그 확인
tail -f web_gui.log

# 종료
pkill -f web_app.py
```

---

## 🐛 문제 해결

### "Address already in use"

```bash
lsof -i :7860
kill -9 <PID>
```

### 원격 접속 안 됨

```bash
# 방화벽 열기
sudo ufw allow 7860/tcp

# 서버 IP 확인
hostname -I
```

### 처리 중 멈춤

- 브라우저 콘솔 (F12) 확인
- 터미널 로그 확인
- 비디오 길이를 1-2초로 줄여서 테스트

---

## 📚 참고

- **QUICKSTART.md**: 5분 빠른 시작
- **README.md**: 전체 문서
- **Gradio 문서**: https://gradio.app/docs/

---

**작성일**: 2025-11-23
**버전**: 2.0 (통합)
**상태**: ✅ 즉시 사용 가능
