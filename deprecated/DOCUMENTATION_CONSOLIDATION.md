# 문서 통합 계획 (Documentation Consolidation Plan)

## 현재 상태 분석

### 문서 목록 (총 12개)

| 파일 | 크기 | 위치 | 상태 | 액션 |
|------|------|------|------|------|
| **README.md** | 29K | 루트 | ✅ 최신 | **유지** (메인 문서) |
| **QUICK_START.md** | 4.9K | 루트 | ✅ 최신 | **유지** (빠른 참조) |
| QUICKSTART.md | 5.8K | 루트 | ⚠️ 중복 | **삭제** |
| **ARCHITECTURE.md** | 19K | 루트 | ⚠️ 오래됨 | **이동** → docs/ |
| **IMPLEMENTATION_SUMMARY.md** | 11K | 루트 | ⚠️ 오래됨 | **병합** → README |
| **PROJECT_SUMMARY.md** | 15K | 루트 | ⚠️ 오래됨 | **병합** → README |
| **README_CHECKPOINTS.md** | 5.1K | 루트 | ⚠️ 오래됨 | **병합** → DEPLOYMENT |
| **UPDATES_LOG.md** | 9.6K | 루트 | ⚠️ 오래됨 | **병합** → CHANGELOG |
| **WEB_GUI_GUIDE.md** | 4.9K | 루트 | ⚠️ 중복 | **삭제** |
| docs/DEPLOYMENT.md | 13K | docs/ | ✅ 최신 | **유지** |
| docs/SESSION_MANAGEMENT.md | 8.6K | docs/ | ✅ 최신 | **유지** |
| docs/COMPARISON_SAM_ANNOTATORS.md | 14K | docs/ | ✅ 최신 | **유지** |

---

## 통합 후 최종 구조

```
sam3d_gui/
├── README.md                           # ⭐ 메인 문서 (프로젝트 개요 + 사용법)
├── QUICK_START.md                      # ⭐ 빠른 참조 (서버 실행/종료)
├── CHANGELOG.md                        # ⭐ 변경 이력 (UPDATES_LOG 통합)
│
└── docs/
    ├── README.md                       # 📋 문서 인덱스
    ├── DEPLOYMENT.md                   # 🚢 배포 가이드 (HF 인증 포함)
    ├── SESSION_MANAGEMENT.md           # 💾 세션 관리 가이드
    ├── COMPARISON_SAM_ANNOTATORS.md    # 📊 Annotator 비교 분석
    ├── ARCHITECTURE.md                 # 🏗️ 시스템 아키텍처
    └── API_REFERENCE.md                # 📚 API 레퍼런스 (신규)
```

---

## 통합 계획 상세

### Phase 1: 중복 제거

#### 1.1 QUICKSTART.md vs QUICK_START.md

**분석**:
- QUICKSTART.md: 5.8K (구버전)
- QUICK_START.md: 4.9K (최신, HF 인증 포함)

**결정**: **QUICKSTART.md 삭제**
```bash
git rm QUICKSTART.md
```

#### 1.2 WEB_GUI_GUIDE.md

**분석**:
- 내용이 README.md 및 QUICK_START.md와 중복
- 오래된 정보 (Interactive Mode 기본 탭 변경 전)

**결정**: **삭제**
```bash
git rm WEB_GUI_GUIDE.md
```

---

### Phase 2: 이동 및 병합

#### 2.1 ARCHITECTURE.md → docs/ARCHITECTURE.md

**이유**:
- 기술 문서는 docs/ 하위에 위치
- 루트는 사용자 대상 문서만

**액션**:
```bash
git mv ARCHITECTURE.md docs/ARCHITECTURE.md
```

**업데이트 필요**:
- README.md에서 링크 수정
- docs/README.md에 추가

---

#### 2.2 README_CHECKPOINTS.md → docs/DEPLOYMENT.md

**현재 내용**:
- SAM 2 체크포인트 다운로드
- SAM 3D 체크포인트 설정
- Config 경로 관리

**통합 방법**:
- DEPLOYMENT.md의 "SAM 3D 체크포인트 다운로드" 섹션에 SAM 2 내용 추가
- 체크포인트 관리 통합 섹션 생성

**액션**:
```bash
# 내용 병합 후
git rm README_CHECKPOINTS.md
```

---

#### 2.3 IMPLEMENTATION_SUMMARY.md + PROJECT_SUMMARY.md → README.md

**현재 문제**:
- IMPLEMENTATION_SUMMARY: 구현 세부사항 (한국어)
- PROJECT_SUMMARY: 프로젝트 개요 (한국어)
- README.md: 이미 포괄적인 내용 (영어 + 일부 한국어)

**통합 방법**:
- 유용한 섹션만 README.md에 추가
- 중복 제거
- 나머지는 CHANGELOG 또는 삭제

**액션**:
```bash
# 내용 검토 후 선별적 통합
git rm IMPLEMENTATION_SUMMARY.md PROJECT_SUMMARY.md
```

---

#### 2.4 UPDATES_LOG.md → CHANGELOG.md

**현재 내용**:
- 날짜별 업데이트 로그
- 버그 수정 내역
- 기능 추가 내역

**변환 방법**:
- Conventional Changelog 형식으로 변환
- 버전별로 그룹화

**액션**:
```bash
# CHANGELOG.md 생성 후
git rm UPDATES_LOG.md
git add CHANGELOG.md
```

---

### Phase 3: 신규 문서 생성

#### 3.1 docs/README.md (문서 인덱스)

**목적**: 모든 문서의 개요 및 링크 제공

```markdown
# SAM 3D GUI - Documentation

## 📚 문서 개요

### 사용자 가이드
- [README.md](../README.md) - 프로젝트 개요 및 전체 사용법
- [QUICK_START.md](../QUICK_START.md) - 서버 실행/종료 빠른 참조

### 배포 및 설정
- [DEPLOYMENT.md](DEPLOYMENT.md) - 배포 가이드 (HF 인증, Git LFS)
- [SESSION_MANAGEMENT.md](SESSION_MANAGEMENT.md) - 세션 저장/로드

### 개발자 문서
- [ARCHITECTURE.md](ARCHITECTURE.md) - 시스템 아키텍처
- [COMPARISON_SAM_ANNOTATORS.md](COMPARISON_SAM_ANNOTATORS.md) - Annotator 비교
- [API_REFERENCE.md](API_REFERENCE.md) - API 문서 (신규)

### 기타
- [CHANGELOG.md](../CHANGELOG.md) - 변경 이력
```

---

#### 3.2 docs/API_REFERENCE.md

**목적**: 프로그래밍 인터페이스 문서화

**포함 내용**:
- `SAMInteractiveWebApp` 클래스 API
- 주요 함수 시그니처
- 사용 예제
- Config 스키마

---

### Phase 4: 링크 업데이트

모든 문서에서 변경된 파일 경로에 대한 링크 업데이트

**변경 사항**:
```markdown
# Before
[Architecture](ARCHITECTURE.md)
[Checkpoints](README_CHECKPOINTS.md)

# After
[Architecture](docs/ARCHITECTURE.md)
[Deployment](docs/DEPLOYMENT.md)
```

---

## 실행 계획

### Step 1: 백업
```bash
git branch backup-docs
```

### Step 2: 중복 제거
```bash
git rm QUICKSTART.md WEB_GUI_GUIDE.md
git commit -m "docs: Remove duplicate documentation files"
```

### Step 3: 이동
```bash
git mv ARCHITECTURE.md docs/ARCHITECTURE.md
git commit -m "docs: Move ARCHITECTURE.md to docs/"
```

### Step 4: 통합
```bash
# README_CHECKPOINTS.md → DEPLOYMENT.md
# UPDATES_LOG.md → CHANGELOG.md
# IMPLEMENTATION_SUMMARY.md, PROJECT_SUMMARY.md → README.md

git rm README_CHECKPOINTS.md UPDATES_LOG.md IMPLEMENTATION_SUMMARY.md PROJECT_SUMMARY.md
git add docs/DEPLOYMENT.md CHANGELOG.md README.md
git commit -m "docs: Consolidate documentation into main files"
```

### Step 5: 신규 생성
```bash
# docs/README.md
# docs/API_REFERENCE.md

git add docs/README.md docs/API_REFERENCE.md
git commit -m "docs: Add documentation index and API reference"
```

### Step 6: 링크 업데이트
```bash
# 모든 문서에서 링크 수정
git add .
git commit -m "docs: Update internal documentation links"
```

---

## 최종 검증 체크리스트

- [ ] 모든 링크가 정상 작동 (broken link 없음)
- [ ] README.md가 포괄적이고 최신 정보 포함
- [ ] docs/ 하위 문서들이 체계적으로 구성됨
- [ ] 중복 제거 완료
- [ ] CHANGELOG.md가 버전별로 정리됨
- [ ] 문서 인덱스 (docs/README.md) 생성됨

---

## 유지보수 가이드

### 문서 작성 규칙

**루트 디렉토리 (`sam3d_gui/`)**:
- ✅ README.md: 프로젝트 개요
- ✅ QUICK_START.md: 빠른 참조
- ✅ CHANGELOG.md: 변경 이력
- ✅ LICENSE: 라이선스
- ❌ 기타 세부 문서 금지 → docs/로 이동

**docs/ 디렉토리**:
- ✅ 배포 가이드
- ✅ 개발자 문서
- ✅ 아키텍처 설명
- ✅ API 레퍼런스
- ✅ 비교 분석
- ✅ 상세 사용법

### 신규 문서 추가 시

1. **사용자 대상**: 루트 또는 docs/ (링크만 README에)
2. **개발자 대상**: docs/
3. **변경 이력**: CHANGELOG.md에 추가

### 문서 업데이트 시

1. **날짜 표시**: 문서 하단에 "최종 업데이트: YYYY-MM-DD"
2. **버전 표시**: 해당되면 버전 번호
3. **링크 검증**: 변경된 파일 경로 확인

---

**작성일**: 2025-11-24  
**버전**: 1.0  
**상태**: ✅ 계획 수립 완료
